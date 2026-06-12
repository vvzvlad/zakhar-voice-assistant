"""Admin-panel HTTP API (aiohttp).

Runs in a TRUSTED zone — there is NO authentication. The panel is driven entirely
by ConfigService, so it carries no provider-specific knowledge: it exposes the
catalog, the raw config document, generic patch/options endpoints, the named
system-prompt profiles, system/version info and live device status. It also
serves the agent-facing MCP endpoint at /mcp (an aiohttp→ASGI bridge into the
FastMCP streamable-HTTP session manager, see src/agent_mcp.AgentMcpEndpoint);
the `core.agent_mcp.enabled` toggle is read live per request.

Start/stop mirror AudioServer (AppRunner + TCPSite). CORS is restricted to the
hardcoded `_ALLOWED_ORIGINS` allowlist (empty — see below); it never reflects a
wildcard, so an arbitrary web page can't read the unauthenticated config (which
carries plaintext secrets) via the operator's browser. The prod frontend is
same-origin and the dev Vite server proxies /api, so neither needs CORS.
"""

import asyncio
import contextlib
import inspect
import io
import json
import os
import time
import wave
from datetime import datetime, timezone

import httpx
import numpy as np
from aiohttp import web
from loguru import logger
from multidict import CIMultiDict
from pydantic import ValidationError

from src import config_store
from src.capture_jobs import CaptureJobManager
from src.plugins.base import get_provider
from src.pipeline import CAPTURE_MAX_SECONDS, CaptureBusyError
from src.runs_store import db_file_size

# HTTP methods the API exposes (used by the CORS preflight headers).
_ALLOW_METHODS = "GET, POST, PATCH, PUT, DELETE, OPTIONS"

# Hop-by-hop headers (RFC 9110 §7.6.1) the /mcp ASGI bridge must not copy from the
# ASGI response into the aiohttp response — the aiohttp transport owns them.
_HOP_BY_HOP_HEADERS = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})

# Browser origins allowed cross-origin access to the (unauthenticated) panel API.
# Hardcoded empty: there is no cross-origin reflection. The prod frontend is
# same-origin and the dev Vite server proxies /api, so neither needs CORS — and an
# empty allowlist guarantees the unauthenticated config (with plaintext secrets) is
# never readable from an arbitrary web page. Not a config knob.
_ALLOWED_ORIGINS: frozenset[str] = frozenset()

# Bundled end-of-phrase chime clips live under <repo-root>/assets/chimes. Resolve the
# directory relative to the repo root (this file is in src/) so the listing works
# regardless of the process CWD; the offered values are repo-root-relative POSIX paths
# (e.g. "assets/chimes/Chime_SFX_001.wav") so they match how the pipeline opens
# core.ack.sound_path (the app runs from the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHIME_EXTS = (".wav", ".mp3", ".flac")


def list_chimes(root: str = _REPO_ROOT) -> list[str]:
    """Return sorted repo-root-relative paths of the bundled chime audio files.

    Scans <root>/assets/chimes for audio files (wav/mp3/flac). Returns an empty list
    when the directory is missing or unreadable."""
    abs_dir = os.path.join(root, "assets", "chimes")
    try:
        names = os.listdir(abs_dir)
    except OSError:
        return []
    files = sorted(
        n for n in names
        if n.lower().endswith(_CHIME_EXTS) and os.path.isfile(os.path.join(abs_dir, n))
    )
    return [f"assets/chimes/{n}" for n in files]


def _add_cors(resp: web.StreamResponse, request: web.Request, allowed_origins) -> web.StreamResponse:
    """Reflect an allowlisted request Origin into CORS headers (never a wildcard).

    Same-origin requests (prod static serving, the dev Vite proxy) need no CORS
    headers; cross-origin browser access is granted only to origins in the
    hardcoded `_ALLOWED_ORIGINS` allowlist (currently empty).
    """
    origin = request.headers.get("Origin")
    if origin and origin in allowed_origins:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"] = _ALLOW_METHODS
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Vary"] = "Origin"
    return resp


def _make_cors_middleware(allowed_origins):
    """Build the CORS middleware bound to a fixed allowlist (set of origin strings)."""
    @web.middleware
    async def _cors_middleware(request: web.Request, handler):
        # CORS headers are attached to BOTH normal responses and raised HTTP
        # exceptions so allowlisted cross-origin clients can read error bodies.
        if request.method == "OPTIONS":
            return _add_cors(web.Response(status=204), request, allowed_origins)
        # WebSocket upgrades: headers are already sent; CORS doesn't gate WS. Pass through.
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await handler(request)
        try:
            return _add_cors(await handler(request), request, allowed_origins)
        except web.HTTPException as ex:
            _add_cors(ex, request, allowed_origins)
            raise
    return _cors_middleware


def _extract_wav_channel(wav: bytes, idx: int) -> bytes | None:
    """Extract one channel of a 16-bit PCM WAV as a standalone mono WAV.

    Returns the original bytes unchanged when the file is already mono and
    channel 0 is requested; None when the requested channel does not exist or
    the input is not a parseable WAV. Used by the run-audio endpoint to serve
    the STT (left) and raw (right) channels of a stored stereo utterance.
    """
    try:
        with wave.open(io.BytesIO(wav)) as src:
            nch = src.getnchannels()
            if idx >= nch:
                return None  # no such channel
            if nch == 1:
                return wav  # already mono, serve as is
            framerate = src.getframerate()
            frames = src.readframes(src.getnframes())
    except (wave.Error, EOFError):
        return None
    # Deinterleave: int16 samples are laid out [ch0, ch1, ...] per frame. Trim a
    # trailing partial frame so the reshape is always valid.
    usable = len(frames) - len(frames) % (2 * nch)
    mono = np.frombuffer(frames[:usable], dtype="<i2").reshape(-1, nch)[:, idx]
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(framerate)
        dst.writeframes(mono.tobytes())
    return out.getvalue()


async def _read_json(request: web.Request):
    """Parse a JSON request body, raising HTTP 400 on malformed input."""
    try:
        return await request.json()
    except Exception:
        raise web.HTTPBadRequest(text='{"error": "invalid JSON body"}',
                                 content_type="application/json")


class PanelServer:
    def __init__(self, svc, host, port, *, version, started_at,
                 device_status=None, device_capture=None, device_play=None,
                 device_controls_get=None, device_controls_set=None,
                 static_dir=None, runs_store=None, tool_sources=None,
                 run_events=None, prompt_store=None, agent_mcp=None,
                 heartbeat_interval: float = 1.0):
        # svc: ConfigService; started_at: float (time.time())
        # device_status: optional callable -> list[dict]; static_dir: optional path to built frontend
        # device_capture: optional async callable (device_name, seconds) -> bytes that records a
        #   manual sample and returns it as WAV bytes (DeviceManager.capture). It is run by a
        #   CaptureJobManager as a background task (so a browser disconnect can't cancel it and
        #   reboot the device); the capture endpoints poll/download that job. None -> the
        #   manager is not built and every /api/capture* endpoint returns 503.
        # runs_store: optional RunsStore for the observability endpoints (None -> empty/zeros)
        # tool_sources: optional zero-arg callable -> list[dict] (ToolHub.describe()), None -> []
        # run_events: optional RunEventsHub for the live WS run stream (None -> WS closes)
        # prompt_store: optional PromptStore (named system-prompt profiles); None ->
        #   every /api/prompt* endpoint returns 503
        # agent_mcp: optional AgentMcpEndpoint (the agent-facing MCP server bridged
        #   at /mcp); None -> /mcp returns 503 (API-only / boot-failure tolerant,
        #   mirroring runs_store=None). Its start/stop lifecycle is owned by this
        #   server's start()/stop().
        self.svc = svc
        self.host = host
        self.port = port
        self.version = version
        self.started_at = started_at
        self.device_status = device_status
        self.device_capture = device_capture
        # device_play: optional async callable (sound_path, device_name=None) -> dict that
        #   plays the preview chime on the speaker(s) (DeviceManager.play_chime). None ->
        #   the /api/chimes/play endpoint returns 503.
        self.device_play = device_play
        # device_controls_get: optional SYNC callable (device_name) -> dict snapshot of the
        #   live device controls (cutoff %, volume %) (DeviceManager.device_controls).
        # device_controls_set: optional SYNC callable (device_name, control_id, value) -> dict
        #   that sets one control and returns the refreshed snapshot
        #   (DeviceManager.set_device_control). For both: None -> the /api/device/controls
        #   endpoints return 503.
        self.device_controls_get = device_controls_get
        self.device_controls_set = device_controls_set
        # Background capture jobs: the recording runs as a server-side task so a
        # browser disconnect no longer cancels it (which used to reboot the device).
        self._capture_jobs = CaptureJobManager(device_capture) if device_capture is not None else None
        self.static_dir = static_dir
        self.runs_store = runs_store
        self.tool_sources = tool_sources
        self.run_events = run_events
        self.prompt_store = prompt_store
        self.agent_mcp = agent_mcp
        # Period (seconds) of the WS system heartbeat used as the panel liveness signal.
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task = None
        self._runner: web.AppRunner | None = None

    # --- handlers ------------------------------------------------------------
    async def _get_catalog(self, request: web.Request) -> web.Response:
        return web.json_response(self.svc.catalog())

    async def _get_config(self, request: web.Request) -> web.Response:
        return web.json_response(self.svc.document())

    async def _patch_config(self, request: web.Request) -> web.Response:
        patch = await _read_json(request)
        if not isinstance(patch, dict):
            return web.json_response(
                {"error": "body must be a JSON object"}, status=400
            )
        try:
            self.svc.apply(patch)
        except ValidationError as e:
            return web.json_response(
                {"error": str(e), "detail": e.errors()}, status=422
            )
        except ValueError as e:
            return web.json_response({"error": str(e), "detail": []}, status=422)
        return web.json_response(self.svc.document())

    async def _get_options(self, request: web.Request) -> web.Response:
        category = request.query.get("category")
        plugin = request.query.get("plugin")
        field = request.query.get("field")
        if not (category and plugin and field):
            return web.json_response(
                {"error": "category, plugin and field are required"}, status=400
            )
        # Optional user-typed search string for providers with server-side
        # catalog search (fields marked "search": "remote").
        q = (request.query.get("q") or "").strip()
        try:
            # ValueError (unknown plugin) is raised by get_provider BEFORE any
            # coroutine is created, so nothing is left un-awaited on that path.
            options = self.svc.options(category, plugin, field, q)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=404)
        # Network-backed lists (provider model catalogs) come back as an awaitable.
        if inspect.isawaitable(options):
            try:
                options = await options
            except httpx.HTTPError as e:
                return web.json_response(
                    {"error": f"upstream fetch failed: {e}"}, status=502
                )
        return web.json_response({"options": options or []})

    async def _get_chimes(self, request: web.Request) -> web.Response:
        """List the bundled end-of-phrase chime files for the ack sound_path selector."""
        return web.json_response({"options": list_chimes()})

    async def _post_play_chime(self, request: web.Request) -> web.Response:
        """Play the selected end-of-phrase chime on the speaker(s) for an operator preview.

        Body: {"sound_path": "<path or ''>", "device": "<name>"?}. sound_path must be empty
        (synthesized chime) or one of the bundled clips from GET /api/chimes — any other
        value is rejected (no arbitrary file read / traversal). Plays on the named device,
        or on every online speaker when "device" is omitted. Returns
        {"played": [...], "offline": [...]}.

        Status codes: 200 ok, 400 bad input / unknown sound_path, 404 unknown device,
        503 device playback not wired (API-only boot).
        """
        if self.device_play is None:
            return web.json_response({"error": "device playback not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        sound_path = body.get("sound_path", "")
        if not isinstance(sound_path, str):
            return web.json_response({"error": 'field "sound_path" must be a string'}, status=400)
        sound_path = sound_path.strip()
        # Defense-in-depth: only the synthesized default ("") or a known bundled clip may be
        # played — never an arbitrary path.
        if sound_path and sound_path not in list_chimes():
            return web.json_response({"error": "unknown chime sound_path"}, status=400)
        device = body.get("device")
        if device is not None and (not isinstance(device, str) or not device):
            return web.json_response({"error": 'field "device" must be a non-empty string'}, status=400)
        try:
            result = await self.device_play(sound_path, device)
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.json_response(result)

    async def _post_tts_test(self, request: web.Request) -> web.Response:
        """Synthesize a short test phrase with caller-supplied TTS settings and return
        the audio for in-browser playback.

        Body: {"provider": "<tts plugin id>", "settings": {...}, "text": "..."}. The
        settings are the panel's CURRENT form draft (possibly unsaved): the backend is
        built ad hoc (ConfigService.create_adhoc) — the stored config and the running
        pipeline are never touched. Provider-agnostic: the draft dict is validated by
        the provider's own ConfigModel, so any TTS plugin works with zero per-provider
        code here.

        Status codes: 200 audio (the provider's native format), 400 bad input,
        404 unknown provider, 422 invalid settings / empty or too-long text /
        nothing to synthesize, 502 upstream synthesis failure.
        """
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        provider = body.get("provider")
        if not isinstance(provider, str) or not provider:
            return web.json_response(
                {"error": 'field "provider" must be a non-empty string'}, status=400
            )
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response(
                {"error": 'field "text" must be a non-empty string'}, status=422
            )
        text = text.strip()
        # A voice test is a short phrase, and cloud TTS bills per character.
        if len(text) > 500:
            return web.json_response({"error": "text too long (max 500 chars)"}, status=422)
        settings = body.get("settings") or {}
        if not isinstance(settings, dict):
            return web.json_response({"error": 'field "settings" must be an object'}, status=400)
        # Resolve the plugin id first: an unknown provider is a 404, distinct from
        # the 422 a backend raises for bad settings (both surface as ValueError).
        try:
            get_provider("tts", provider)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=404)
        try:
            # to_thread is REQUIRED: provider create() is cheap for the cloud bricks,
            # but the piper brick loads an ONNX model synchronously (PiperVoice.load),
            # which must not block the event loop.
            backend = await asyncio.to_thread(self.svc.create_adhoc, "tts", provider, settings)
        except ValidationError as e:
            return web.json_response({"error": str(e), "detail": e.errors()}, status=422)
        except (ValueError, OSError) as e:
            # Backend init rejected the settings (e.g. a missing api_key). OSError is
            # included because provider create() may touch the filesystem: piper loads
            # an ONNX voice from voice_path (PiperVoice.load), so a missing/invalid
            # path raises FileNotFoundError — a config problem (422), not a server
            # fault.
            return web.json_response({"error": str(e)}, status=422)
        try:
            mime, audio = await backend.synthesize(text)
        except (RuntimeError, httpx.HTTPError) as e:
            return web.json_response({"error": str(e)}, status=502)
        if not audio:
            # Unvoiceable text (punctuation-only etc.): the backend served no audio.
            return web.json_response({"error": "nothing to synthesize"}, status=422)
        # Serve the backend's NATIVE format: browsers decode both audio/mpeg and
        # audio/wav, so no transcoding here (audio_codec.to_playable is the speaker
        # delivery boundary, not this browser preview path).
        return web.Response(body=audio, content_type=mime)

    # --- system prompt (named profiles over PromptStore) ----------------------
    def _profile_id(self, request: web.Request) -> int | None:
        """Parse the {id} match-info segment as int; None on a non-numeric id."""
        try:
            return int(request.match_info["id"])
        except (TypeError, ValueError):
            return None

    async def _get_prompt(self, request: web.Request) -> web.Response:
        """Back-compat: the ACTIVE profile as {"id", "name", "text"}."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        active = await asyncio.to_thread(self.prompt_store.active)
        if active is None:
            return web.json_response({"error": "no active profile"}, status=404)
        return web.json_response(
            {"id": active["id"], "name": active["name"], "text": active["text"]}
        )

    async def _put_prompt(self, request: web.Request) -> web.Response:
        """Back-compat: {"text"} updates the ACTIVE profile's text."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            return web.json_response(
                {"error": 'field "text" (string) is required'}, status=400
            )
        active = await asyncio.to_thread(self.prompt_store.active)
        if active is None:
            return web.json_response({"error": "no active profile"}, status=404)
        await asyncio.to_thread(
            self.prompt_store.update, active["id"], text=body["text"]
        )
        return web.json_response({"ok": True})

    async def _get_prompt_profiles(self, request: web.Request) -> web.Response:
        """All profiles (summaries, no text) plus the active profile's id."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        profiles = await asyncio.to_thread(self.prompt_store.list_profiles)
        active_id = next((p["id"] for p in profiles if p["is_active"]), None)
        return web.json_response({"profiles": profiles, "active_id": active_id})

    async def _post_prompt_profile(self, request: web.Request) -> web.Response:
        """Create a profile. Body: {name, text?}; a missing/None text copies the
        CURRENT ACTIVE profile's text (duplicate-friendly default)."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            return web.json_response(
                {"error": 'field "name" (non-empty string) is required'}, status=400
            )
        text = body.get("text")
        if text is not None and not isinstance(text, str):
            return web.json_response(
                {"error": 'field "text" must be a string'}, status=400
            )
        if text is None:
            active = await asyncio.to_thread(self.prompt_store.active)
            text = active["text"] if active is not None else ""
        try:
            profile = await asyncio.to_thread(self.prompt_store.create, name.strip(), text)
        except ValueError as e:  # duplicate name
            return web.json_response({"error": str(e)}, status=409)
        return web.json_response(profile)

    async def _get_prompt_profile(self, request: web.Request) -> web.Response:
        """Full profile (incl. text) by id."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        pid = self._profile_id(request)
        if pid is None:
            return web.json_response({"error": "invalid id"}, status=400)
        profile = await asyncio.to_thread(self.prompt_store.get, pid)
        if profile is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(profile)

    async def _put_prompt_profile(self, request: web.Request) -> web.Response:
        """Partial update. Body: {name?, text?} — at least one is required."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        pid = self._profile_id(request)
        if pid is None:
            return web.json_response({"error": "invalid id"}, status=400)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        name = body.get("name")
        text = body.get("text")
        if name is None and text is None:
            return web.json_response(
                {"error": 'at least one of "name" / "text" is required'}, status=400
            )
        if name is not None and (not isinstance(name, str) or not name.strip()):
            return web.json_response(
                {"error": 'field "name" must be a non-empty string'}, status=400
            )
        if text is not None and not isinstance(text, str):
            return web.json_response(
                {"error": 'field "text" must be a string'}, status=400
            )
        try:
            profile = await asyncio.to_thread(
                self.prompt_store.update, pid,
                name=name.strip() if name is not None else None, text=text,
            )
        except ValueError as e:  # duplicate name
            return web.json_response({"error": str(e)}, status=409)
        if profile is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(profile)

    async def _delete_prompt_profile(self, request: web.Request) -> web.Response:
        """Delete a profile; the active one is refused (409)."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        pid = self._profile_id(request)
        if pid is None:
            return web.json_response({"error": "invalid id"}, status=400)
        try:
            existed = await asyncio.to_thread(self.prompt_store.delete, pid)
        except ValueError as e:  # the active profile
            return web.json_response({"error": str(e)}, status=409)
        if not existed:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"ok": True})

    async def _activate_prompt_profile(self, request: web.Request) -> web.Response:
        """Make one profile active. Optional body {"apply_voice": true} also applies
        that profile's preferred-voice marker (when present) to the TTS config."""
        if self.prompt_store is None:
            return web.json_response({"error": "prompt store not available"}, status=503)
        pid = self._profile_id(request)
        if pid is None:
            return web.json_response({"error": "invalid id"}, status=400)
        # Body is optional: legacy callers POST without one. Tolerate empty/missing.
        apply_voice = False
        if request.body_exists:
            raw = (await request.text()).strip()
            if raw:
                try:
                    body = json.loads(raw)
                except ValueError:
                    return web.json_response({"error": "invalid JSON body"}, status=400)
                if not isinstance(body, dict):
                    return web.json_response({"error": "body must be a JSON object"}, status=400)
                apply_voice = bool(body.get("apply_voice"))
        ok = await asyncio.to_thread(self.prompt_store.activate, pid)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        out = {"ok": True, "active_id": pid}
        if apply_voice:
            out.update(await self._apply_profile_voice(pid))
        return web.json_response(out)

    async def _apply_profile_voice(self, pid: int) -> dict:
        """Parse the profile's preferred-voice marker and apply it to the TTS config.

        Returns a dict merged into the activate response:
          {"voice_applied": {"provider": ..., "fields": {...}}}  on success,
          {"voice_applied": None}                                when there is no marker,
          {"voice_error": "<message>"}                           on a bad provider/config.
        Never raises: activation already succeeded, so a voice problem must not fail it."""
        from src.voice_marker import parse_voice_marker, build_voice_patch
        profile = await asyncio.to_thread(self.prompt_store.get, pid)
        parsed = parse_voice_marker(profile["text"] if profile else "")
        if parsed is None:
            return {"voice_applied": None}
        try:
            prov = get_provider("tts", parsed["provider"])
        except ValueError:
            return {"voice_error": f"unknown TTS provider {parsed['provider']!r}"}
        allowed = set(prov.ConfigModel.model_fields)
        patch = build_voice_patch(parsed["provider"], parsed["fields"], allowed)
        try:
            self.svc.apply(patch)
        except (ValidationError, ValueError) as e:
            return {"voice_error": str(e)}
        return {"voice_applied": {"provider": parsed["provider"], "fields": patch["tts"]["instances"][parsed["provider"]]}}

    def _system_snapshot(self) -> dict:
        # Lightweight system/liveness payload shared by /api/system and the WS
        # heartbeat. Excludes db_size_bytes (relatively expensive to stat and not
        # needed on every 1 s heartbeat tick).
        started = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
        return {
            "version": self.version,
            "started": started,
            "uptime_seconds": int(time.time() - self.started_at),
            "running": True,
            "log_level": self.svc.core.log_level,
            # Live device statuses for the panel: flows through the 1 s WS heartbeat
            # so the Devices page / Sidebar update without polling. Cheap in-memory read.
            "devices": self.device_status() if self.device_status else [],
        }

    async def _get_system(self, request: web.Request) -> web.Response:
        snap = self._system_snapshot()
        # On-disk size of the runs DB (with WAL/SHM sidecars). When the runs store is
        # open it owns the path; otherwise (recording disabled) stat the canonical
        # file directly so the indicator still reports any leftover DB on disk.
        if self.runs_store is not None:
            snap["db_size_bytes"] = self.runs_store.db_size_bytes()
        else:
            snap["db_size_bytes"] = db_file_size(os.path.join(config_store.DATA_DIR, "runs.db"))
        return web.json_response(snap)

    async def _heartbeat_loop(self) -> None:
        # Push a {"type":"system",...} frame to every live WS client on a fixed
        # interval. The panel uses these both as a liveness signal (frames stop
        # when the server dies) and to show a live-updating uptime without polling.
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.run_events.broadcast({"type": "system", **self._system_snapshot()})
            except Exception as e:  # noqa: BLE001 - a transient send error must not kill the heartbeat
                logger.debug(f"panel heartbeat broadcast failed: {e}")

    async def _get_devices(self, request: web.Request) -> web.Response:
        return web.json_response(self.device_status() if self.device_status else [])

    async def _get_device_controls(self, request: web.Request) -> web.Response:
        """Current control values (cutoff %, volume %) for one speaker."""
        if self.device_controls_get is None:
            return web.json_response({"error": "device controls not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response({"error": 'query param "device" is required'}, status=400)
        try:
            return web.json_response(self.device_controls_get(device))
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)

    async def _post_device_controls(self, request: web.Request) -> web.Response:
        """Set one control on a speaker. Body: {device, control, value(number)}."""
        if self.device_controls_set is None:
            return web.json_response({"error": "device controls not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response({"error": 'field "device" (string) is required'}, status=400)
        control = body.get("control")
        if not isinstance(control, str) or not control:
            return web.json_response({"error": 'field "control" (string) is required'}, status=400)
        value = body.get("value")
        # bools are ints in Python — reject them explicitly; accept int or float.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return web.json_response({"error": 'field "value" (number) is required'}, status=400)
        try:
            return web.json_response(self.device_controls_set(device, control, float(value)))
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:  # offline
            return web.json_response({"error": str(e)}, status=409)

    async def _post_capture(self, request: web.Request) -> web.Response:
        """Start a background capture of a manual sample on a speaker (non-blocking).

        Body: {"device": "<name>", "seconds": <int 1..300>}. The device records that
        many seconds of mic audio (no STT/LLM/TTS). The recording runs as a
        SERVER-SIDE background task and is NOT tied to this HTTP request: closing the
        browser no longer cancels it (which used to reboot the device). The browser
        then polls GET /api/capture for the live countdown and downloads the result
        via GET /api/capture/result.

        Status codes:
          - 202 started (returns the initial status snapshot, state "recording")
          - 400 bad input
          - 409 a capture is already in progress on that device (CaptureBusyError)
          - 503 capture not wired (no DeviceManager)
        Capture failures (offline device, unknown device, empty recording, timeout)
        no longer fail this request — they surface on the status endpoint as state
        "error" with the message.
        """
        if self._capture_jobs is None:
            return web.json_response({"error": "capture not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response(
                {"error": 'field "device" (string) is required'}, status=400
            )
        seconds = body.get("seconds")
        # Reject non-integers (bools are ints in Python, so exclude them explicitly).
        if not isinstance(seconds, int) or isinstance(seconds, bool):
            return web.json_response(
                {"error": f'field "seconds" (integer 1..{CAPTURE_MAX_SECONDS}) is required'},
                status=400,
            )
        if not (1 <= seconds <= CAPTURE_MAX_SECONDS):
            return web.json_response(
                {"error": f"seconds must be between 1 and {CAPTURE_MAX_SECONDS}"},
                status=400,
            )
        try:
            snap = self._capture_jobs.start(device, seconds)
        except CaptureBusyError as e:
            # A capture is already in-flight on this device -> conflict.
            return web.json_response({"error": str(e)}, status=409)
        return web.json_response(snap, status=202)

    async def _get_capture_status(self, request: web.Request) -> web.Response:
        """Poll a device's background capture status (drives the live countdown)."""
        if self._capture_jobs is None:
            return web.json_response({"error": "capture not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response(
                {"error": 'query param "device" is required'}, status=400
            )
        return web.json_response(self._capture_jobs.status(device))

    async def _get_capture_result(self, request: web.Request) -> web.Response:
        """Download a completed capture's WAV (one-shot; consumed server-side)."""
        if self._capture_jobs is None:
            return web.json_response({"error": "capture not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response(
                {"error": 'query param "device" is required'}, status=400
            )
        res = self._capture_jobs.take_result(device)
        if res is None:
            return web.json_response({"error": "no capture result"}, status=404)
        wav, seconds = res
        # Sanitize the device name for the download filename (ascii alnum/._- only).
        safe_device = "".join(
            c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in device
        ) or "device"
        # Timestamp the download (UTC) so successive captures of the same device
        # don't collide on the filename and the moment of recording is visible.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"zakhar_{safe_device}_{seconds}s_{stamp}.wav"
        return web.Response(
            body=wav,
            content_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    async def _get_tools(self, request: web.Request) -> web.Response:
        # Live tool sources from the ToolHub (external MCP + built-ins). describe()
        # is in-memory/sync, so no to_thread is needed. None -> empty list.
        sources = self.tool_sources() if self.tool_sources else []
        return web.json_response({"sources": sources})

    # --- observability (run log + metrics) -----------------------------------
    async def _get_runs(self, request: web.Request) -> web.Response:
        if self.runs_store is None:
            return web.json_response({"runs": []})
        device = request.query.get("device") or None
        result = request.query.get("result") or None
        search = request.query.get("search") or None
        try:
            limit = int(request.query.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(limit, 500))
        runs = await asyncio.to_thread(
            self.runs_store.list,
            device=device, result=result, search=search, limit=limit,
        )
        return web.json_response({"runs": runs})

    async def _get_run(self, request: web.Request) -> web.Response:
        if self.runs_store is None:
            return web.json_response({"error": "not found"}, status=404)
        try:
            run_id = int(request.match_info["id"])
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid id"}, status=400)
        run = await asyncio.to_thread(self.runs_store.get, run_id)
        if run is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(run)

    async def _get_run_audio(self, request: web.Request) -> web.Response:
        """Serve the stored utterance WAV for one run (inline, playable/downloadable).

        Without a `channel` query param the original stored file is served as is
        (mono or stereo — this is what Download uses). `channel=stt` extracts
        channel 0 (left, what STT received) and `channel=raw` channel 1 (right,
        the other mic channel) as standalone mono WAVs, split on the fly.
        """
        if self.runs_store is None:
            return web.json_response({"error": "not found"}, status=404)
        try:
            run_id = int(request.match_info["id"])
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid id"}, status=400)
        channel = request.query.get("channel")
        if channel:
            idx = {"stt": 0, "raw": 1}.get(channel)
            if idx is None:
                return web.json_response(
                    {"error": 'channel must be "stt" or "raw"'}, status=400)
        wav = await asyncio.to_thread(self.runs_store.get_audio, run_id)
        if wav is None:
            return web.json_response({"error": "not found"}, status=404)
        filename = f"zakhar_run_{run_id}.wav"
        if channel:
            wav = await asyncio.to_thread(_extract_wav_channel, wav, idx)
            if wav is None:
                return web.json_response({"error": "channel not available"}, status=404)
            filename = f"zakhar_run_{run_id}_{channel}.wav"
        return web.Response(
            body=wav,
            content_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    async def _runs_stream(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint streaming each finalized run as {"type":"run","run":{...}}.

        heartbeat keeps the link alive and detects dead peers. We don't expect any
        client messages — we iterate only to drive ping/pong and detect close.
        """
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        if self.run_events is None:
            await ws.close()
            return ws
        self.run_events.register(ws)
        try:
            async for _msg in ws:
                pass  # inbound messages are ignored; this loop just keeps the socket live
        finally:
            self.run_events.unregister(ws)
        return ws

    async def _get_metrics(self, request: web.Request) -> web.Response:
        if self.runs_store is None:
            return web.json_response({
                "requests_24h": 0,
                "p50_ms": None,
                "p95_ms": None,
                "error_rate": 0.0,
                "per_stage_avg_ms": {"vad": None, "stt": None, "llm": None, "tts": None},
            })
        metrics = await asyncio.to_thread(self.runs_store.metrics, now=time.time())
        return web.json_response(metrics)

    async def _spa_index(self, request: web.Request) -> web.Response:
        """Serve index.html for any non-/api GET so client-side routing works."""
        return web.FileResponse(os.path.join(self.static_dir, "index.html"))

    # --- agent MCP endpoint (aiohttp → ASGI bridge) ----------------------------
    async def _mcp(self, request: web.Request) -> web.StreamResponse:
        """Bridge one /mcp request into the FastMCP streamable-HTTP session manager.

        The endpoint object exposes a plain path-agnostic ASGI callable
        (AgentMcpEndpoint.handle); this handler translates the aiohttp request
        into an ASGI http scope, feeds the (fully read) body as a single
        http.request event, and streams the ASGI response back. Streaming-safe:
        chunks are written as they arrive, so both the buffered JSON responses of
        stateless+json mode and SSE chunks would pass through.

        The `core.agent_mcp.enabled` toggle is read LIVE per request through the
        ConfigService — flipping it in the panel applies immediately, no rebuild.
        """
        if self.agent_mcp is None:
            return web.json_response(
                {"error": "agent MCP endpoint not available"}, status=503
            )
        if not self.svc.core.agent_mcp.enabled:
            return web.json_response(
                {"error": "MCP server is disabled (core.agent_mcp.enabled = false)"},
                status=403,
            )
        body = await request.read()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": f"{request.version.major}.{request.version.minor}",
            "method": request.method,
            "scheme": request.scheme,
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": request.query_string.encode(),
            "root_path": "",
            # Host is included automatically (aiohttp keeps it in request.headers).
            "headers": [
                (k.lower().encode("latin-1"), v.encode("latin-1"))
                for k, v in request.headers.items()
            ],
            "client": (request.remote, 0),
            "server": (self.host, self.port),
        }

        body_sent = False

        async def receive():
            # One-shot body, then disconnect on any further pull.
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        resp: web.StreamResponse | None = None

        async def send(message):
            nonlocal resp
            if message["type"] == "http.response.start":
                headers = CIMultiDict()
                for name, value in message.get("headers", []):
                    n = name.decode("latin-1")
                    # Hop-by-hop headers are owned by the aiohttp transport.
                    if n.lower() in _HOP_BY_HOP_HEADERS:
                        continue
                    headers.add(n, value.decode("latin-1"))
                resp = web.StreamResponse(status=message["status"], headers=headers)
                await resp.prepare(request)
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    await resp.write(chunk)
                if not message.get("more_body"):
                    await resp.write_eof()

        await self.agent_mcp.handle(scope, receive, send)
        if resp is None:  # the ASGI app produced no response at all
            return web.json_response({"error": "MCP endpoint sent no response"},
                                     status=500)
        return resp

    # --- wiring --------------------------------------------------------------
    def build_app(self) -> web.Application:
        allowed = _ALLOWED_ORIGINS
        app = web.Application(middlewares=[_make_cors_middleware(allowed)])
        app.add_routes([
            web.get("/api/catalog", self._get_catalog),
            web.get("/api/config", self._get_config),
            web.patch("/api/config", self._patch_config),
            web.get("/api/options", self._get_options),
            web.get("/api/chimes", self._get_chimes),
            web.post("/api/chimes/play", self._post_play_chime),
            web.post("/api/tts/test", self._post_tts_test),
            web.get("/api/prompt", self._get_prompt),
            web.put("/api/prompt", self._put_prompt),
            web.get("/api/prompt/profiles", self._get_prompt_profiles),
            web.post("/api/prompt/profiles", self._post_prompt_profile),
            web.get("/api/prompt/profiles/{id}", self._get_prompt_profile),
            web.put("/api/prompt/profiles/{id}", self._put_prompt_profile),
            web.delete("/api/prompt/profiles/{id}", self._delete_prompt_profile),
            web.post("/api/prompt/profiles/{id}/activate", self._activate_prompt_profile),
            web.get("/api/system", self._get_system),
            web.get("/api/devices", self._get_devices),
            web.get("/api/device/controls", self._get_device_controls),
            web.post("/api/device/controls", self._post_device_controls),
            web.post("/api/capture", self._post_capture),
            web.get("/api/capture", self._get_capture_status),
            web.get("/api/capture/result", self._get_capture_result),
            web.get("/api/tools", self._get_tools),
            web.get("/api/runs", self._get_runs),
            web.get("/api/runs/stream", self._runs_stream),  # before {id}: literal wins
            web.get("/api/runs/{id}/audio", self._get_run_audio),
            web.get("/api/runs/{id}", self._get_run),
            web.get("/api/metrics", self._get_metrics),
            # Agent-facing MCP endpoint (streamable HTTP). Registered BEFORE the SPA
            # fallback below — aiohttp resolves routes in registration order.
            web.post("/mcp", self._mcp),
            web.get("/mcp", self._mcp),
            web.delete("/mcp", self._mcp),
        ])
        # Static frontend is optional: only mount it when a built dist exists.
        if self.static_dir and os.path.isdir(self.static_dir):
            # SPA catch-all: any non-/api GET path returns index.html so the
            # client-side router can take over. Static assets (js/css) are still
            # resolved first by add_static below (when an assets/ dir exists).
            assets = os.path.join(self.static_dir, "assets")
            if os.path.isdir(assets):
                app.router.add_static("/assets/", assets)
            app.router.add_get("/", self._spa_index)
            # /mcp is excluded so an unregistered method on the MCP endpoint can
            # never fall through into the SPA index.
            app.router.add_get("/{path:(?!api/|mcp$).*}", self._spa_index)

        if self.run_events is not None:
            async def _start_heartbeat(_app: web.Application) -> None:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            app.on_startup.append(_start_heartbeat)

            async def _stop_heartbeat(_app: web.Application) -> None:
                if self._heartbeat_task is not None:
                    self._heartbeat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._heartbeat_task
                    self._heartbeat_task = None
            app.on_shutdown.append(_stop_heartbeat)

        async def _close_ws_clients(_app: web.Application) -> None:
            # Fired by AppRunner.cleanup() BEFORE it waits (up to shutdown_timeout) for
            # in-flight handlers. Closing the live run-stream sockets here unblocks the
            # _runs_stream handler so shutdown (and Ctrl+C) completes immediately instead
            # of hanging until the timeout elapses.
            if self.run_events is not None:
                await self.run_events.close_all()
        app.on_shutdown.append(_close_ws_clients)

        async def _close_capture_jobs(_app: web.Application) -> None:
            # Cancel any in-flight background capture tasks on shutdown so cleanup()
            # doesn't wait on them (and so they don't outlive the server).
            if self._capture_jobs is not None:
                await self._capture_jobs.close()
        app.on_shutdown.append(_close_capture_jobs)
        return app

    async def start(self) -> None:
        # The agent MCP endpoint's session manager runs for the panel's lifetime
        # (its start/stop are owned here; AgentMcpEndpoint.start is a no-op when
        # already started).
        if self.agent_mcp is not None:
            await self.agent_mcp.start()
        # access_log=None disables per-request access logs; shutdown_timeout bounds how
        # long cleanup() waits for in-flight handlers (defense-in-depth so even a lingering
        # connection can't make Ctrl+C hang for the 60s aiohttp default).
        self._runner = web.AppRunner(self.build_app(), access_log=None, shutdown_timeout=5.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"panel API on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        # Stopped AFTER the listener so no request can race the teardown; stop()
        # is idempotent and safe when start() failed before reaching it.
        if self.agent_mcp is not None:
            await self.agent_mcp.stop()
