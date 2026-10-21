"""Admin-panel HTTP API (aiohttp).

Runs in a TRUSTED zone — there is NO authentication. The panel is driven entirely
by ConfigService, so it carries no provider-specific knowledge: it exposes the
catalog, the raw config document, generic patch/options endpoints, the named
system-prompt profiles, system/version info and live device status. It also
serves the agent-facing MCP endpoint at /mcp (an aiohttp→ASGI bridge into the
FastMCP streamable-HTTP session manager, see src/agent_mcp.AgentMcpEndpoint);
the `core.agent_mcp.enabled` toggle is read live per request.

Start/stop mirror the device server (AppRunner + TCPSite). CORS is restricted to the
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
import re
import time
import wave
from datetime import datetime, timezone

import httpx
import numpy as np
from aiohttp import web
from loguru import logger
from multidict import CIMultiDict
from pydantic import ValidationError

from src import captures, config_store, dataset_jobs, device_logs
from src.audio_prep import pcm_to_wav_bytes
from src.device_server import (
    CAPTURE_MAX_SECONDS, RELEASE_WAKE_MODEL, UPLOADED_WAKE_PREFIX, CaptureBusyError, DeviceOfflineError,
    list_wake_models, uploaded_wake_models_dir,
)
from src.plugins.base import get_provider
from src.runs_store import db_file_size
from src.tts import pcm_rate

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
_CHIME_EXTS = (".wav",)

# Characters an uploaded wake-word model's file stem may keep; any other becomes "_", so the
# saved name is a plain file name (no path separators).
_UNSAFE_MODEL_STEM_RE = re.compile(r"[^A-Za-z0-9._-]")

# Stored TTS audio mime -> download file extension.
_EXT_FOR_MIME = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac"}


def ext_for_mime(mime: str) -> str:
    """File extension for a stored TTS audio mime; defaults to mp3 for unknown types."""
    return _EXT_FOR_MIME.get(mime, "mp3")


def list_chimes(root: str = _REPO_ROOT) -> list[str]:
    """Return sorted repo-root-relative paths of the bundled chime audio files.

    Scans <root>/assets/chimes for WAV files. Returns an empty list
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


async def _aclose_quietly(aiter) -> None:
    """Best-effort aclose() of an async iterator that will not be (further)
    iterated. Async generators (the default buffered TtsBackend adapter and the
    streaming providers) have aclose(); a plain iterator may not. A close failure
    is swallowed so it cannot mask the real outcome on the error/empty paths."""
    aclose = getattr(aiter, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:  # noqa: BLE001 - closing is best-effort
        pass


class PanelServer:
    def __init__(self, svc, host, port, *, version, started_at,
                 devices=None,
                 static_dir=None, runs_store=None, tool_sources=None,
                 run_events=None, prompt_store=None, agent_mcp=None,
                 assistant_task_reply=None,
                 heartbeat_interval: float = 1.0, reload_status=None):
        # svc: ConfigService; started_at: float (time.time())
        # devices: optional DeviceManager behind the device endpoints (status, controls, wake
        #   probability, detect, model reload, say, chime preview, captures, dataset sessions).
        #   None -> the device lists are empty and every endpoint that needs a device returns
        #   503; the captures on disk are still listed and downloaded.
        # static_dir: optional path to built frontend
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
        self.devices = devices
        # Wake-word dataset sessions (a beep, then a capture, per take) run server-side too;
        # without a device manager every /api/datasets* endpoint returns 503.
        self._dataset_jobs = (
            dataset_jobs.DatasetJobManager(devices) if devices is not None else None
        )
        self.static_dir = static_dir
        self.runs_store = runs_store
        self.tool_sources = tool_sources
        self.run_events = run_events
        self.prompt_store = prompt_store
        self.agent_mcp = agent_mcp
        # assistant_task_reply: optional async callable (payload) -> (status, body) that
        #   takes one reply of the ZeroClaw assistant (AssistantTasks.on_reply). None ->
        #   POST /api/assistant-tasks/reply returns 503.
        self.assistant_task_reply = assistant_task_reply
        # reload_status: optional zero-arg callable -> list[str] of what is currently
        #   rebuilding (a backend category whose model load is in flight, or "tools" while
        #   the tool sources connect); None -> the snapshot reports an empty list.
        self.reload_status = reload_status
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

    async def _get_wake_models(self, request: web.Request) -> web.Response:
        """List the names of the wake-word models (the release's and the uploaded ones) for a
        device's wake_model selector, plus the one an empty wake_model falls back to."""
        return web.json_response({"options": list_wake_models(), "release": RELEASE_WAKE_MODEL})

    async def _post_wake_model(self, request: web.Request) -> web.Response:
        """Save an uploaded wake-word model and return {"name": "uploaded/<stem>"}, the name
        GET /api/wake-models now lists.

        Body: multipart/form-data with the field "file", a <stem>.tflite. The stem keeps ASCII
        letters, digits, ".", "_" and "-" (any other character becomes "_"); the file goes to
        <DATA_DIR>/wake_models/<stem>.tflite, replacing one of that name. aiohttp's default
        1 MiB body limit applies (a model is ~80 KB).

        Status codes: 200 saved, 400 no file / not a .tflite / empty name.
        """
        form = await request.post()
        upload = form.get("file")
        if not isinstance(upload, web.FileField):
            return web.json_response({"error": 'multipart field "file" is required'}, status=400)
        filename = upload.filename or ""
        if not filename.lower().endswith(".tflite"):
            return web.json_response({"error": "the file must be a .tflite model"}, status=400)
        stem = _UNSAFE_MODEL_STEM_RE.sub("_", filename[:-len(".tflite")])
        if not stem:
            return web.json_response({"error": "the file name before .tflite is empty"}, status=400)
        folder = uploaded_wake_models_dir()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{stem}.tflite").write_bytes(upload.file.read())
        name = f"{UPLOADED_WAKE_PREFIX}{stem}"
        # A device already set to this name would keep the replaced file: its wake_model does
        # not change, so nothing re-sends it. Send it now to the online ones; an offline one
        # gets the new file at its next connection.
        if self.devices is not None:
            for dev in self.svc.core.devices:
                if dev.wake_model == name:
                    try:
                        self.devices.find(dev.name).reload_model()
                    except (LookupError, RuntimeError):
                        pass
        return web.json_response({"name": name})

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
        if self.devices is None:
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
            result = await self.devices.play_chime(sound_path, device)
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.json_response(result)

    async def _post_assistant_task_reply(self, request: web.Request) -> web.Response:
        """The send_url of ZeroClaw's webhook channel: every reply of the ZeroClaw assistant
        lands here, {"content", "thread_id"?, "recipient"?}, and is stored on its task and
        spoken on the task's speaker. No auth, like the rest of the trusted-LAN panel.

        Status codes: 200 taken (also when it could not be spoken), 400 malformed JSON,
        404 no task for the thread id, 503 assistant tasks not wired.
        """
        if self.assistant_task_reply is None:
            return web.json_response({"error": "assistant tasks not available"}, status=503)
        payload = await _read_json(request)
        status, body = await self.assistant_task_reply(payload)
        return web.json_response(body, status=status)

    async def _post_tts_test(self, request: web.Request) -> web.StreamResponse:
        """Synthesize a short test phrase with caller-supplied TTS settings and return
        the audio for in-browser playback.

        Body: {"provider": "<tts plugin id>", "settings": {...}, "text": "..."}. The
        settings are the panel's CURRENT form draft (possibly unsaved): the backend is
        built ad hoc (ConfigService.create_adhoc) — the stored config and the running
        pipeline are never touched. Provider-agnostic: the draft dict is validated by
        the provider's own ConfigModel, so any TTS plugin works with zero per-provider
        code here.

        Status codes: 200 audio (WAV), 400 bad input,
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
        # Stream the synthesized audio so the browser can begin playback before the
        # whole clip is ready. synthesize_stream() validates connect/auth/4xx
        # EAGERLY (raises before returning the iterator, per the TtsBackend
        # contract), so those still map to a 502. We then PEEK the first chunk:
        # this (a) surfaces an error embedded in the first chunk as a 502 and
        # (b) lets an empty stream (unvoiceable text) return 422 — all while the
        # response headers are still unsent and a JSON error is still possible.
        # Only a failure AFTER the first chunk (headers already flushed) degrades
        # to a dropped/truncated stream, which the browser surfaces as an error.
        try:
            mime, chunks = await backend.synthesize_stream(text)
            aiter = chunks.__aiter__()
            try:
                first = await aiter.__anext__()
            except StopAsyncIteration:
                first = b""
        except (RuntimeError, httpx.HTTPError) as e:
            return web.json_response({"error": str(e)}, status=502)
        if not first:
            # Unvoiceable text (punctuation-only etc.): the backend served no audio.
            # Close the iterator so a streaming backend releases its HTTP response.
            await _aclose_quietly(aiter)
            return web.json_response({"error": "nothing to synthesize"}, status=422)
        rate = pcm_rate(mime)
        if rate is not None:
            # Browsers cannot play headerless PCM: wrap the whole clip as WAV.
            try:
                pcm = first + b"".join([c async for c in aiter])
            except (RuntimeError, httpx.HTTPError) as e:
                await _aclose_quietly(aiter)
                return web.json_response({"error": str(e)}, status=502)
            return web.Response(body=pcm_to_wav_bytes(pcm, rate=rate), content_type="audio/wav")
        # A WAV backend is served as is. No Content-Length -> aiohttp uses chunked
        # transfer encoding.
        resp = web.StreamResponse(status=200)
        resp.content_type = mime
        await resp.prepare(request)
        try:
            await resp.write(first)
            async for chunk in aiter:
                if chunk:
                    await resp.write(chunk)
        except (RuntimeError, httpx.HTTPError, ConnectionError, OSError):
            # Mid-stream synthesis failure or client disconnect AFTER headers were
            # sent (ConnectionResetError/BrokenPipeError/ConnectionAbortedError are
            # all OSError subtypes): the status can no longer change. Stop writing
            # (aiohttp closes the connection; the browser reports a truncated
            # stream) and close the iterator so the backend releases its upstream
            # HTTP response. If the error came FROM the iterator it already
            # self-terminated; aclose() is then a no-op.
            await _aclose_quietly(aiter)
            return resp
        await resp.write_eof()
        return resp

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
        variants = self.svc.document()["tts"]["variants"]
        patch = build_voice_patch(parsed["provider"], parsed["fields"], allowed, variants)
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
            "devices": self.devices.statuses() if self.devices is not None else [],
            # Backend categories whose hot-reload (model load) is in flight, so the panel
            # can show a 'loading' indicator. Cheap in-memory read; flows through both
            # GET /api/system and the 1 s WS heartbeat.
            "reloading": self.reload_status() if self.reload_status else [],
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
        return web.json_response(self.devices.statuses() if self.devices is not None else [])

    async def _get_device_controls(self, request: web.Request) -> web.Response:
        """Current control values (cutoff %, volume %) for one speaker."""
        if self.devices is None:
            return web.json_response({"error": "device controls not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response({"error": 'query param "device" is required'}, status=400)
        try:
            return web.json_response(self.devices.find(device).controls_snapshot())
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)

    async def _post_device_controls(self, request: web.Request) -> web.Response:
        """Set one control on a speaker. Body: {device, control, value(number)}."""
        if self.devices is None:
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
            target = self.devices.find(device)
            target.set_control(control, float(value))
            return web.json_response(target.controls_snapshot())
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:  # offline
            return web.json_response({"error": str(e)}, status=409)

    async def _get_wake_probability(self, request: web.Request) -> web.Response:
        """Current live Wake Probability snapshot for one speaker (drives the panel meter)."""
        if self.devices is None:
            return web.json_response({"error": "wake probability not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response({"error": 'query param "device" is required'}, status=400)
        try:
            return web.json_response(self.devices.find(device).wake_prob())
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)

    async def _post_wake_probability(self, request: web.Request) -> web.Response:
        """Enable/disable the on-device Wake Probability stream. Body: {device, enabled(bool)}.

        The panel POSTs enabled=true when the device modal opens and enabled=false when it
        closes, so the firmware only streams the probability while someone is watching."""
        if self.devices is None:
            return web.json_response({"error": "wake probability not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response({"error": 'field "device" (string) is required'}, status=400)
        enabled = body.get("enabled")
        # Strict bool: reject ints/strings/None (avoid an ambiguous truthiness toggle).
        if not isinstance(enabled, bool):
            return web.json_response({"error": 'field "enabled" (bool) is required'}, status=400)
        try:
            return web.json_response(self.devices.find(device).set_wake_prob_stream(bool(enabled)))
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:  # offline
            return web.json_response({"error": str(e)}, status=409)

    async def _post_detect(self, request: web.Request) -> web.Response:
        """Debug switch: turn a device's wake-word detection off/on. Body: {device, enabled(bool)}.

        Returns {device, detect}. The server keeps no state: the device turns detection back
        on at every new connection. Status codes: 200 ok, 400 bad input, 404 unknown device,
        409 offline, 503 not wired."""
        if self.devices is None:
            return web.json_response({"error": "detect switch not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response({"error": 'field "device" (string) is required'}, status=400)
        enabled = body.get("enabled")
        # Strict bool: reject ints/strings/None (avoid an ambiguous truthiness toggle).
        if not isinstance(enabled, bool):
            return web.json_response({"error": 'field "enabled" (bool) is required'}, status=400)
        try:
            return web.json_response(self.devices.find(device).set_detect(enabled))
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:  # offline
            return web.json_response({"error": str(e)}, status=409)

    async def _post_model(self, request: web.Request) -> web.Response:
        """Reload a device's models: the server re-reads the wake-word and VAD model files and
        sends them, the device swaps them without a restart. Body: {device}. Returns {device,
        models}. Status codes: 200 ok, 400 bad input, 404 unknown device, 409 offline, 503 not
        wired."""
        if self.devices is None:
            return web.json_response({"error": "model reload not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response({"error": 'field "device" (string) is required'}, status=400)
        try:
            return web.json_response(self.devices.find(device).reload_model())
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except RuntimeError as e:  # offline
            return web.json_response({"error": str(e)}, status=409)

    async def _post_say(self, request: web.Request) -> web.Response:
        """Speak text on a speaker through the TTS (the device modal's volume check).
        Body: {device, text}. Answers once the speech has played. Status codes: 200 ok,
        400 bad input, 404 unknown device, 409 offline, 502 the speech failed, 503 not
        wired."""
        if self.devices is None:
            return web.json_response({"error": "say not available"}, status=503)
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response({"error": 'field "device" (string) is required'}, status=400)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"error": 'field "text" (non-empty string) is required'}, status=400)
        try:
            await self.devices.find(device).say(text)
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except DeviceOfflineError as e:
            return web.json_response({"error": str(e)}, status=409)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"ok": True})

    async def _post_capture(self, request: web.Request) -> web.Response:
        """Start a capture of a manual sample on a device (non-blocking).

        Body: {"device": "<name>", "seconds": <int 1..300>}. The device records that many
        seconds of mic audio (no STT/LLM/TTS) on its own, and the server writes it to
        <DATA_DIR>/captures/<device>/<YYYYmmdd-HHMMSS>-<id>.wav as it arrives, so closing the
        browser, a lost link or a server restart does not lose it (docs/device-protocol.md,
        "Capture"). The browser polls GET /api/capture for the countdown and downloads the
        result from GET /api/capture/result; GET /api/captures lists every capture kept.

        Status codes:
          - 202 started (returns the status snapshot, state "starting"; "recording" comes
            with the first audio)
          - 400 bad input
          - 404 unknown device
          - 409 the device is offline, or a capture is already running on it
          - 503 capture not wired (no DeviceManager)
        A capture that fails later (the device is busy with a voice session, no audio came)
        surfaces on the status endpoint as state "error" with the message.
        """
        if self.devices is None:
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
            snap = self.devices.find(device).start_capture(seconds).snapshot()
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        except (DeviceOfflineError, CaptureBusyError) as e:
            return web.json_response({"error": str(e)}, status=409)
        return web.json_response(snap, status=202)

    async def _get_capture_status(self, request: web.Request) -> web.Response:
        """A device's capture status (drives the live countdown): its running capture, else
        its newest one on disk, else state "idle". The snapshot: {state (starting | recording
        | done | error | idle), device, name, seconds, remaining, duration (seconds of audio
        written), gaps ([{at, seconds}]: audio lost to a dropped link or a device restart,
        zeros in the file), error}."""
        if self.devices is None:
            return web.json_response({"error": "capture not available"}, status=503)
        device = request.query.get("device")
        if not device:
            return web.json_response(
                {"error": 'query param "device" is required'}, status=400
            )
        return web.json_response(self.devices.capture_status(device))

    @staticmethod
    async def _capture_download(cap: captures.Capture) -> web.Response:
        """The capture's WAV as an attachment named after its device, length and name."""
        body = await asyncio.to_thread(captures.wav_bytes, cap)
        # Sanitize the device name for the download filename (ascii alnum/._- only).
        safe_device = "".join(
            c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in cap.device
        ) or "device"
        filename = f"zakhar_{safe_device}_{cap.seconds}s_{cap.name}.wav"
        return web.Response(
            body=body,
            content_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    async def _get_capture_result(self, request: web.Request) -> web.Response:
        """Download the device's newest finished ("done") capture; 404 when it has none.
        Nothing is consumed: the capture stays until DELETE /api/captures/{device}/{name}."""
        device = request.query.get("device")
        if not device:
            return web.json_response(
                {"error": 'query param "device" is required'}, status=400
            )
        cap = await asyncio.to_thread(captures.newest, device, "done")
        if cap is None:
            return web.json_response({"error": "no capture result"}, status=404)
        return await self._capture_download(cap)

    async def _get_captures(self, request: web.Request) -> web.Response:
        """Every capture on disk, newest first: {"captures": [snapshot]} (the snapshot of
        GET /api/capture)."""
        caps = await asyncio.to_thread(captures.everything)
        return web.json_response({"captures": [c.snapshot() for c in caps]})

    async def _get_capture_wav(self, request: web.Request) -> web.Response:
        """Download one capture by its device and name; a running one downloads as far as it
        got. 400 for an invalid device or name, 404 when there is no such capture."""
        device, name = request.match_info["device"], request.match_info["name"]
        if not captures.valid(device, name):
            return web.json_response({"error": "invalid device or capture name"}, status=400)
        cap = await asyncio.to_thread(captures.find, device, name)
        if cap is None:
            return web.json_response({"error": "not found"}, status=404)
        return await self._capture_download(cap)

    async def _delete_capture(self, request: web.Request) -> web.Response:
        """Delete one capture (its WAV and sidecar). 400 for an invalid device or name, 404
        when there is no such capture, 409 while it runs, 503 not wired."""
        if self.devices is None:
            return web.json_response({"error": "capture not available"}, status=503)
        device, name = request.match_info["device"], request.match_info["name"]
        if not captures.valid(device, name):
            return web.json_response({"error": "invalid device or capture name"}, status=400)
        try:
            deleted = self.devices.capture_delete(device, name)
        except CaptureBusyError as e:
            return web.json_response({"error": str(e)}, status=409)
        if not deleted:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"ok": True})

    # --- device logs (src/device_logs.py) ---------------------------------------
    async def _get_device_logs(self, request: web.Request) -> web.Response:
        """The device log days on disk: {"devices": {device: [YYYY-MM-DD, newest first]}}."""
        return web.json_response({"devices": await asyncio.to_thread(device_logs.days)})

    async def _get_device_log(self, request: web.Request) -> web.Response:
        """One device log day from byte ?offset= (default 0) up to its last whole line:
        {"text", "offset": the byte position to read on from}. 400 for a non-integer offset,
        404 when the device has no such day."""
        device, day = request.match_info["device"], request.match_info["day"]
        try:
            offset = int(request.query.get("offset", "0"))
        except ValueError:
            return web.json_response({"error": "offset must be an integer"}, status=400)
        got = await asyncio.to_thread(device_logs.read, device, day, offset)
        if got is None:
            return web.json_response({"error": "not found"}, status=404)
        text, offset = got
        return web.json_response({"text": text, "offset": offset})

    # --- wake-word dataset recording -------------------------------------------
    def _dataset_refusal(self, name: str) -> web.Response | None:
        """503 when dataset recording is not wired, 400 for an invalid set name, else None."""
        if self._dataset_jobs is None:
            return web.json_response({"error": "dataset recording not available"}, status=503)
        if not dataset_jobs.valid_name(name):
            return web.json_response(
                {"error": 'set name must be 1..64 letters, digits, "_" or "-"'}, status=400
            )
        return None

    async def _post_dataset(self, request: web.Request) -> web.Response:
        """Start a wake-word dataset recording session on a device (non-blocking).

        Path: the set name (1..64 letters, digits, "_" or "-"; Cyrillic is fine). Body:
        {"device": "<name>", "takes": <int 1..1000>, "seconds": <int 1..10, default 3>}.
        Each take: the device beeps (the synthesized chime; a device without a speaker
        records without it), then records a capture of `seconds` (see POST /api/capture),
        which is moved into the set as its next <DATA_DIR>/datasets/<name>/<NNNN>.wav —
        numbering continues after the highest existing clip, so a later session extends the
        set. Takes run back to back as a SERVER-SIDE background task (a 200-take session
        runs ~15 minutes), so the request returns at once; progress shows in
        GET /api/datasets.

        Status codes:
          - 202 started (returns the job snapshot, state "recording")
          - 400 bad input or set name
          - 409 a dataset session is already running on that device
          - 503 dataset recording not wired (no DeviceManager)
        A take whose capture failed or has a gap (a lost link) is deleted and recorded
        again; it never fails this request. A gap does not count, and while the device is
        offline the session is in state "waiting" and tries again every second. Any other
        failure (unknown device, busy with a voice session, no audio) counts and pauses the
        session for 5 s, and the third failure in a row of one take ends the session as
        state "error" with the message; the clips already written stay. Wake-word detection
        stays on between takes, so a word said after the take's window can start a voice
        session and the next take then fails as busy; POST /api/device/detect
        {"enabled": false} before the session (and true after) keeps that from happening.
        """
        name = request.match_info["name"]
        refusal = self._dataset_refusal(name)
        if refusal is not None:
            return refusal
        body = await _read_json(request)
        if not isinstance(body, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        device = body.get("device")
        if not isinstance(device, str) or not device:
            return web.json_response(
                {"error": 'field "device" (string) is required'}, status=400
            )
        takes = body.get("takes")
        # Reject non-integers (bools are ints in Python, so exclude them explicitly).
        if (not isinstance(takes, int) or isinstance(takes, bool)
                or not 1 <= takes <= dataset_jobs.MAX_TAKES):
            return web.json_response(
                {"error": f'field "takes" (integer 1..{dataset_jobs.MAX_TAKES}) is required'},
                status=400,
            )
        seconds = body.get("seconds", dataset_jobs.DEFAULT_SECONDS)
        if (not isinstance(seconds, int) or isinstance(seconds, bool)
                or not 1 <= seconds <= dataset_jobs.MAX_SECONDS):
            return web.json_response(
                {"error": f'field "seconds" must be an integer 1..{dataset_jobs.MAX_SECONDS}'},
                status=400,
            )
        try:
            snap = self._dataset_jobs.start(name, device, takes, seconds)
        except CaptureBusyError as e:
            return web.json_response({"error": str(e)}, status=409)
        return web.json_response(snap, status=202)

    async def _get_datasets(self, request: web.Request) -> web.Response:
        """List the dataset sets: {"sets": [{"name", "clips", "job"}]}.

        Every set directory on disk with its WAV count, plus the set's latest session
        snapshot (running or finished; kept until the next session of that set starts), or
        null when no session of it ran since the server started.
        """
        if self._dataset_jobs is None:
            return web.json_response({"error": "dataset recording not available"}, status=503)
        return web.json_response({"sets": self._dataset_jobs.sets()})

    async def _post_dataset_stop(self, request: web.Request) -> web.Response:
        """Stop the set's running session after the current take (it ends as "stopped").

        Returns the session snapshot (still "recording" until that take is written);
        404 when no session of the set is running.
        """
        name = request.match_info["name"]
        refusal = self._dataset_refusal(name)
        if refusal is not None:
            return refusal
        try:
            snap = self._dataset_jobs.stop(name)
        except LookupError as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.json_response(snap)

    async def _get_dataset_zip(self, request: web.Request) -> web.Response:
        """Download the set's WAVs as one zip archive; 404 when the set does not exist."""
        name = request.match_info["name"]
        refusal = self._dataset_refusal(name)
        if refusal is not None:
            return refusal
        # A large set is tens of megabytes: read and pack it off the event loop.
        data = await asyncio.to_thread(dataset_jobs.zip_set, name)
        if data is None:
            return web.json_response({"error": f"no dataset {name!r}"}, status=404)
        # Sanitize the set name for the download filename (ascii alnum/._- only).
        safe_name = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in name)
        return web.Response(
            body=data,
            content_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"'},
        )

    async def _get_tools(self, request: web.Request) -> web.Response:
        # Live tool sources from the ToolHub (external MCP + built-ins). describe()
        # is in-memory/sync, so no to_thread is needed. None -> empty list.
        sources = self.tool_sources() if self.tool_sources else []
        return web.json_response({"sources": sources})

    # --- observability (run log + metrics) -----------------------------------
    async def _get_runs(self, request: web.Request) -> web.Response:
        if self.runs_store is None:
            return web.json_response({"runs": [], "total": 0})
        device = request.query.get("device") or None
        result = request.query.get("result") or None
        search = request.query.get("search") or None
        chain = request.query.get("chain") or None
        try:
            limit = int(request.query.get("limit", 100))
        except (TypeError, ValueError):
            limit = 100
        limit = max(1, min(limit, 500))
        # Offset for numbered pagination. Missing/malformed -> first page (0), not
        # an error; clamped to >= 0 so a negative value can't walk past the start.
        try:
            offset = int(request.query.get("offset", 0))
        except (TypeError, ValueError):
            offset = 0
        offset = max(0, offset)
        # Fetch the page and the total sequentially: both reads share the store's
        # write lock, so gathering them would only contend on that lock.
        runs = await asyncio.to_thread(
            self.runs_store.list,
            device=device, result=result, search=search, chain=chain,
            limit=limit, offset=offset,
        )
        total = await asyncio.to_thread(
            self.runs_store.count, device=device, result=result, search=search, chain=chain,
        )
        return web.json_response({"runs": runs, "total": total})

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

    async def _get_run_tts_audio(self, request: web.Request) -> web.Response:
        """Serve the stored generated TTS reply audio for one run (inline, playable/
        downloadable). The blob is the TTS backend's native clip (raw PCM is stored as
        WAV); the stored mime is used as the Content-Type and to pick the download
        extension."""
        if self.runs_store is None:
            return web.json_response({"error": "not found"}, status=404)
        try:
            run_id = int(request.match_info["id"])
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid id"}, status=400)
        result = await asyncio.to_thread(self.runs_store.get_tts_audio, run_id)
        if result is None:
            return web.json_response({"error": "not found"}, status=404)
        audio, mime = result
        filename = f"zakhar_run_{run_id}_tts.{ext_for_mime(mime)}"
        return web.Response(
            body=audio,
            content_type=mime,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    async def _runs_stream(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint streaming each finalized run as {"type":"run","run":{...}}.

        heartbeat keeps the link alive and detects dead peers. We don't expect any
        client messages — we iterate only to drive ping/pong and detect close.
        """
        ws = web.WebSocketResponse(heartbeat=30.0)
        try:
            await ws.prepare(request)
        except (ConnectionError, OSError) as e:
            # The client disconnected during the WebSocket handshake, before aiohttp
            # could write the 101 response (ClientConnectionResetError is a
            # ConnectionResetError -> ConnectionError -> OSError). At this point
            # ws._payload_writer is already set but ws._writer is not, so returning
            # the half-prepared response as-is would make aiohttp's finish_response()
            # run write_eof() -> close() and raise RuntimeError("Call .prepare()
            # first"), which would just be re-logged as a fresh "Unhandled exception"
            # traceback. Mark the response finalized so finish_response() is a clean
            # no-op (prepare() short-circuits on _payload_writer, write_eof() returns
            # immediately on _eof_sent). Report at debug level, skip registration.
            logger.debug(f"runs-stream: client gone during handshake: {e}")
            ws._eof_sent = True
            return ws
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
                "rejected_24h": 0,
                "per_stage_avg_ms": {"vad": None, "wakeword": None, "stt": None, "llm": None, "stress": None, "tts": None},
                "per_chain": {},
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
            web.get("/api/wake-models", self._get_wake_models),
            web.post("/api/wake-models", self._post_wake_model),
            web.post("/api/tts/test", self._post_tts_test),
            web.post("/api/assistant-tasks/reply", self._post_assistant_task_reply),
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
            web.get("/api/device/wake-probability", self._get_wake_probability),
            web.post("/api/device/wake-probability", self._post_wake_probability),
            web.post("/api/device/detect", self._post_detect),
            web.post("/api/device/model", self._post_model),
            web.post("/api/device/say", self._post_say),
            web.post("/api/capture", self._post_capture),
            web.get("/api/capture", self._get_capture_status),
            web.get("/api/capture/result", self._get_capture_result),
            web.get("/api/captures", self._get_captures),
            web.get("/api/captures/{device}/{name}.wav", self._get_capture_wav),
            web.delete("/api/captures/{device}/{name}", self._delete_capture),
            web.get("/api/device-logs", self._get_device_logs),
            web.get("/api/device-logs/{device}/{day}", self._get_device_log),
            web.get("/api/datasets", self._get_datasets),
            web.post("/api/datasets/{name}", self._post_dataset),
            web.post("/api/datasets/{name}/stop", self._post_dataset_stop),
            web.get("/api/datasets/{name}.zip", self._get_dataset_zip),
            web.get("/api/tools", self._get_tools),
            web.get("/api/runs", self._get_runs),
            web.get("/api/runs/stream", self._runs_stream),  # before {id}: literal wins
            web.get("/api/runs/{id}/audio", self._get_run_audio),
            web.get("/api/runs/{id}/tts-audio", self._get_run_tts_audio),
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

        async def _close_dataset_jobs(_app: web.Application) -> None:
            # Cancel any in-flight dataset tasks on shutdown so cleanup() doesn't wait on
            # them (and so they don't outlive the server).
            if self._dataset_jobs is not None:
                await self._dataset_jobs.close()
        app.on_shutdown.append(_close_dataset_jobs)
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
