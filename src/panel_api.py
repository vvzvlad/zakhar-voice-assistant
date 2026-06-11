"""Admin-panel HTTP API (aiohttp).

Runs in a TRUSTED zone — there is NO authentication. The panel is driven entirely
by ConfigService, so it carries no provider-specific knowledge: it exposes the
catalog, the raw config document, generic patch/options endpoints, the system
prompt file, system/version info and live device status.

Start/stop mirror AudioServer (AppRunner + TCPSite). CORS is restricted to the
hardcoded `_ALLOWED_ORIGINS` allowlist (empty — see below); it never reflects a
wildcard, so an arbitrary web page can't read the unauthenticated config (which
carries plaintext secrets) via the operator's browser. The prod frontend is
same-origin and the dev Vite server proxies /api, so neither needs CORS.
"""

import asyncio
import contextlib
import io
import os
import time
import wave
from datetime import datetime, timezone

import numpy as np
from aiohttp import web
from loguru import logger
from pydantic import ValidationError

from src import config_store
from src.capture_jobs import CaptureJobManager
from src.pipeline import CAPTURE_MAX_SECONDS, CaptureBusyError
from src.prompt import load_system_prompt, save_system_prompt
from src.runs_store import db_file_size

# HTTP methods the API exposes (used by the CORS preflight headers).
_ALLOW_METHODS = "GET, POST, PATCH, PUT, OPTIONS"

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
                 run_events=None, heartbeat_interval: float = 1.0):
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
        try:
            options = self.svc.options(category, plugin, field) or []
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=404)
        return web.json_response({"options": options})

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

    async def _get_prompt(self, request: web.Request) -> web.Response:
        path = self.svc.core.prompt.system_prompt_path
        return web.json_response({"path": path, "text": load_system_prompt(path)})

    async def _put_prompt(self, request: web.Request) -> web.Response:
        body = await _read_json(request)
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            return web.json_response(
                {"error": 'field "text" (string) is required'}, status=400
            )
        text = body["text"]
        path = self.svc.core.prompt.system_prompt_path
        save_system_prompt(path, text)
        return web.json_response({"ok": True})

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
            web.get("/api/prompt", self._get_prompt),
            web.put("/api/prompt", self._put_prompt),
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
            app.router.add_get("/{path:(?!api/).*}", self._spa_index)

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
