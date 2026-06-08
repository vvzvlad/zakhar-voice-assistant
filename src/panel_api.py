"""Admin-panel HTTP API (aiohttp).

Runs in a TRUSTED zone — there is NO authentication. The panel is driven entirely
by ConfigService, so it carries no provider-specific knowledge: it exposes the
catalog, the raw config document, generic patch/options endpoints, the system
prompt file, system/version info, a restart trigger and live device status.

Start/stop mirror AudioServer (AppRunner + TCPSite). CORS is restricted to the
hardcoded `_ALLOWED_ORIGINS` allowlist (empty — see below); it never reflects a
wildcard, so an arbitrary web page can't read the unauthenticated config (which
carries plaintext secrets) via the operator's browser. The prod frontend is
same-origin and the dev Vite server proxies /api, so neither needs CORS.
"""

import asyncio
import os
import time
from datetime import datetime, timezone

from aiohttp import web
from loguru import logger
from pydantic import ValidationError

from src.pipeline import CAPTURE_MAX_SECONDS, CaptureBusyError, CaptureEmptyError
from src.prompt import load_system_prompt, save_system_prompt

# HTTP methods the API exposes (used by the CORS preflight headers).
_ALLOW_METHODS = "GET, POST, PATCH, PUT, OPTIONS"

# Browser origins allowed cross-origin access to the (unauthenticated) panel API.
# Hardcoded empty: there is no cross-origin reflection. The prod frontend is
# same-origin and the dev Vite server proxies /api, so neither needs CORS — and an
# empty allowlist guarantees the unauthenticated config (with plaintext secrets) is
# never readable from an arbitrary web page. Not a config knob.
_ALLOWED_ORIGINS: frozenset[str] = frozenset()


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


async def _read_json(request: web.Request):
    """Parse a JSON request body, raising HTTP 400 on malformed input."""
    try:
        return await request.json()
    except Exception:
        raise web.HTTPBadRequest(text='{"error": "invalid JSON body"}',
                                 content_type="application/json")


class PanelServer:
    def __init__(self, svc, host, port, *, version, started_at,
                 restart_event, device_status=None, device_capture=None,
                 static_dir=None, runs_store=None, tool_sources=None,
                 run_events=None, pending_restart=None):
        # svc: ConfigService; started_at: float (time.time()); restart_event: asyncio.Event
        # device_status: optional callable -> list[dict]; static_dir: optional path to built frontend
        # device_capture: optional async callable (device_name, seconds) -> bytes that records a
        #   manual sample and returns it as WAV bytes (DeviceManager.capture). None -> /api/capture 503.
        # runs_store: optional RunsStore for the observability endpoints (None -> empty/zeros)
        # tool_sources: optional zero-arg callable -> list[dict] (ToolHub.describe()), None -> []
        # run_events: optional RunEventsHub for the live WS run stream (None -> WS closes)
        # pending_restart: optional zero-arg callable -> bool (the Reconfigurator's flag);
        #   None -> always reports False (e.g. in tests without a reconfigurator).
        self.svc = svc
        self.host = host
        self.port = port
        self.version = version
        self.started_at = started_at
        self.restart_event = restart_event
        self.device_status = device_status
        self.device_capture = device_capture
        self.static_dir = static_dir
        self.runs_store = runs_store
        self.tool_sources = tool_sources
        self.run_events = run_events
        self.pending_restart = pending_restart
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

    async def _get_system(self, request: web.Request) -> web.Response:
        started = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
        return web.json_response({
            "version": self.version,
            "started": started,
            "uptime_seconds": int(time.time() - self.started_at),
            "running": True,
            "log_level": self.svc.core.log_level,
            "pending_restart": bool(self.pending_restart()) if self.pending_restart else False,
        })

    async def _post_restart(self, request: web.Request) -> web.Response:
        self.restart_event.set()
        return web.json_response({"restarting": True}, status=202)

    async def _get_devices(self, request: web.Request) -> web.Response:
        return web.json_response(self.device_status() if self.device_status else [])

    async def _post_capture(self, request: web.Request) -> web.Response:
        """Record a manual sample on a speaker and stream the WAV straight back.

        Body: {"device": "<name>", "seconds": <int 1..300>}. The device records that
        many seconds of mic audio (no STT/LLM/TTS); the recording is EPHEMERAL —
        device_capture returns the WAV bytes in memory and we hand them to the
        caller as an audio/wav attachment download. Nothing is kept on the server.

        Status codes:
          - 200 success (WAV body)
          - 400 bad input
          - 404 unknown device
          - 409 not actionable now: device offline / no capture entities / a capture
                is already in progress on that device (CaptureBusyError)
          - 500 server-side capture failure, e.g. the recording produced no audio
                (CaptureEmptyError) or any other unexpected error
          - 503 capture not wired (no DeviceManager)
          - 504 the recording timed out (device never streamed)
        The handler may block for up to ~seconds + margin while the device streams
        the audio — that is expected.
        """
        if self.device_capture is None:
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
            wav = await self.device_capture(device, seconds)
        except LookupError:
            return web.json_response(
                {"error": f"unknown device {device!r}"}, status=404
            )
        except TimeoutError as e:
            # The device never streamed the audio in time -> gateway timeout.
            return web.json_response({"error": str(e)}, status=504)
        except CaptureBusyError as e:
            # A capture is already armed/in-flight on this device -> conflict.
            return web.json_response({"error": str(e)}, status=409)
        except CaptureEmptyError as e:
            # The capture ran but produced no audio -> server-side capture failure,
            # NOT a "not actionable now" (409) condition.
            return web.json_response({"error": str(e)}, status=500)
        except RuntimeError as e:
            # Offline device or missing firmware entities -> not actionable right now.
            return web.json_response({"error": str(e)}, status=409)
        except Exception as e:
            # Any other unexpected capture failure -> server error.
            return web.json_response({"error": str(e)}, status=500)
        # Sanitize the device name for the download filename (ascii alnum/._- only).
        safe_device = "".join(
            c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in device
        ) or "device"
        filename = f"zakhar_{safe_device}_{seconds}s.wav"
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
            web.get("/api/prompt", self._get_prompt),
            web.put("/api/prompt", self._put_prompt),
            web.get("/api/system", self._get_system),
            web.post("/api/restart", self._post_restart),
            web.get("/api/devices", self._get_devices),
            web.post("/api/capture", self._post_capture),
            web.get("/api/tools", self._get_tools),
            web.get("/api/runs", self._get_runs),
            web.get("/api/runs/stream", self._runs_stream),  # before {id}: literal wins
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

        async def _close_ws_clients(_app: web.Application) -> None:
            # Fired by AppRunner.cleanup() BEFORE it waits (up to shutdown_timeout) for
            # in-flight handlers. Closing the live run-stream sockets here unblocks the
            # _runs_stream handler so shutdown (and Ctrl+C) completes immediately instead
            # of hanging until the timeout elapses.
            if self.run_events is not None:
                await self.run_events.close_all()
        app.on_shutdown.append(_close_ws_clients)
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
