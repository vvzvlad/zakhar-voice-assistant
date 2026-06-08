"""Admin-panel HTTP API (aiohttp).

Runs in a TRUSTED zone — there is NO authentication. The panel is driven entirely
by ConfigService, so it carries no provider-specific knowledge: it exposes the
catalog, the raw config document, generic patch/options endpoints, the system
prompt file, system/version info, a restart trigger and live device status.

Start/stop mirror AudioServer (AppRunner + TCPSite). A permissive wildcard CORS
middleware is fine here because the panel always runs in a trusted segment and the
Vite dev server lives on a different port.
"""

import asyncio
import os
import time
from datetime import datetime, timezone

from aiohttp import web
from loguru import logger
from pydantic import ValidationError

from src.prompt import load_system_prompt, save_system_prompt

# HTTP methods the API exposes (used by the CORS preflight headers).
_ALLOW_METHODS = "GET, POST, PATCH, PUT, OPTIONS"


def _add_cors(resp: web.StreamResponse) -> web.StreamResponse:
    """Add permissive wildcard CORS headers to a response (or HTTP exception)."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = _ALLOW_METHODS
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@web.middleware
async def _cors_middleware(request: web.Request, handler):
    """Permissive wildcard CORS (trusted zone) + 204 preflight for OPTIONS.

    CORS headers are attached to BOTH normal responses and raised HTTP
    exceptions (bad JSON -> 400, router 404, our 400s) so error responses stay
    readable by the browser frontend on a different port.
    """
    if request.method == "OPTIONS":
        return _add_cors(web.Response(status=204))
    try:
        return _add_cors(await handler(request))
    except web.HTTPException as ex:
        # ex is itself a Response subclass; mutate its headers in place.
        _add_cors(ex)
        raise


async def _read_json(request: web.Request):
    """Parse a JSON request body, raising HTTP 400 on malformed input."""
    try:
        return await request.json()
    except Exception:
        raise web.HTTPBadRequest(text='{"error": "invalid JSON body"}',
                                 content_type="application/json")


class PanelServer:
    def __init__(self, svc, host, port, *, version, started_at,
                 restart_event, device_status=None, static_dir=None,
                 runs_store=None):
        # svc: ConfigService; started_at: float (time.time()); restart_event: asyncio.Event
        # device_status: optional callable -> list[dict]; static_dir: optional path to built frontend
        # runs_store: optional RunsStore for the observability endpoints (None -> empty/zeros)
        self.svc = svc
        self.host = host
        self.port = port
        self.version = version
        self.started_at = started_at
        self.restart_event = restart_event
        self.device_status = device_status
        self.static_dir = static_dir
        self.runs_store = runs_store
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
            "pending_restart": self.svc.pending_restart,
        })

    async def _post_restart(self, request: web.Request) -> web.Response:
        self.restart_event.set()
        return web.json_response({"restarting": True}, status=202)

    async def _get_devices(self, request: web.Request) -> web.Response:
        return web.json_response(self.device_status() if self.device_status else [])

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
        app = web.Application(middlewares=[_cors_middleware])
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
            web.get("/api/runs", self._get_runs),
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
        return app

    async def start(self) -> None:
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"panel API on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
