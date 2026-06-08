"""Application composition root: boots from data/config.json and runs forever."""

import asyncio
import json
import os
import time

import httpx
import zeroconf
from loguru import logger

import src.plugins  # noqa: F401  triggers provider registration
from src import config_store
from src.audio_server import AudioServer
from src.builtin_mcp.weather import build_weather_server
from src.config_service import ConfigDoc, ConfigService
from src.esphome_client import DeviceManager
from src.mcp_client import McpToolHub
from src.panel_api import PanelServer
from src.plugins.base import Deps
from src.runs_store import RunsStore
from src.tool_hub import BuiltinMcpSource, HttpMcpSource, ToolHub
from src.version import __version__


def load_or_create_config() -> dict:
    """Return the config document, creating data/config.json from the template on first boot."""
    doc = config_store.load()
    if not doc:
        with open("templates/default_config.json", encoding="utf-8") as f:
            doc = json.load(f)
        config_store.save(doc, config_store.DEFAULT_PATH)
        logger.info("created default config at data/config.json")
    return doc


async def main() -> None:
    """Build shared dependencies, start all speakers, and run until cancelled."""
    restart_event = asyncio.Event()
    started_at = time.time()

    doc = load_or_create_config()
    core = ConfigDoc(**doc).core  # parse once to read proxy/timeout for Deps

    # client_ext (proxied) -> cloud STT/LLM + weather; client_local -> local TTS.
    client_ext = httpx.AsyncClient(proxy=(core.network.external_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)
    deps = Deps(http_cloud=client_ext, http_local=client_local, tts_timeout=core.tts_timeout)

    svc = ConfigService(doc, deps)
    core = svc.core
    stt_backend = svc.create("stt")
    tts_backend = svc.create("tts")
    llm_backend = svc.create("llm")
    llm_cfg = svc.get("llm")

    # Observability: persist every finalized pipeline run to SQLite (gated on config).
    # Pruned once at boot; the panel API serves the run log + 24h metrics from it.
    runs_store = None
    if core.runs.enabled:
        runs_store = RunsStore(os.path.join(core.context.dir, "runs.db"))
        runs_store.prune(now=time.time(), retention_days=core.runs.retention_days)

    audio_server = AudioServer(core.audio.host, core.audio.port, core.audio.ttl)
    await audio_server.start()

    # Multi-source tool hub: the external smart-home MCP server (HttpMcpSource) plus
    # in-process built-in MCP servers (weather first). Built only here, in the same
    # task as stop(), per anyio cancel-scope rules. A source failing to start is
    # handled gracefully inside ToolHub.start(). The weather tool is gated on an OWM
    # api key; built-in weather uses the proxied client_ext for its OWM call.
    sources = []
    if core.mcp.url:
        sources.append(HttpMcpSource("home", McpToolHub(core.mcp.url, core.mcp.token or None)))
    if core.weather.api_key:
        sources.append(
            BuiltinMcpSource(
                "weather", build_weather_server(client_ext, core.weather.api_key, core.weather.city)
            )
        )
    # Built-in calendar MCP (CalDAV). Gated on url + username; the caldav lib is
    # synchronous, so its tools offload to a worker thread inside the server.
    if core.calendar.url and core.calendar.username:
        from src.builtin_mcp.calendar import CalendarClient, build_calendar_server
        cal_client = CalendarClient(core.calendar.url, core.calendar.username,
                                    core.calendar.password, core.calendar.calendar)
        sources.append(BuiltinMcpSource("calendar", build_calendar_server(cal_client)))
    hub = ToolHub(sources)
    await hub.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(
        zc, hub, stt_backend, llm_backend, tts_backend, audio_server,
        core, llm_cfg, runs_store=runs_store,
    )

    # Admin panel HTTP API. Serves the built frontend if it has been bundled into
    # frontend/react-export/dist; otherwise runs API-only. Constructed and started
    # INSIDE the try below so a failed start (e.g. port already taken) still runs
    # the finally cleanup for the resources opened above.
    static_dir = "frontend/react-export/dist"
    panel = None

    try:
        panel = PanelServer(
            svc, core.panel.host, core.panel.port,
            version=__version__, started_at=started_at,
            restart_event=restart_event, device_status=manager.statuses,
            static_dir=static_dir if os.path.isdir(static_dir) else None,
            runs_store=runs_store,
        )
        await panel.start()
        await manager.start()
        # Block until POST /api/restart sets the event (or the task is cancelled).
        # docker `restart: always` brings the process back after the clean exit.
        await restart_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        if panel is not None:
            await panel.stop()
        await manager.stop()
        await hub.stop()
        await audio_server.stop()
        await client_ext.aclose()
        await client_local.aclose()
        if runs_store is not None:
            runs_store.close()
        zc.close()
