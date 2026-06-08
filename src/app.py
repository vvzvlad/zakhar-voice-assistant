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
from src.run_events import RunEventsHub
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

    # The legacy single-server 'core.mcp' key was replaced by the 'core.mcp_servers'
    # list; it is silently dropped on parse. Surface that once so the drop is visible.
    if isinstance(doc.get("core"), dict) and "mcp" in doc["core"]:
        logger.warning(
            "config: legacy 'core.mcp' is ignored — add external servers under "
            "'core.mcp_servers' (panel: Tool sources)."
        )

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

    # Live run stream: a broadcast hub shared by the pipeline (producer) and the
    # panel WebSocket endpoint (consumers). Cheap; always created.
    run_events = RunEventsHub()

    # Fail loudly: an empty public_base_url yields host-less TTS URLs that speakers
    # silently fail to fetch (the run otherwise logs as successful).
    if not core.audio.public_base_url:
        logger.warning(
            f"audio.public_base_url is empty — speakers will receive host-less TTS "
            f"URLs (e.g. /tts/<id>.mp3) and play nothing. Set core.audio.public_base_url "
            f"in data/config.json (e.g. http://<this-host>:{core.audio.port})."
        )

    audio_server = AudioServer(core.audio.host, core.audio.port, core.audio.ttl)
    await audio_server.start()

    # Multi-source tool hub: an arbitrary list of external MCP servers (one
    # HttpMcpSource each) plus in-process built-in MCP servers (weather first). Built
    # only here, in the same task as stop(), per anyio cancel-scope rules. A source
    # failing to start is handled gracefully inside ToolHub.start(). The weather tool
    # is gated on an OWM api key; built-in weather uses the proxied client_ext for its
    # OWM call. Each external server's source id is its (unique) name.
    sources = []
    for srv in core.mcp_servers:
        if srv.url and srv.name:
            sources.append(HttpMcpSource(srv.name, McpToolHub(srv.url, srv.token or None, srv.transport)))
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
    # Built-in reminders MCP (one-shot voice reminders). The scheduler needs the device
    # manager to deliver; the manager needs the hub that holds this source — so the
    # deliver callback is late-bound after the manager is constructed (below).
    reminders_store = None
    scheduler = None
    if core.reminders.enabled:
        from src.reminders import ReminderScheduler, RemindersStore
        reminders_store = RemindersStore(os.path.join(core.context.dir, "reminders.db"))
        scheduler = ReminderScheduler(reminders_store)
        from src.builtin_mcp.reminders import build_reminders_server
        sources.append(BuiltinMcpSource("reminders", build_reminders_server(scheduler)))
    hub = ToolHub(sources)
    await hub.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(
        zc, hub, stt_backend, llm_backend, tts_backend, audio_server,
        core, llm_cfg, runs_store=runs_store, run_events=run_events,
    )

    # Late-bind delivery now that the manager exists (resolves the circular dependency).
    if scheduler is not None:
        scheduler.deliver = manager.announce

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
            tool_sources=hub.describe,
            run_events=run_events,
        )
        await panel.start()
        await manager.start()
        if scheduler is not None:
            await scheduler.start()
        # Block until POST /api/restart sets the event (or the task is cancelled).
        # docker `restart: always` brings the process back after the clean exit.
        await restart_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        if panel is not None:
            await panel.stop()
        if scheduler is not None:
            await scheduler.stop()
        await manager.stop()
        await hub.stop()
        await audio_server.stop()
        await client_ext.aclose()
        await client_local.aclose()
        if runs_store is not None:
            runs_store.close()
        if reminders_store is not None:
            reminders_store.close()
        zc.close()
