"""Application composition root: boots from data/config.json and runs forever."""

import asyncio
import contextlib
import json
import os
import time

import httpx
import zeroconf
from loguru import logger

import src.plugins  # noqa: F401  triggers provider registration
from src import config_store
from src.audio_server import AudioServer
from src.config_service import ConfigDoc, ConfigService
from src.esphome_client import DeviceManager
from src.panel_api import PanelServer
from src.plugins.base import Deps
from src.reconfig import Reconfigurator
from src.run_events import RunEventsHub
from src.runs_store import RunsStore
from src.runtime import Runtime
from src.tool_factory import build_sources
from src.tool_hub import ToolHub
from src.version import __version__


def migrate_vad_plugin(doc: dict) -> bool:
    """One-time migration: move the WebRTC-specific VAD knobs out of core.vad into
    the vad/webrtc stage instance. Returns True when the doc was modified.

    Old docs carried `core.vad.aggressiveness` (every pre-migration save wrote the
    full core.vad section, so its presence is the reliable old-doc marker). Its value
    moves to vad.instances.webrtc.aggressiveness and the core key is DELETED. In the
    same pass, a true `core.vad.mic_normalize` is COPIED (not moved) to
    vad.instances.webrtc.auto_gain: both the pre-STT normalization (still gated by
    mic_normalize) and the VAD decision boost (now the plugin's auto_gain) used to be
    gated by that one toggle, so copying keeps the old behavior intact. The copy is
    gated on the aggressiveness marker so a later panel change to auto_gain is never
    overwritten on a subsequent boot.
    """
    core = doc.get("core")
    core_vad = core.get("vad") if isinstance(core, dict) else None
    if not isinstance(core_vad, dict) or "aggressiveness" not in core_vad:
        return False
    slot = doc.setdefault("vad", {"selected": "webrtc", "instances": {}})
    inst = slot.setdefault("instances", {}).setdefault("webrtc", {})
    inst["aggressiveness"] = core_vad.pop("aggressiveness")
    if core_vad.get("mic_normalize"):
        inst["auto_gain"] = True
    return True


def load_or_create_config() -> dict:
    """Return the config document, creating data/config.json from the template on first boot."""
    doc = config_store.load()
    if not doc:
        with open("templates/default_config.json", encoding="utf-8") as f:
            doc = json.load(f)
        config_store.save(doc, config_store.DEFAULT_PATH)
        logger.info("created default config at data/config.json")
    if migrate_vad_plugin(doc):
        config_store.save(doc, config_store.DEFAULT_PATH)
        logger.info(
            "config migrated: core.vad.aggressiveness (and the mic_normalize VAD-boost "
            "half as auto_gain) moved to the vad/webrtc plugin instance"
        )
    return doc


def warn_legacy_mcp(doc: dict) -> None:
    """Warn when the deprecated 'core.mcp' key is present in the raw config doc.

    The legacy single-server 'core.mcp' key was replaced by the 'core.mcp_servers'
    list; it is silently dropped on parse. Surface that once so the drop is visible.
    """
    core = doc.get("core")
    if isinstance(core, dict) and "mcp" in core:
        logger.warning(
            "config: legacy 'core.mcp' is ignored — add external servers under "
            "'core.mcp_servers' (panel: Tool sources)."
        )


def validate_boot_config(core) -> None:
    """Warn when the parsed config would make speakers play nothing.

    Fail loudly: an empty public_base_url yields host-less TTS URLs that speakers
    silently fail to fetch (the run otherwise logs as successful).
    """
    if not core.audio.public_base_url:
        logger.warning(
            f"audio.public_base_url is empty — speakers will receive host-less TTS "
            f"URLs (e.g. /tts/<id>.mp3) and play nothing. Set core.audio.public_base_url "
            f"in data/config.json (e.g. http://<this-host>:{core.audio.port})."
        )


async def main() -> None:
    """Build shared dependencies, start all speakers, and run until cancelled."""
    started_at = time.time()

    doc = load_or_create_config()

    warn_legacy_mcp(doc)

    core = ConfigDoc(**doc).core  # parse once to read proxy/timeout for Deps

    # client_ext (proxied) -> cloud STT/LLM + weather; client_local -> local TTS.
    client_ext = httpx.AsyncClient(proxy=(core.network.external_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)
    deps = Deps(http_cloud=client_ext, http_local=client_local, tts_timeout=core.tts_timeout)

    svc = ConfigService(doc, deps)
    core = svc.core
    vad_backend = svc.create("vad")
    stt_backend = svc.create("stt")
    tts_backend = svc.create("tts")
    llm_backend = svc.create("llm")

    # Observability: persist every finalized pipeline run to SQLite (gated on config).
    # Pruned once at boot; the panel API serves the run log + 24h metrics from it.
    runs_store = None
    if core.runs.enabled:
        runs_store = RunsStore(os.path.join(config_store.DATA_DIR, "runs.db"))
        runs_store.prune(now=time.time(), retention_days=core.runs.retention_days)

    # Live run stream: a broadcast hub shared by the pipeline (producer) and the
    # panel WebSocket endpoint (consumers). Cheap; always created.
    run_events = RunEventsHub()

    validate_boot_config(core)

    audio_server = AudioServer(core.audio.host, core.audio.port, core.audio.ttl)
    await audio_server.start()

    # Built-in reminders MCP (one-shot voice reminders). The scheduler needs the device
    # manager to deliver; the manager needs the hub that holds this source — so the
    # deliver callback is late-bound after the manager is constructed (below). The
    # scheduler is built here (before build_sources) because it gates the reminders
    # source; build_sources omits reminders when scheduler is None.
    reminders_store = None
    scheduler = None
    if core.reminders.enabled:
        from src.reminders import ReminderScheduler, RemindersStore
        reminders_store = RemindersStore(os.path.join(config_store.DATA_DIR, "reminders.db"))
        scheduler = ReminderScheduler(reminders_store)

    # Multi-source tool hub: an arbitrary list of external MCP servers (one
    # HttpMcpSource each) plus in-process built-in MCP servers (openweathermap first).
    # The source list is built by the shared build_sources() factory so boot and
    # hot-reload produce exactly the same set. Built only here, in the same task as
    # stop(), per anyio cancel-scope rules. A source failing to start is handled
    # gracefully inside ToolHub.start(). The weather tool is gated on an OWM api key;
    # built-in OpenWeatherMap uses the proxied client_ext for its OWM call. Each
    # external server's source id is its (unique) name.
    sources = build_sources(core, client_ext, scheduler)
    hub = ToolHub(sources)
    await hub.start()

    # Mutable runtime holder shared by reference across all pipelines. Live config
    # is read THROUGH it (via svc); backends/subsystems are swappable attributes for
    # later hot-reload tiers. Built once hub/audio_server exist, before the manager.
    rt = Runtime(
        svc,
        vad_backend=vad_backend,
        stt_backend=stt_backend, llm_backend=llm_backend, tts_backend=tts_backend,
        hub=hub, audio_server=audio_server,
        runs_store=runs_store, run_events=run_events,
    )

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(zc, rt)
    # Back-refs so the Reconfigurator can reach these subsystems on hot-reload.
    rt.zc = zc
    rt.manager = manager

    # Hot-reload coordinator: classifies each config change and applies what can be
    # applied live. Registered as a ConfigService change callback. Heavy
    # rebuilds (e.g. backends) are queued here and drained by run_loop in the main
    # task, off the panel request task. Reuses the SAME deps bag built above.
    reconfig_queue: asyncio.Queue = asyncio.Queue()
    reconf = Reconfigurator(rt, deps, reconfig_queue)
    svc.on_change(reconf.on_config_change)

    # Late-bind delivery now that the manager exists (resolves the circular dependency).
    if scheduler is not None:
        scheduler.deliver = manager.announce
    rt.scheduler = scheduler                 # may be None
    rt.reminders_store = reminders_store     # may be None; a hot toggle may swap both

    # Admin panel HTTP API. Serves the built frontend if it has been bundled into
    # frontend/react-export/dist; otherwise runs API-only. Constructed and started
    # INSIDE the try below so a failed start (e.g. port already taken) still runs
    # the finally cleanup for the resources opened above.
    static_dir = "frontend/react-export/dist"
    panel = None
    reconfig_task = None

    # The admin panel's bind host/port are the one setting NOT in the JSON config:
    # they come from env (applied at process start), since nothing in the config
    # requires a restart anymore.
    panel_host = os.environ.get("PANEL_HOST", "0.0.0.0")
    panel_port = int(os.environ.get("PANEL_PORT", "8201"))

    try:
        panel = PanelServer(
            svc, panel_host, panel_port,
            version=__version__, started_at=started_at,
            device_status=manager.statuses,
            device_capture=manager.capture,
            device_play=manager.play_chime,
            device_controls_get=manager.device_controls,
            device_controls_set=manager.set_device_control,
            static_dir=static_dir if os.path.isdir(static_dir) else None,
            runs_store=runs_store,
            tool_sources=hub.describe,
            run_events=run_events,
        )
        # Back-ref so the Reconfigurator can re-point the panel's runs-store at a
        # hot-swapped store. Set before the reconfig loop can run.
        rt.panel = panel
        await panel.start()
        await manager.start()
        if scheduler is not None:
            await scheduler.start()
        # Drain heavy reconfiguration jobs in this (main) task so blocking model
        # loads stay off the panel request task and off the event loop (to_thread).
        reconfig_task = asyncio.create_task(reconf.run_loop())
        # There is no in-app restart trigger anymore: block on a never-fired event
        # so the process runs until cancelled by a signal (SIGINT/SIGTERM).
        await asyncio.Event().wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        # Stop draining reconfiguration jobs before tearing down the resources the
        # jobs touch. May be unbound if main() failed before the task was created.
        if reconfig_task is not None:
            reconfig_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reconfig_task
        if panel is not None:
            await panel.stop()
        # Tear down the LIVE scheduler/store: a reminders hot toggle may have swapped
        # rt.scheduler/rt.reminders_store away from the boot-time locals (or to None).
        if rt.scheduler is not None:
            await rt.scheduler.stop()
        await manager.stop()
        await hub.stop()
        await audio_server.stop()
        # Close the CURRENT clients from deps: _rebuild_http may have swapped (and
        # already closed) the boot-time client_ext local, so deps holds the live ones.
        # httpx aclose() is idempotent, so re-closing an already-closed client is safe.
        await deps.http_cloud.aclose()
        await deps.http_local.aclose()
        if runs_store is not None:
            runs_store.close()
        if rt.reminders_store is not None:
            rt.reminders_store.close()
        zc.close()
