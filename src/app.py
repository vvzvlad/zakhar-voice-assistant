"""Application composition root: boots from data/config.json and runs forever."""

import asyncio
import contextlib
import json
import os
import time

import httpx
from loguru import logger

import src.plugins  # noqa: F401  triggers provider registration
from src import config_store, device_logs
from src.agent_mcp import AgentMcpEndpoint
from src.assistant_tasks import AssistantTasks, AssistantTasksStore
from src.chains import referenced_instances
from src.config_service import ConfigDoc, ConfigService
from src.device_server import DeviceManager
from src.mdns_advert import MdnsAdvert
from src.panel_api import PanelServer
from src.plugins.base import Deps
from src.prompt_store import PromptStore
from src.reconfig import Reconfigurator
from src.run_events import RunEventsHub
from src.runs_store import RunsStore
from src.runtime import Runtime
from src.tool_factory import build_sources
from src.tool_hub import ToolHub
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
    started_at = time.time()

    doc = load_or_create_config()

    core = ConfigDoc(**doc).core  # parse once to read proxy/timeout for Deps

    # client_ext (proxied) -> cloud STT/LLM + weather; client_local -> local TTS.
    client_ext = httpx.AsyncClient(proxy=(core.network.external_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)
    deps = Deps(http_cloud=client_ext, http_local=client_local, tts_timeout=core.tts_timeout)

    svc = ConfigService(doc, deps)
    # Bound before any backend is built: the command handlers read the command catalog
    # (core.commands) through deps.commands — simple-nlu when it is built, jev on every
    # request; the jev follow-up check reads the jev handler's model and key through
    # deps.instance_config.
    deps.instance_config = svc.instance_config
    deps.commands = lambda: svc.core.commands
    core = svc.core
    vad_backend = svc.create("vad")
    # One STT / handler backend per provider an enabled recognition chain references.
    pools = {"stt": {}, "llm": {}}
    for cat, pid in sorted(referenced_instances(core.chains)):
        pools[cat][pid] = svc.create_instance(cat, pid)
    # One TTS backend per enabled variant.
    tts_backends = {pid: svc.create_instance("tts", pid) for pid in svc.tts_variants()}
    # The follow-up check loads no model, so it is built at boot, not deferred.
    followup_backend = svc.create("followup")
    # Defer the accent (stress) backend: RuAccent loads an ONNX model from
    # disk/HuggingFace, which would block boot for seconds. Start with no stress
    # backend (the pipeline skips the stage) and load it in the background after the
    # panel + drain loop are up (see request_initial_load below), so boot stays
    # responsive and the load is surfaced via reloading() in the UI.
    stress_backend = None
    # Defer the wakeword backend too: the Vosk grammar verifier loads a model from
    # disk, which would block boot. Start with no wakeword backend (the pipeline
    # accepts/skips the gate) and load it in the background after boot (see
    # request_initial_load below), like the stress stage.
    wakeword_backend = None
    # Startup log: name the exact backend ("provider/model") behind every stage.
    stages = [("vad", vad_backend)]
    stages += [(cat, b) for cat in ("stt", "llm") for b in pools[cat].values()]
    stages += [("tts", b) for b in tts_backends.values()]
    stages += [("followup", followup_backend)]
    for cat, b in stages:
        logger.info(f"{cat} backend: {getattr(b, 'backend_desc', type(b).__name__)}")
    logger.info("stress/wakeword backends: deferred — load in the background after boot (if enabled)")

    # Observability: persist every finalized pipeline run to SQLite (gated on config).
    # Pruned once at boot; the panel API serves the run log + 24h metrics from it.
    runs_store = None
    if core.runs.enabled:
        runs_store = RunsStore(os.path.join(config_store.DATA_DIR, "runs.db"))
        runs_store.prune(now=time.time(), retention_days=core.runs.retention_days)
    # A device's log files are pruned when its new day's file starts, and here for every device,
    # so one that stopped sending does not keep its old days.
    device_logs.prune(core.device_log_days)

    # Named system-prompt profiles (always on — the pipeline reads the active
    # profile's text as the LLM system prompt). On first boot the empty DB is
    # seeded from the legacy data/system_prompt.md file (kept as a backup) or,
    # absent that, from the committed default template.
    prompt_store = PromptStore(
        os.path.join(config_store.DATA_DIR, "prompts.db"),
        seed_path=core.prompt.system_prompt_path,
    )

    # Live run stream: a broadcast hub shared by the pipeline (producer) and the
    # panel WebSocket endpoint (consumers). Cheap; always created.
    run_events = RunEventsHub()

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

    # Jobs handed off to the ZeroClaw assistant. Always built: the panel takes
    # its replies even while the tool source is off. The service reads its config and the
    # speakers through `rt`, which is late-bound below (the tool source built off it
    # exists before the Runtime does).
    assistant_tasks_store = AssistantTasksStore(os.path.join(config_store.DATA_DIR, "assistant_tasks.db"))
    assistant_tasks = AssistantTasks(assistant_tasks_store)

    # Multi-source tool hub: an arbitrary list of external MCP servers (one
    # HttpMcpSource each) plus in-process built-in MCP servers (openweathermap first).
    # The source list is built by the shared build_sources() factory so boot and
    # hot-reload produce exactly the same set. Built only here, in the same task as
    # stop(), per anyio cancel-scope rules. A source failing to start is handled
    # gracefully inside ToolHub.start(). The weather tool is gated on an OWM api key;
    # built-in OpenWeatherMap uses the proxied client_ext for its OWM call. Each
    # external server's source id is its (unique) name.
    # The volume tool reads the manager at call time: `rt` is assigned further down.
    sources = build_sources(core, client_ext, scheduler, assistant_tasks, lambda: rt.manager)
    hub = ToolHub(sources)
    await hub.start()

    # Mutable runtime holder shared by reference across all pipelines. Live config
    # is read THROUGH it (via svc); backends/subsystems are swappable attributes for
    # later hot-reload tiers. Built once the hub exists, before the manager.
    rt = Runtime(
        svc,
        vad_backend=vad_backend,
        stt_backends=pools["stt"], llm_backends=pools["llm"], tts_backends=tts_backends,
        stress_backend=stress_backend, wakeword_backend=wakeword_backend,
        followup_backend=followup_backend,
        hub=hub,
        runs_store=runs_store, run_events=run_events,
        prompt_store=prompt_store,
    )
    assistant_tasks.rt = rt
    rt.assistant_tasks = assistant_tasks

    # The device server (/device WebSocket + /health on core.audio.host/port).
    manager = DeviceManager(rt)
    # Back-ref so the Reconfigurator can reach it on hot-reload.
    rt.manager = manager

    # Hot-reload coordinator: classifies each config change and applies what can be
    # applied live. Registered as a ConfigService change callback. Heavy
    # rebuilds (e.g. backends) are queued here and drained by run_loop in the main
    # task, off the panel request task. Reuses the SAME deps bag built above.
    reconfig_queue: asyncio.Queue = asyncio.Queue()
    reconf = Reconfigurator(rt, deps, reconfig_queue)
    svc.on_change(reconf.on_config_change)
    # Prepare the chain handlers for the tool catalog in the background (simple-nlu
    # trains its model on it), so neither boot nor the first request waits for it.
    # The reference keeps the task from being garbage-collected.
    warm_task = asyncio.create_task(reconf.warm_handlers())

    # Late-bind delivery now that the manager exists (resolves the circular dependency).
    if scheduler is not None:
        scheduler.deliver = manager.announce
    rt.scheduler = scheduler                 # may be None
    rt.reminders_store = reminders_store     # may be None; a hot toggle may swap both

    # Admin panel HTTP API. Serves the built frontend if it has been bundled into
    # frontend/dist; otherwise runs API-only. Constructed and started
    # INSIDE the try below so a failed start (e.g. port already taken) still runs
    # the finally cleanup for the resources opened above.
    static_dir = "frontend/dist"
    panel = None
    reconfig_task = None

    # The admin panel's bind host/port are the one setting NOT in the JSON config:
    # they come from env (applied at process start), since nothing in the config
    # requires a restart anymore.
    panel_host = os.environ.get("PANEL_HOST", "0.0.0.0")
    panel_port = int(os.environ.get("PANEL_PORT", "8201"))

    # Agent-facing MCP endpoint, served BY the panel at /mcp on the panel port
    # (external agents drive the assistant over streamable HTTP). Reads everything
    # live through `rt`; the core.agent_mcp.enabled toggle is checked per request.
    # Auxiliary: a failed build must not break boot — the panel then answers 503.
    agent_mcp = None
    try:
        agent_mcp = AgentMcpEndpoint(rt)
    except Exception as e:
        logger.error(f"agent MCP endpoint failed to build: {e} — continuing without it")

    try:
        panel = PanelServer(
            svc, panel_host, panel_port,
            version=__version__, started_at=started_at,
            device_status=manager.statuses,
            device_capture=manager.capture,
            device_capture_start=manager.capture_start,
            device_capture_status=manager.capture_status,
            device_capture_delete=manager.capture_delete,
            device_play=manager.play_chime,
            device_controls_get=manager.device_controls,
            device_controls_set=manager.set_device_control,
            device_wake_prob_get=manager.wake_prob,
            device_wake_prob_set=manager.set_wake_prob_stream,
            device_detect_set=manager.set_detect,
            device_model_reload=manager.reload_model,
            device_say=manager.say,
            static_dir=static_dir if os.path.isdir(static_dir) else None,
            runs_store=runs_store,
            tool_sources=hub.describe,
            run_events=run_events,
            prompt_store=prompt_store,
            agent_mcp=agent_mcp,
            assistant_task_reply=assistant_tasks.on_reply,
            reload_status=reconf.reloading,
        )
        # Back-ref so the Reconfigurator can re-point the panel's runs-store at a
        # hot-swapped store. Set before the reconfig loop can run.
        rt.panel = panel
        await manager.start()
        # Devices find the device server by mDNS; a rebind restarts the advert (Reconfigurator).
        advert = MdnsAdvert()
        await advert.start(rt.core.audio.advertise_host or rt.core.audio.host, manager.port)
        rt.advert = advert
        # panel.start()/stop() own the agent MCP endpoint's lifecycle too.
        await panel.start()
        if scheduler is not None:
            await scheduler.start()
        # Drain heavy reconfiguration jobs in this (main) task so blocking model
        # loads stay off the panel request task and off the event loop (to_thread).
        reconfig_task = asyncio.create_task(reconf.run_loop())
        # Now that the panel is serving and the drain loop is running, start the deferred
        # accent (stress) model load in the background. It rebuilds via the hot-reload
        # path: marks reloading() (the panel shows "Loading… Accents") and swaps the real
        # backend into the runtime when ready; the stage is skipped until then.
        reconf.request_initial_load({"stress", "wakeword"})
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
        if rt.advert is not None:
            await rt.advert.stop()
        await manager.stop()
        await hub.stop()
        # Close the CURRENT clients from deps: _rebuild_http may have swapped (and
        # already closed) the boot-time client_ext local, so deps holds the live ones.
        # httpx aclose() is idempotent, so re-closing an already-closed client is safe.
        await deps.http_cloud.aclose()
        await deps.http_local.aclose()
        if runs_store is not None:
            runs_store.close()
        prompt_store.close()
        assistant_tasks_store.close()
        if rt.reminders_store is not None:
            rt.reminders_store.close()
