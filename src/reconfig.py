"""Hot-reload coordinator: classify config changes and apply what can be applied
without a process restart.

`changed_paths` and `action_for` are pure functions (also used by ConfigService
to compute the change set). `Reconfigurator` is wired in app.py: it is registered
via ConfigService.on_change and decides, per change, whether the running process
can absorb it live or still needs a restart.
"""

import asyncio
import os
import time

import httpx
from loguru import logger

from src import config_store
from src.logging_setup import setup_logging
from src.runs_store import RunsStore
from src.tool_factory import build_sources


def changed_paths(old, new, prefix: str = "") -> set[str]:
    """Return dotted paths of leaves that differ between two config docs.

    Dicts are descended key-by-key (union of keys). A missing or non-dict side
    is treated as empty so we still descend into the present dict side and report
    SPECIFIC leaf paths rather than the parent. Non-dict values (including lists)
    are compared by equality and treated as a single leaf.
    """
    if isinstance(old, dict) or isinstance(new, dict):
        o = old if isinstance(old, dict) else {}
        n = new if isinstance(new, dict) else {}
        paths: set[str] = set()
        for k in set(o) | set(n):
            p = f"{prefix}.{k}" if prefix else k
            paths |= changed_paths(o.get(k), n.get(k), p)
        return paths
    return {prefix} if (old != new and prefix) else set()


# Live LLM provider fields read per request (not baked into the backend).
_LLM_LIVE_LEAVES = {"reply_rate_limit", "reply_empty", "reply_empty_after_tools",
                    "reply_error", "max_tool_rounds"}


def action_for(path: str) -> str:
    """Map one changed dotted path to a reconfiguration action.
    Actions: 'live', 'logging', 'rebuild_backends', 'rebuild_http',
    'rebuild_tools', 'rebuild_audio', 'rebuild_devices', 'rebuild_runs',
    'rebuild_reminders', 'restart'. Unknown paths default to 'restart' (safe)."""
    # Display-only "<field>_label" companion fields (the persisted human label of a
    # dynamic-select choice) are panel metadata that no backend reads, so an isolated
    # change needs no rebuild — apply it live. (In the normal flow the label changes
    # together with its value field, whose own path still drives the needed rebuild.)
    if path.endswith("_label"):
        return "live"
    # core.* rules (most specific first)
    if path == "core.log_level":
        return "logging"
    if path.startswith("core.context") or path.startswith("core.vad"):
        return "live"          # incl. vad.mic_channel/mic_normalize/mic_highpass (read via the Runtime read-through)
    if path.startswith("core.capture"):
        return "live"          # capture.enabled/dir are read per-run via the Runtime read-through
    if path.startswith("core.ack"):
        return "live"          # ack.enabled/sound_path are read per-run via the Runtime read-through
    if path in ("core.audio.public_base_url", "core.audio.ttl", "core.audio.stream_tts"):
        return "live"          # stream_tts is read per run via the Runtime read-through
    if path.startswith("core.audio"):            # host/port -> rebind
        return "rebuild_audio"
    if path.startswith("core.prompt"):           # system_prompt_path read live
        return "live"
    if path.startswith("core.network"):
        return "rebuild_http"        # external_proxy needs an http-client rebuild (Tier 3)
    if path == "core.tts_timeout":
        return "rebuild_backends"
    if (path.startswith("core.openweathermap") or path.startswith("core.calendar")
            or path.startswith("core.mcp_servers")):
        return "rebuild_tools"
    if path.startswith("core.agent_mcp"):
        return "live"          # enabled is read per /mcp request through the panel
    if path.startswith("core.reminders"):
        return "rebuild_reminders"
    if path.startswith("core.runs"):
        return "rebuild_runs"
    if path.startswith("core.devices") or path.startswith("core.esphome"):
        return "rebuild_devices"
    # stage providers
    if (path.startswith("vad") or path.startswith("wakeword") or path.startswith("stt")
            or path.startswith("tts") or path.startswith("stress")):
        return "rebuild_backends"
    if path.startswith("llm"):
        leaf = path.rsplit(".", 1)[-1]
        return "live" if leaf in _LLM_LIVE_LEAVES else "rebuild_backends"
    return "restart"


# Actions whose work is performed off the panel request task, in the queue-draining
# task (run_loop -> apply_job), because they may block (model loads, socket re-bind,
# SQLite open/prune, tool-source start network I/O, http-client rebuild, device
# reconnect start, reminders store open/close).
ASYNC_ACTIONS = {"rebuild_backends", "rebuild_audio", "rebuild_runs",
                 "rebuild_tools", "rebuild_http",
                 "rebuild_devices", "rebuild_reminders"}


def backend_categories(paths) -> set[str]:
    """Subset of {'vad','wakeword','stt','llm','stress','tts'} whose backend must be rebuilt for these paths."""
    cats = set()
    for p in paths:
        if action_for(p) != "rebuild_backends":
            continue
        top = p.split(".", 1)[0]
        if top in ("vad", "wakeword", "stt", "llm", "stress", "tts"):
            cats.add(top)
        elif p == "core.tts_timeout":
            cats.add("tts")
    return cats


class Reconfigurator:
    """Owns the runtime-side response to config changes."""

    def __init__(self, runtime, deps, queue):
        self.rt = runtime
        self.deps = deps
        self.queue = queue          # asyncio.Queue drained by run_loop in the main task
        self._reloading: set[str] = set()   # categories currently rebuilding

    def reloading(self) -> list[str]:
        """Backend categories whose rebuild (model load) is in flight right now.
        Read by the panel's system snapshot to surface a 'loading' state in the UI.
        Mutated only from the drain task; sorted for a stable payload."""
        return sorted(self._reloading)

    def request_initial_load(self, cats) -> None:
        """Queue a one-shot background (re)build of the given stage backend categories so a
        slow model load (e.g. the RuAccent stress model) does NOT block boot. Drained by
        run_loop in the main task: it marks reloading() while the load is in flight (the
        panel surfaces a 'loading' state) and swaps the real backend into the runtime when
        ready; until then the runtime keeps its boot value (None for stress -> the pipeline
        skips the stage). Routed through the SAME queue as live config changes so the warm-up
        is serialized and coalesced with any change that arrives during boot (one rebuild per
        category, last-writer-wins)."""
        # Synthetic "<cat>.__warmup__" path: action_for() maps any path under a stage
        # category to a backend rebuild and backend_categories() reduces it to {cat}; the
        # "__warmup__" leaf marks it as a boot warm-up rather than a real field change.
        self.queue.put_nowait({f"{c}.__warmup__" for c in cats})

    def on_config_change(self, paths) -> None:
        """ConfigService.on_change callback (SYNC, runs on the panel request task).
        Live fields are already effective through the Runtime read-through; here we
        (a) reinit logging if the level changed, (b) push live audio.ttl onto the
        running server, and (c) hand heavy/blocking rebuilds to the main-task queue.
        Must stay fast: no blocking work and no model loads happen here."""
        actions = {action_for(p) for p in paths}
        if "logging" in actions:
            setup_logging(self.rt.core.log_level)
        # audio.ttl is live-applied by mutating the running server (it caches ttl).
        if "core.audio.ttl" in paths and self.rt.audio_server is not None:
            self.rt.audio_server.ttl = self.rt.core.audio.ttl
        # Hand heavy/blocking rebuilds to the main-task queue (drained by run_loop).
        if actions & ASYNC_ACTIONS:
            self.queue.put_nowait(set(paths))

    async def run_loop(self) -> None:
        """Drain the reconfiguration queue until cancelled, applying jobs in THIS task.
        Coalesces a burst of queued jobs into one apply so each affected stage rebuilds
        at most once."""
        while True:
            paths = await self.queue.get()
            # Coalesce everything already waiting (a burst of PATCHes) into one apply.
            while not self.queue.empty():
                try:
                    paths = paths | self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                await self.apply_job(paths)
            except Exception as e:
                logger.exception(f"reconfigure job failed: {e}")

    async def apply_job(self, paths) -> None:
        """Apply all hot-reloadable actions implied by `paths`, in this (drain) task.
        Contracts (unchanged): a job carries the FACT paths changed, not a value
        snapshot; config is read at apply time (last-writer-wins). Each subsystem is
        rebuilt independently; a failure is logged, while the others still apply
        (apply-what-you-can)."""
        actions = {action_for(p) for p in paths}
        # rebuild_http rebuilds the external client AND everything built off it
        # (cloud backends + the OpenWeatherMap tool source). When it is present in a
        # coalesced job, skip the standalone backend/tools branches so they are not
        # rebuilt twice — the http path covers them off the new client.
        do_http = "rebuild_http" in actions
        if "rebuild_backends" in actions and not do_http:
            await self._rebuild_backends(paths)
        if "rebuild_tools" in actions and not do_http:
            await self._rebuild_tools()
        if "rebuild_audio" in actions:
            await self._rebuild_audio()
        if "rebuild_runs" in actions:
            await self._rebuild_runs()
        if "rebuild_devices" in actions:
            await self._rebuild_devices()
        if "rebuild_reminders" in actions:
            await self._rebuild_reminders()
        if do_http:
            await self._rebuild_http(paths)

    async def _rebuild_backends(self, paths) -> None:
        """Rebuild only the affected stage backends and swap them into the runtime.
        create() may load a model (vosk/piper), so it is offloaded to a thread to
        keep the event loop responsive. A failed rebuild keeps the old backend so
        the running stage is unaffected by the change that did not take.

        Contracts:
        - A queued job carries the FACT that paths changed, NOT a snapshot of values;
          the new config is read from the ConfigService (via the Runtime read-through)
          at apply time. Rapid successive changes therefore converge last-writer-wins.
        - Mutable `Deps` fields (currently only `tts_timeout`) are written ONLY here,
          from the drain task, immediately before the dependent stage is rebuilt;
          readers (`provider.create` in the worker thread) see a consistent scalar.
        - Partial-failure policy: each category is rebuilt independently; a failing
          `create()` keeps that stage's old backend and is logged, while the other
          affected stages are still rebuilt (apply-what-you-can)."""
        await self._rebuild_backend_cats(backend_categories(paths))

    async def _rebuild_backend_cats(self, cats) -> None:
        """Rebuild the given backend categories (subset of {'vad','wakeword','stt','llm','stress','tts'}) and swap
        them into the runtime. Shared by path-driven rebuilds (_rebuild_backends) and the
        http rebuild (which rebuilds all cloud backends off the new client). Same
        contracts as _rebuild_backends: tts_timeout push-before-create, per-category
        isolation, old backend kept on a failed create()."""
        if not cats:
            return
        svc = self.rt.svc
        if "tts" in cats:
            # tts_timeout is baked into the TTS backend; push it into the shared
            # Deps bag BEFORE rebuilding so the new backend picks it up.
            self.deps.tts_timeout = self.rt.core.tts_timeout
        for cat in cats:
            # Mark this category as loading for the duration of the (possibly slow,
            # e.g. RuAccent model download) create(). The set is mutated ONLY from
            # this drain task and read from the heartbeat task at its own tick; there
            # is no await between add() and the to_thread call, and discard() runs in
            # finally, so the panel never observes a half-updated set.
            self._reloading.add(cat)   # mark loading for the (possibly slow) create()
            try:
                backend = await asyncio.to_thread(svc.create, cat)
            except Exception as e:
                logger.error(f"hot-reload of {cat} backend failed: {e}")
                continue
            finally:
                self._reloading.discard(cat)   # clears on success, on the except-continue, and on cancel
            setattr(self.rt, f"{cat}_backend", backend)
            logger.info(
                f"hot-reloaded {cat} backend -> "
                f"{getattr(backend, 'backend_desc', type(backend).__name__)}"
            )

    async def _rebuild_audio(self) -> None:
        """Re-bind the audio server to the configured host/port without a restart."""
        core = self.rt.core
        try:
            await self.rt.audio_server.rebind(core.audio.host, core.audio.port)
            logger.info(f"audio server re-bound to {core.audio.host}:{core.audio.port}")
        except Exception as e:
            # rebind() already tore down the old binding before failing, so the
            # audio server is now DOWN; nothing serves TTS until a restart.
            logger.error(
                f"audio re-bind to {core.audio.host}:{core.audio.port} failed: {e}; "
                f"audio server is now DOWN until a restart"
            )

    async def _rebuild_runs(self) -> None:
        """Apply core.runs changes hot: create/close the SQLite runs store and re-point
        both the pipelines (via runtime) and the panel endpoints at the current store."""
        core = self.rt.core
        try:
            if core.runs.enabled:
                if self.rt.runs_store is None:
                    # Enabling persistence is not retroactive: runs that finalize
                    # during the enable itself (before the store ref is published
                    # via _set_runs_store) are intentionally not recorded.
                    path = os.path.join(config_store.DATA_DIR, "runs.db")
                    store = await asyncio.to_thread(RunsStore, path)
                    await asyncio.to_thread(
                        store.prune, now=time.time(),
                        retention_days=core.runs.retention_days,
                    )
                    self._set_runs_store(store)
                    logger.info("runs store enabled (hot)")
                else:
                    # Already enabled: a retention change just re-prunes.
                    await asyncio.to_thread(
                        self.rt.runs_store.prune, now=time.time(),
                        retention_days=core.runs.retention_days,
                    )
            else:
                if self.rt.runs_store is not None:
                    old = self.rt.runs_store
                    self._set_runs_store(None)
                    await asyncio.to_thread(old.close)
                    logger.info("runs store disabled (hot)")
        except Exception as e:
            logger.error(f"runs reconfigure failed: {e}")

    def _set_runs_store(self, store) -> None:
        """Point both the pipelines (via runtime) and the panel at `store`."""
        self.rt.runs_store = store
        if self.rt.panel is not None:
            self.rt.panel.runs_store = store

    async def _rebuild_devices(self) -> None:
        """Reconcile ESPHome device clients with the current config (hot)."""
        try:
            await self.rt.manager.reconfigure()
            logger.info("device clients reconciled (hot)")
        except Exception as e:
            logger.error(f"device reconfigure failed: {e}")

    async def _rebuild_reminders(self) -> None:
        """Enable/disable the reminders subsystem hot: (un)start the scheduler, (de)register
        its tool source via a tool-set rebuild, and open/close the store. The reminders
        MCP source is gated on rt.scheduler, so rebuilding the tool set adds/removes it."""
        core = self.rt.core
        try:
            if core.reminders.enabled and self.rt.scheduler is None:
                from src.reminders import ReminderScheduler, RemindersStore
                store = await asyncio.to_thread(
                    RemindersStore, os.path.join(config_store.DATA_DIR, "reminders.db")
                )
                scheduler = ReminderScheduler(store)
                scheduler.deliver = self.rt.manager.announce
                self.rt.reminders_store = store
                self.rt.scheduler = scheduler
                await scheduler.start()
                await self._rebuild_tools()   # build_sources now includes the reminders source
                logger.info("reminders subsystem enabled (hot)")
            elif not core.reminders.enabled and self.rt.scheduler is not None:
                scheduler = self.rt.scheduler
                store = self.rt.reminders_store
                await scheduler.stop()
                self.rt.scheduler = None       # build_sources will now omit the reminders source
                await self._rebuild_tools()    # removes the reminders source from the hub
                if store is not None:
                    await asyncio.to_thread(store.close)
                self.rt.reminders_store = None
                logger.info("reminders subsystem disabled (hot)")
            # else: no enable<->disable transition -> nothing to do.
        except Exception as e:
            logger.error(f"reminders reconfigure failed: {e}")

    async def _rebuild_tools(self) -> None:
        """Rebuild the tool source set from current config and hot-swap it into the hub."""
        try:
            sources = build_sources(self.rt.core, self.deps.http_cloud, self.rt.scheduler)
            await self.rt.hub.set_sources(sources)
            logger.info("tool sources rebuilt (hot)")
        except Exception as e:
            logger.error(f"tool sources reconfigure failed: {e}")

    async def _rebuild_http(self, paths) -> None:
        """Rebuild the external (proxied) HTTP client, then rebuild the stages bound to it
        PLUS any stages whose own config changed in this job, then close the old client.
        Rebuilt stages are the UNION of (a) the cloud stages whose selected provider reports
        uses_http_cloud — these MUST rebuild because their create() binds to deps.http_cloud,
        which just changed — and (b) the stages named by `paths` (incl. OFFLINE backends like
        vosk STT / piper TTS), so a coalesced patch touching both core.network.*
        and a selected offline backend does not silently drop that backend's change. The
        OpenWeatherMap tool source also captures http_cloud, so tools are always rebuilt.
        Proxy changes are rare; an in-flight request on the old client may fail when it is
        closed."""
        core = self.rt.core
        old = self.deps.http_cloud
        try:
            new_client = httpx.AsyncClient(proxy=(core.network.external_proxy or None), verify=False)
        except Exception as e:
            logger.error(f"building new HTTP client failed: {e}")
            return
        self.deps.http_cloud = new_client
        # Rebuild the cloud stages (their client changed) UNION the stages whose own config
        # changed in this job (incl. offline backends), so nothing is silently dropped.
        svc = self.rt.svc
        # stress and wakeword are intentionally absent from this tuple: they are
        # offline (uses_http_cloud defaults False), so a proxy change never needs to
        # rebuild them. A coalesced patch that also touches their config is still
        # picked up via backend_categories(paths) below.
        cloud_cats = {c for c in ("vad", "stt", "llm", "tts") if svc.provider(c).uses_http_cloud}
        cats = cloud_cats | backend_categories(paths)
        await self._rebuild_backend_cats(cats)
        await self._rebuild_tools()              # OpenWeatherMap source also uses http_cloud
        # Close the old client after dependents have been rebuilt off the new one.
        try:
            await old.aclose()
        except Exception as e:
            logger.warning(f"closing old HTTP client failed: {e}")
        logger.info("external HTTP client rebuilt (hot)")
