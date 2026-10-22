"""Hot-reload coordinator: classify config changes and apply what can be applied
without a process restart.

`changed_paths` and `action_for` are pure functions (also used by ConfigService
to compute the change set). `Reconfigurator` is wired in app.py: it is registered
via ConfigService.on_change and applies each change live or by rebuilding the part
it belongs to.
"""

import asyncio
import os
import time

import httpx
from loguru import logger

from src import config_store
from src.chains import referenced_instances
from src.logging_setup import setup_logging
from src.plugins.base import get_provider
from src.plugins.llm.base import ReplyConfig
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


# Live LLM provider fields read per request (not baked into the backend): the
# ReplyConfig fields every llm provider inherits (src/plugins/llm/base.py).
_LLM_LIVE_LEAVES = frozenset(ReplyConfig.model_fields)


def action_for(path: str) -> str:
    """Map one changed dotted path to a reconfiguration action.
    Actions: 'live', 'logging', 'rebuild_backends', 'rebuild_http',
    'rebuild_tools', 'rebuild_audio', 'rebuild_devices', 'rebuild_runs',
    'rebuild_reminders', 'rebuild_telemetry'. A path no rule covers is 'unknown':
    nothing applies it, so on_config_change logs it, and
    test_every_config_field_has_an_apply_rule fails on a config field without a rule."""
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
    if path.startswith("core.ack"):
        return "live"          # ack.enabled/sound_path are read per-run via the Runtime read-through
    if path.startswith("core.audio"):            # host/port -> rebind; all -> advert restart
        return "rebuild_audio"
    if path.startswith("core.prompt"):           # system_prompt_path read live
        return "live"
    if path.startswith("core.network"):
        return "rebuild_http"        # external_proxy needs an http-client rebuild (Tier 3)
    if path == "core.tts_timeout":
        return "rebuild_backends"
    if (path.startswith("core.openweathermap") or path.startswith("core.calendar")
            or path.startswith("core.mcp_servers") or path.startswith("core.assistant_tasks")
            or path.startswith("core.memory")):
        return "rebuild_tools"
    if path.startswith("core.agent_mcp"):
        return "live"          # enabled is read per /mcp request through the panel
    if path.startswith("core.reminders"):
        return "rebuild_reminders"
    if path.startswith("core.runs"):
        return "rebuild_runs"
    if path.startswith("core.telemetry"):
        return "rebuild_telemetry"   # the MQTT client is built for one broker: restart it
    if path.startswith("core.devices"):
        return "rebuild_devices"
    if path == "core.device_log_days":
        return "live"          # read at every prune (src/device_logs.py)
    if path == "core.chains":
        return "rebuild_backends"   # builds newly referenced STT/handler backends, drops unreferenced
    if path.startswith("core.commands"):
        return "rebuild_backends"   # rebuilds the handlers in use: simple-nlu builds its model from the catalog
    # stage providers
    if (path.startswith("vad") or path.startswith("wakeword") or path.startswith("stt")
            or path.startswith("tts") or path.startswith("stress")
            or path.startswith("followup")):
        return "rebuild_backends"
    if path.startswith("llm"):
        leaf = path.rsplit(".", 1)[-1]
        return "live" if leaf in _LLM_LIVE_LEAVES else "rebuild_backends"
    return "unknown"


# Actions whose work is performed off the panel request task, in the queue-draining
# task (run_loop -> apply_job), because they may block (model loads, socket re-bind,
# SQLite open/prune, tool-source start network I/O, http-client rebuild, device
# connection updates, reminders store open/close, the MQTT network thread's join).
ASYNC_ACTIONS = {"rebuild_backends", "rebuild_audio", "rebuild_runs",
                 "rebuild_tools", "rebuild_http",
                 "rebuild_devices", "rebuild_reminders", "rebuild_telemetry"}


def backend_categories(paths) -> set[str]:
    """Subset of the single-backend stages {'vad','wakeword','stress','followup'} whose
    backend must be rebuilt for these paths."""
    cats = set()
    for p in paths:
        if action_for(p) != "rebuild_backends":
            continue
        top = p.split(".", 1)[0]
        if top in ("vad", "wakeword", "stress", "followup"):
            cats.add(top)
    return cats


def pool_instances(paths) -> set[tuple[str, str]]:
    """(category, provider id) of the STT/handler/TTS instances whose own settings
    changed in a way that needs a backend rebuild (`stt|llm|tts.instances.<id>.<field>`)."""
    keys = set()
    for p in paths:
        parts = p.split(".")
        if (len(parts) > 3 and parts[0] in ("stt", "llm", "tts") and parts[1] == "instances"
                and action_for(p) == "rebuild_backends"):
            keys.add((parts[0], parts[2]))
    return keys


def reconciled_pools(paths) -> set[str]:
    """Pool categories whose set of backends in use may have changed: stt/llm on a
    core.chains change, tts on a tts.variants change."""
    cats = set()
    if "core.chains" in paths:
        cats |= {"stt", "llm"}
    if "tts.variants" in paths:
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
        """Backend categories whose rebuild (model load) is in flight right now, plus
        "tools" while the tool sources are being rebuilt. Read by the panel's system
        snapshot to surface a 'loading' state in the UI. Mutated only from the drain
        task; sorted for a stable payload."""
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

    def request_initial_tools_load(self) -> None:
        """Queue a one-shot background build of the tool sources so a slow external MCP
        connect does NOT block boot. Drained by run_loop in the main task: _rebuild_tools
        marks reloading() ("tools") while the sources connect and swaps them into the hub
        when ready; until then the hub serves no tools. Routed through the SAME queue as
        live config changes so it is serialized and coalesced with a tool-config change
        that arrives during boot."""
        # Synthetic "core.mcp_servers.__warmup__" path: action_for() maps it to
        # rebuild_tools; the "__warmup__" leaf marks it as a boot warm-up rather than a
        # real field change.
        self.queue.put_nowait({"core.mcp_servers.__warmup__"})

    def on_config_change(self, paths) -> None:
        """ConfigService.on_change callback (SYNC, runs on the panel request task).
        Live fields are already effective through the Runtime read-through; here we
        (a) reinit logging if the level changed and (b) hand heavy/blocking rebuilds
        to the main-task queue.
        Must stay fast: no blocking work and no model loads happen here."""
        by_path = {p: action_for(p) for p in paths}
        for path, action in by_path.items():
            if action == "unknown":
                logger.warning(f"config path {path} has no apply rule in reconfig.action_for: "
                               f"not applied live (a key the config model does not define "
                               f"is ignored entirely)")
        actions = set(by_path.values())
        if "logging" in actions:
            setup_logging(self.rt.core.log_level)
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
        if "rebuild_telemetry" in actions:
            await self._rebuild_telemetry()
        if do_http:
            await self._rebuild_http(paths)
        # A job may have built a new handler backend or changed the tool catalog.
        await self.warm_handlers()

    async def warm_handlers(self) -> None:
        """Prepare every handler backend that has warm_up() for the current tool
        catalog (simple-nlu trains its model on it), so the next request does not pay
        for it. Run after every job, the boot one that connects the tool sources
        included."""
        for backend in list(self.rt.llm_backends.values()):
            warm_up = getattr(backend, "warm_up", None)
            if warm_up is not None:
                await warm_up(self.rt.hub.tools or [])

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
        await self._rebuild_pool(self._pool_keys(paths), reconcile=reconciled_pools(paths))

    def _in_use(self) -> set[tuple[str, str]]:
        """(category, provider id) of every pool backend the runtime keeps built: the
        STT and handler of each enabled chain and each enabled TTS variant."""
        return referenced_instances(self.rt.core.chains) | {
            ("tts", pid) for pid in self.rt.svc.tts_variants()}

    def _pool_keys(self, paths) -> set[tuple[str, str]]:
        """Pool instances to rebuild for `paths`: those whose own settings changed,
        plus every enabled TTS variant on a core.tts_timeout change (the timeout is
        baked into each TTS backend) and every handler an enabled chain uses on a
        core.commands change (a handler may build its model from the catalog when it
        is created)."""
        keys = pool_instances(paths)
        if "core.tts_timeout" in paths:
            keys |= {("tts", pid) for pid in self.rt.svc.tts_variants()}
        if any(p.startswith("core.commands") for p in paths):
            keys |= {(cat, pid) for cat, pid in self._in_use() if cat == "llm"}
        return keys

    async def _rebuild_backend_cats(self, cats) -> None:
        """Rebuild the given backend categories (subset of {'vad','wakeword','stress',
        'followup'}) and swap them into the runtime. Shared by path-driven rebuilds
        (_rebuild_backends) and the http rebuild (which rebuilds all cloud backends off
        the new client). Same
        contracts as _rebuild_backends: per-category isolation, old backend kept on a
        failed create()."""
        if not cats:
            return
        svc = self.rt.svc
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

    async def _rebuild_pool(self, keys, *, reconcile=frozenset()) -> None:
        """Rebuild the pool backends named by `keys` ((category, provider id)) that are
        in use — the STT/handler of an enabled chain, or an enabled TTS variant — and
        swap each into the runtime's `<cat>_backends` dict; an instance not in use is
        not built. For each category in `reconcile` (see reconciled_pools) it also
        builds every newly used instance and drops the ones no longer used.

        Same contracts as _rebuild_backend_cats: config read at apply time, each
        instance rebuilt independently, the old backend kept on a failed create(),
        and reloading() reports the category ("stt"/"llm"/"tts") while it loads.
        tts_timeout is baked into a TTS backend, so it is pushed into the shared Deps
        bag BEFORE any create() of this job when a TTS backend is among them."""
        wanted = self._in_use()
        keys = set(keys) & wanted
        for cat in reconcile:
            pool = getattr(self.rt, f"{cat}_backends")
            why = "not an enabled variant" if cat == "tts" else "no enabled chain uses it"
            for pid in [p for p in pool if (cat, p) not in wanted]:
                del pool[pid]
                logger.info(f"dropped {cat} backend {pid!r}: {why}")
            keys |= {(c, p) for c, p in wanted if c == cat and p not in pool}
        if any(c == "tts" for c, _ in keys):
            self.deps.tts_timeout = self.rt.core.tts_timeout
        svc = self.rt.svc
        for cat, pid in sorted(keys):
            self._reloading.add(cat)
            try:
                backend = await asyncio.to_thread(svc.create_instance, cat, pid)
            except Exception as e:
                logger.error(f"hot-reload of {cat} backend {pid!r} failed: {e}")
                continue
            finally:
                self._reloading.discard(cat)
            getattr(self.rt, f"{cat}_backends")[pid] = backend
            logger.info(
                f"hot-reloaded {cat} backend {pid!r} -> "
                f"{getattr(backend, 'backend_desc', type(backend).__name__)}"
            )

    async def _rebuild_audio(self) -> None:
        """Re-bind the device server to the configured host/port without a restart and
        re-announce it by mDNS at the new address."""
        core = self.rt.core
        try:
            await self.rt.manager.rebind(core.audio.host, core.audio.port)
            logger.info(f"device server re-bound to {core.audio.host}:{core.audio.port}")
        except Exception as e:
            # rebind() already tore down the old binding before failing, so the
            # device server is now DOWN; no device can connect until a restart.
            logger.error(
                f"device server re-bind to {core.audio.host}:{core.audio.port} failed: {e}; "
                f"device server is now DOWN until a restart"
            )
            return
        try:
            await self.rt.advert.stop()
            await self.rt.advert.start(core.audio.advertise_host or core.audio.host, self.rt.manager.port)
        except Exception as e:
            logger.error(
                f"device server re-bound, but its mDNS advert failed: {e}; "
                f"devices cannot find it until a restart"
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
        """Apply the device config to the live device connections (hot)."""
        try:
            await self.rt.manager.reconfigure()
            logger.info("device connections reconciled (hot)")
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
                    RemindersStore, os.path.join(config_store.DATA_DIR, "schedule.db")
                )
                scheduler = ReminderScheduler(store)
                scheduler.deliver = self.rt.manager.fire
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

    async def _rebuild_telemetry(self) -> None:
        """Restart the telemetry publisher on the current core.telemetry (hot): the old
        broker connection is closed, a new one opens when telemetry is on with a broker."""
        try:
            await self.rt.telemetry.stop()
            await self.rt.telemetry.start()
            logger.info("telemetry restarted (hot)")
        except Exception as e:
            logger.error(f"telemetry reconfigure failed: {e}")

    async def _rebuild_tools(self) -> None:
        """Rebuild the tool source set from current config and hot-swap it into the hub.
        reloading() reports "tools" while the (possibly slow) external MCP connects run."""
        self._reloading.add("tools")
        try:
            sources = build_sources(self.rt.core, self.deps.http_cloud, self.rt.scheduler,
                                    self.rt.assistant_tasks, lambda: self.rt.manager, self.rt.memory)
            await self.rt.hub.set_sources(sources)
            logger.info("tool sources rebuilt (hot)")
        except Exception as e:
            logger.error(f"tool sources reconfigure failed: {e}")
        finally:
            self._reloading.discard("tools")   # clears on success, on failure and on cancel

    async def _rebuild_http(self, paths) -> None:
        """Rebuild the external (proxied) HTTP client, then rebuild the stages bound to it
        PLUS any stages whose own config changed in this job, then close the old client.
        Rebuilt stages are the UNION of (a) the cloud backends — the single-backend stages
        whose selected provider reports uses_http_cloud, and the STT/handler instances an
        enabled chain references and the enabled TTS variants whose provider does — these
        MUST rebuild because their create() binds to deps.http_cloud, which just changed —
        and (b) the backends named by `paths` (incl. OFFLINE ones like vosk STT / piper
        TTS, and a core.chains / tts.variants change), so a coalesced patch touching both
        core.network.* and an offline backend does not silently drop that backend's
        change. The OpenWeatherMap tool source also captures http_cloud, so tools are
        always rebuilt. Proxy changes are rare; an in-flight request on the old client may
        fail when it is closed."""
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
        cloud_cats = {c for c in ("vad", "followup") if svc.provider(c).uses_http_cloud}
        cats = cloud_cats | backend_categories(paths)
        await self._rebuild_backend_cats(cats)
        cloud_pool = {(c, p) for c, p in self._in_use() if get_provider(c, p).uses_http_cloud}
        await self._rebuild_pool(cloud_pool | self._pool_keys(paths),
                                 reconcile=reconciled_pools(paths))
        await self._rebuild_tools()              # OpenWeatherMap source also uses http_cloud
        # Close the old client after dependents have been rebuilt off the new one.
        try:
            await old.aclose()
        except Exception as e:
            logger.warning(f"closing old HTTP client failed: {e}")
        logger.info("external HTTP client rebuilt (hot)")
