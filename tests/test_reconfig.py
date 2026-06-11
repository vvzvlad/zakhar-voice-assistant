"""Unit tests for the hot-reload classification helpers and the Reconfigurator."""

import asyncio
import contextlib
import os
import types

import pytest

from src.reconfig import (
    Reconfigurator,
    action_for,
    backend_categories,
    changed_paths,
)


# --- changed_paths -----------------------------------------------------------

def test_changed_paths_nested_diff():
    old = {"core": {"context": {"max_turns": 5, "ttl_seconds": 300}}}
    new = {"core": {"context": {"max_turns": 9, "ttl_seconds": 300}}}
    assert changed_paths(old, new) == {"core.context.max_turns"}


def test_changed_paths_unchanged_returns_empty():
    doc = {"core": {"vad": {"aggressiveness": 2, "silence_ms": 800}}, "llm": {"x": 1}}
    assert changed_paths(doc, doc) == set()


def test_changed_paths_list_is_single_leaf():
    # Lists are compared by equality and recorded as one dotted path, not per-item.
    old = {"core": {"mcp_servers": [{"name": "a"}]}}
    new = {"core": {"mcp_servers": [{"name": "a"}, {"name": "b"}]}}
    assert changed_paths(old, new) == {"core.mcp_servers"}


def test_changed_paths_added_and_removed_keys():
    old = {"a": 1, "b": 2}
    new = {"a": 1, "c": 3}
    assert changed_paths(old, new) == {"b", "c"}


def test_changed_paths_dict_to_scalar_yields_specific_leaf():
    # The dict side is descended even though the other side is a scalar.
    assert changed_paths({"a": {"b": 1}}, {"a": 5}) == {"a.b"}


def test_changed_paths_scalar_to_dict_yields_specific_leaf():
    assert changed_paths({"a": 5}, {"a": {"b": 1}}) == {"a.b"}


def test_changed_paths_added_subtree_yields_specific_leaves():
    # A wholly new subtree reports its leaf paths, not the parent.
    old = {"core": {"audio": {"ttl": 300}}}
    new = {"core": {"audio": {"ttl": 300}, "vad": {"aggressiveness": 2}}}
    assert changed_paths(old, new) == {"core.vad.aggressiveness"}


def test_changed_paths_removed_subtree_yields_specific_leaves():
    old = {"core": {"audio": {"ttl": 300}, "vad": {"aggressiveness": 2}}}
    new = {"core": {"audio": {"ttl": 300}}}
    assert changed_paths(old, new) == {"core.vad.aggressiveness"}


# --- action_for --------------------------------------------------------------

def test_action_for_representative_paths():
    cases = {
        "core.log_level": "logging",
        "core.context.max_turns": "live",
        "core.capture.enabled": "live",      # read per-run via the Runtime read-through
        "core.capture.dir": "live",          # ditto; read per-run via the Runtime read-through
        "core.ack.enabled": "live",          # read per-run via the Runtime read-through
        "core.ack.sound_path": "live",       # ditto; read per-run via the Runtime read-through
        "core.vad.aggressiveness": "live",
        "core.audio.public_base_url": "live",
        "core.audio.ttl": "live",
        "core.audio.host": "rebuild_audio",
        "core.audio.port": "rebuild_audio",
        "core.prompt.system_prompt_path": "live",
        "core.network.external_proxy": "rebuild_http",
        "core.tts_timeout": "rebuild_backends",
        "core.openweathermap.api_key": "rebuild_tools",
        "core.calendar.url": "rebuild_tools",
        "core.mcp_servers": "rebuild_tools",
        "core.reminders.enabled": "rebuild_reminders",
        "core.reminders.something": "rebuild_reminders",
        "core.runs.enabled": "rebuild_runs",
        "core.runs.retention_days": "rebuild_runs",
        "core.devices": "rebuild_devices",
        "core.devices.0.host": "rebuild_devices",
        "core.esphome.port": "rebuild_devices",
        "llm.instances.openrouter.temperature": "rebuild_backends",
        "llm.instances.openrouter.reply_empty": "live",
        "llm.instances.openrouter.max_tool_rounds": "live",
        "tts.instances.yandex.voice": "rebuild_backends",
        "stt.instances.groq.api_key": "rebuild_backends",
        "vad.instances.webrtc.aggressiveness": "rebuild_backends",
        "vad.instances.webrtc.auto_gain": "rebuild_backends",
        "vad.instances.silero.threshold": "rebuild_backends",
        "vad.selected": "rebuild_backends",
        "something.unknown.path": "restart",
    }
    for path, expected in cases.items():
        assert action_for(path) == expected, path


def test_action_for_llm_reply_error_is_live():
    # reply_error is one of the live LLM provider leaves (read per request, not baked
    # into the backend), so changing it must NOT trigger a backend rebuild.
    assert action_for("llm.instances.openrouter.reply_error") == "live"


# --- backend_categories ------------------------------------------------------

def test_backend_categories_maps_each_stage_path():
    assert backend_categories({"tts.instances.yandex.voice"}) == {"tts"}
    assert backend_categories({"llm.selected"}) == {"llm"}
    assert backend_categories({"core.tts_timeout"}) == {"tts"}
    assert backend_categories({"vad.instances.webrtc.aggressiveness"}) == {"vad"}
    assert backend_categories(
        {"stt.instances.groq.api_key", "tts.selected"}
    ) == {"stt", "tts"}


def test_backend_categories_ignores_non_rebuild_paths():
    # A live path contributes nothing — only rebuild_backends paths count.
    assert backend_categories({"core.vad.aggressiveness"}) == set()


# --- Reconfigurator ----------------------------------------------------------

def _stub_runtime(log_level="INFO", audio_ttl=300):
    """A minimal runtime exposing `.core.log_level`, `.core.audio.ttl` and a live
    `.audio_server` whose `ttl` is writable (mirrors the real AudioServer cache)."""
    core = types.SimpleNamespace(
        log_level=log_level,
        audio=types.SimpleNamespace(ttl=audio_ttl),
    )
    audio_server = types.SimpleNamespace(ttl=audio_ttl)
    return types.SimpleNamespace(core=core, audio_server=audio_server)


def _make_reconf(rt, *, deps=None):
    """Build a Reconfigurator with a fresh queue and (by default) a stub deps bag."""
    if deps is None:
        deps = types.SimpleNamespace(tts_timeout=30)
    return Reconfigurator(rt, deps, asyncio.Queue())


def test_reconfigurator_live_only_no_job():
    reconf = _make_reconf(_stub_runtime())
    reconf.on_config_change({"core.context.max_turns", "core.vad.aggressiveness"})
    assert reconf.queue.qsize() == 0


def test_reconfigurator_capture_is_live_no_job():
    # core.capture.* is read per-run via the Runtime read-through, so it applies live:
    # it must NOT enqueue an async rebuild job.
    reconf = _make_reconf(_stub_runtime())
    reconf.on_config_change({"core.capture.enabled"})
    assert reconf.queue.qsize() == 0


def test_reconfigurator_ack_is_live_no_job():
    # core.ack.* (enabled / sound_path) is read per-run via the Runtime read-through,
    # so it applies live: it must NOT enqueue an async rebuild job.
    reconf = _make_reconf(_stub_runtime())
    reconf.on_config_change({"core.ack.enabled", "core.ack.sound_path"})
    assert reconf.queue.qsize() == 0


@pytest.mark.parametrize(
    "paths, expected_action",
    [
        # rebuild_backends (mixed with a live path that needs no async work).
        ({"core.context.max_turns", "tts.instances.yandex.voice"}, "rebuild_backends"),
        # core.network.external_proxy -> rebuild_http (Tier 3b).
        ({"core.network.external_proxy"}, "rebuild_http"),
        # core.openweathermap/calendar/mcp_servers -> rebuild_tools (Tier 3b).
        ({"core.openweathermap.api_key"}, "rebuild_tools"),
        # core.audio.host -> rebuild_audio.
        ({"core.audio.host"}, "rebuild_audio"),
        # core.runs.enabled -> rebuild_runs.
        ({"core.runs.enabled"}, "rebuild_runs"),
        # core.devices/core.esphome.port -> rebuild_devices (Tier 3c).
        ({"core.esphome.port"}, "rebuild_devices"),
        # core.reminders.* -> rebuild_reminders (Tier 3c).
        ({"core.reminders.enabled"}, "rebuild_reminders"),
    ],
)
def test_reconfigurator_async_action_enqueues(paths, expected_action):
    # Every async rebuild action is applied hot: on_config_change enqueues exactly one
    # job for the drain task (carrying the changed paths). expected_action is the async
    # action these paths trigger.
    assert expected_action in {action_for(p) for p in paths}
    reconf = _make_reconf(_stub_runtime())
    reconf.on_config_change(paths)
    assert reconf.queue.qsize() == 1
    assert reconf.queue.get_nowait() == paths


def test_reconfigurator_audio_ttl_applied_live():
    # core.audio.ttl is live: the Reconfigurator pushes the new value onto the
    # running server (which caches ttl).
    rt = _stub_runtime(audio_ttl=300)
    rt.core.audio.ttl = 900            # simulate the already-applied document
    reconf = _make_reconf(rt)
    reconf.on_config_change({"core.audio.ttl"})
    assert rt.audio_server.ttl == 900


def test_reconfigurator_log_level_reinits_logging(monkeypatch):
    calls = []
    monkeypatch.setattr("src.reconfig.setup_logging", lambda level: calls.append(level))
    reconf = _make_reconf(_stub_runtime(log_level="DEBUG"))
    reconf.on_config_change({"core.log_level"})
    # Logging reinit invoked with the live level.
    assert calls == ["DEBUG"]


# --- apply_job (async backend rebuild) ---------------------------------------

class _StubSvc:
    """create(cat) returns a sentinel encoding the category AND the tts_timeout it saw.

    Reading `deps.tts_timeout` here lets tests assert the push-before-create ordering:
    the rebuilt backend records exactly the timeout the worker observed at create time.
    The stub shares the SAME `deps` object the Reconfigurator holds (set via attach).
    Also tracks per-category call counts to assert coalescing (rebuild once per burst).

    provider(cat) returns an object whose uses_http_cloud bool mirrors the selected
    provider's flag. By default every category is cloud (uses_http_cloud True); pass
    `cloud_cats` to mark only a subset cloud (the rest report offline -> not rebuilt
    on a proxy change). This drives _rebuild_http's proxied-only category selection.
    """

    def __init__(self, fail_on=None, cloud_cats=("stt", "llm", "tts")):
        self._fail_on = fail_on   # category whose create() should raise
        self.deps = None          # set by attach() to the Reconfigurator's deps bag
        self.calls = {}           # per-category create() call count
        self._cloud_cats = set(cloud_cats)   # categories whose provider is cloud

    def attach(self, deps):
        """Point the stub at the same deps object the Reconfigurator mutates."""
        self.deps = deps
        return self

    def create(self, cat):
        self.calls[cat] = self.calls.get(cat, 0) + 1
        if cat == self._fail_on:
            raise RuntimeError(f"boom:{cat}")
        timeout = self.deps.tts_timeout if self.deps is not None else None
        return ("backend", cat, timeout)

    def provider(self, cat):
        """Selected-provider stub exposing only uses_http_cloud (what _rebuild_http reads)."""
        return types.SimpleNamespace(uses_http_cloud=cat in self._cloud_cats)


def _job_runtime(svc, *, tts_timeout=99):
    """Runtime stub for apply_job: holds the four stage backends, the svc and a core
    whose tts_timeout the TTS rebuild copies into deps."""
    return types.SimpleNamespace(
        svc=svc,
        core=types.SimpleNamespace(tts_timeout=tts_timeout),
        vad_backend="old-vad",
        stt_backend="old-stt",
        llm_backend="old-llm",
        tts_backend="old-tts",
    )


def _make_job_reconf(svc, *, tts_timeout=99, deps_timeout=30):
    """Wire a Reconfigurator + runtime + deps for apply_job/run_loop tests, sharing the
    deps object with the stub svc so create() observes the timeout the loop pushed."""
    deps = types.SimpleNamespace(tts_timeout=deps_timeout)
    svc.attach(deps)
    rt = _job_runtime(svc, tts_timeout=tts_timeout)
    return Reconfigurator(rt, deps, asyncio.Queue()), rt, deps


@pytest.mark.asyncio
async def test_apply_job_rebuilds_only_tts_and_pushes_timeout():
    # create() reads deps.tts_timeout; asserting the sentinel encodes the NEW timeout
    # proves the push happened BEFORE create (push-before-create ordering).
    reconf, rt, deps = _make_job_reconf(_StubSvc(), tts_timeout=99, deps_timeout=30)
    await reconf.apply_job({"tts.instances.yandex.voice"})
    assert rt.tts_backend == ("backend", "tts", 99)   # create saw the pushed timeout
    assert rt.stt_backend == "old-stt"     # untouched
    assert rt.llm_backend == "old-llm"     # untouched
    assert deps.tts_timeout == 99          # pushed from core.tts_timeout before rebuild


@pytest.mark.asyncio
async def test_apply_job_rebuilds_only_vad():
    # A vad instance change rebuilds ONLY the vad backend; stt/llm/tts stay untouched.
    reconf, rt, deps = _make_job_reconf(_StubSvc(), deps_timeout=30)
    await reconf.apply_job({"vad.instances.webrtc.aggressiveness"})
    assert rt.vad_backend == ("backend", "vad", 30)   # swapped to the new sentinel
    assert rt.stt_backend == "old-stt"     # untouched
    assert rt.llm_backend == "old-llm"     # untouched
    assert rt.tts_backend == "old-tts"     # untouched
    assert deps.tts_timeout == 30          # no tts rebuild -> timeout not touched


@pytest.mark.asyncio
async def test_apply_job_rebuilds_only_stt():
    reconf, rt, deps = _make_job_reconf(_StubSvc(), deps_timeout=30)
    await reconf.apply_job({"stt.selected"})
    assert rt.stt_backend == ("backend", "stt", 30)
    assert rt.llm_backend == "old-llm"     # untouched
    assert rt.tts_backend == "old-tts"     # untouched
    assert deps.tts_timeout == 30          # no tts rebuild -> timeout not touched


@pytest.mark.asyncio
async def test_apply_job_rebuilds_multiple_categories():
    # A multi-category job rebuilds BOTH stt and tts (new sentinels); llm untouched.
    # The tts push runs once, up front, so any create() in this job (incl. stt) sees
    # the already-updated deps.tts_timeout (99).
    reconf, rt, _deps = _make_job_reconf(_StubSvc(), tts_timeout=99, deps_timeout=30)
    await reconf.apply_job({"stt.selected", "tts.selected"})
    assert rt.stt_backend == ("backend", "stt", 99)   # swapped to the new sentinel
    assert rt.tts_backend == ("backend", "tts", 99)   # tts rebuilt with pushed timeout
    assert rt.llm_backend == "old-llm"     # untouched


@pytest.mark.asyncio
async def test_apply_job_failed_create_keeps_old_backend():
    reconf, rt, _deps = _make_job_reconf(_StubSvc(fail_on="llm"))
    await reconf.apply_job({"llm.selected"})
    assert rt.llm_backend == "old-llm"     # rebuild failed -> old backend kept


@pytest.mark.asyncio
async def test_apply_job_partial_failure_rebuilds_what_it_can():
    # One category fails, the other succeeds in the same job: apply-what-you-can.
    reconf, rt, _deps = _make_job_reconf(_StubSvc(fail_on="llm"), deps_timeout=30)
    await reconf.apply_job({"stt.selected", "llm.selected"})
    assert rt.stt_backend == ("backend", "stt", 30)   # rebuilt despite the sibling failure
    assert rt.llm_backend == "old-llm"     # failed -> old backend kept


# --- run_loop (queue drain task) ---------------------------------------------

async def _wait_for(predicate, *, max_iters=50):
    """Yield to the loop (bounded) until predicate() is truthy; return whether it became
    true. Deterministic and bounded -- no unbounded waits."""
    for _ in range(max_iters):
        if predicate():
            return True
        await asyncio.sleep(0)
    return predicate()


@pytest.mark.asyncio
async def test_run_loop_applies_a_job():
    reconf, rt, _deps = _make_job_reconf(_StubSvc(), tts_timeout=99, deps_timeout=30)
    task = asyncio.create_task(reconf.run_loop())
    try:
        reconf.queue.put_nowait({"tts.selected"})
        assert await _wait_for(lambda: rt.tts_backend != "old-tts")
        assert rt.tts_backend == ("backend", "tts", 99)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_run_loop_survives_apply_job_exception(monkeypatch):
    # First apply raises, second succeeds: the loop must keep draining after the error.
    reconf, rt, _deps = _make_job_reconf(_StubSvc(), tts_timeout=99, deps_timeout=30)
    real_apply = reconf.apply_job
    state = {"first": True}

    async def flaky_apply(paths):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("boom in apply_job")
        await real_apply(paths)

    monkeypatch.setattr(reconf, "apply_job", flaky_apply)
    task = asyncio.create_task(reconf.run_loop())
    try:
        reconf.queue.put_nowait({"stt.selected"})        # makes apply_job raise once
        # Wait until the first (failing) job was consumed, so the good job is a SEPARATE
        # apply call (not coalesced into the failing one).
        assert await _wait_for(lambda: not state["first"])
        reconf.queue.put_nowait({"tts.selected"})        # good job processed afterwards
        assert await _wait_for(lambda: rt.tts_backend != "old-tts")
        assert rt.tts_backend == ("backend", "tts", 99)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_run_loop_coalesces_bursted_jobs():
    # Two job sets enqueued BEFORE the loop drains are coalesced into one apply, so each
    # stage rebuilds at most once (assert per-category create() call counts).
    svc = _StubSvc()
    reconf, rt, _deps = _make_job_reconf(svc, tts_timeout=99, deps_timeout=30)
    reconf.queue.put_nowait({"stt.selected"})
    reconf.queue.put_nowait({"tts.selected"})
    task = asyncio.create_task(reconf.run_loop())
    try:
        assert await _wait_for(
            lambda: rt.stt_backend != "old-stt" and rt.tts_backend != "old-tts"
        )
        assert rt.stt_backend == ("backend", "stt", 99)   # tts push ran first this job
        assert rt.tts_backend == ("backend", "tts", 99)
        assert svc.calls.get("stt") == 1   # coalesced -> rebuilt once
        assert svc.calls.get("tts") == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# --- apply_job (Tier 3a: audio re-bind) --------------------------------------

class _StubAudioServer:
    """Records rebind(host, port) calls; can be told to raise to exercise the
    failure path that is logged and continues."""

    def __init__(self, *, fail=False):
        self._fail = fail
        self.rebinds = []        # list of (host, port) the reconfigurator asked for

    async def rebind(self, host, port):
        self.rebinds.append((host, port))
        if self._fail:
            raise RuntimeError("bind: address already in use")


def _audio_runtime(audio_server, *, host="0.0.0.0", port=8081):
    return types.SimpleNamespace(
        audio_server=audio_server,
        core=types.SimpleNamespace(audio=types.SimpleNamespace(host=host, port=port)),
    )


@pytest.mark.asyncio
async def test_apply_job_audio_rebinds_to_configured_host_port():
    audio = _StubAudioServer()
    rt = _audio_runtime(audio, host="127.0.0.1", port=9000)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    await reconf.apply_job({"core.audio.host"})
    assert audio.rebinds == [("127.0.0.1", 9000)]


@pytest.mark.asyncio
async def test_apply_job_audio_rebind_failure_is_attempted():
    audio = _StubAudioServer(fail=True)
    rt = _audio_runtime(audio, host="127.0.0.1", port=9000)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    await reconf.apply_job({"core.audio.port"})
    assert audio.rebinds == [("127.0.0.1", 9000)]   # attempted


# --- apply_job (Tier 3a: runs store toggle / retention) ----------------------

class _FakeRunsStore:
    """Stand-in for RunsStore: records prune calls and whether it was closed.
    Constructed with a path (mirrors the real signature) so the monkeypatched
    factory can be called positionally from a worker thread."""

    instances = []   # every fake created, in order (class-level for test assertions)

    def __init__(self, path):
        self.path = path
        self.prunes = []     # list of (now, retention_days)
        self.closed = False
        _FakeRunsStore.instances.append(self)

    def prune(self, *, now, retention_days):
        self.prunes.append((now, retention_days))

    def close(self):
        self.closed = True


def _runs_runtime(*, enabled, retention_days=7, runs_store=None, panel=None):
    # The runs.db path now derives from the hardcoded config_store.DATA_DIR (not from
    # core.context.dir), so the runtime stub no longer needs a context dir.
    return types.SimpleNamespace(
        core=types.SimpleNamespace(
            runs=types.SimpleNamespace(enabled=enabled, retention_days=retention_days),
        ),
        runs_store=runs_store,
        panel=panel,
    )


@pytest.fixture(autouse=True)
def _reset_fake_runs_instances():
    _FakeRunsStore.instances.clear()
    yield
    _FakeRunsStore.instances.clear()


@pytest.mark.asyncio
async def test_apply_job_runs_enable_creates_store_and_reaches_panel(monkeypatch):
    # Enabling while runs_store is None: a store is created, pruned with the configured
    # retention, and BOTH the runtime and the panel are re-pointed at it.
    monkeypatch.setattr("src.reconfig.RunsStore", _FakeRunsStore)
    # runs.db lives under the hardcoded DATA_DIR; pin it so we can assert the path.
    monkeypatch.setattr("src.config_store.DATA_DIR", "/tmp/zakhar-runs-test")
    panel = types.SimpleNamespace(runs_store=None)
    rt = _runs_runtime(enabled=True, retention_days=14, runs_store=None, panel=panel)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())

    await reconf.apply_job({"core.runs.enabled"})

    assert len(_FakeRunsStore.instances) == 1
    store = _FakeRunsStore.instances[0]
    assert store.path == os.path.join("/tmp/zakhar-runs-test", "runs.db")
    assert len(store.prunes) == 1 and store.prunes[0][1] == 14   # retention pushed
    assert rt.runs_store is store          # pipelines see the new store
    assert panel.runs_store is store       # panel endpoints see the new store too


@pytest.mark.asyncio
async def test_apply_job_runs_disable_closes_store_and_clears_panel(monkeypatch):
    # Disabling while a store exists: the old store is closed and BOTH refs cleared.
    monkeypatch.setattr("src.reconfig.RunsStore", _FakeRunsStore)
    existing = _FakeRunsStore("/tmp/zakhar-runs-test/runs.db")
    panel = types.SimpleNamespace(runs_store=existing)
    rt = _runs_runtime(enabled=False, runs_store=existing, panel=panel)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())

    await reconf.apply_job({"core.runs.enabled"})

    assert existing.closed is True
    assert rt.runs_store is None
    assert panel.runs_store is None


@pytest.mark.asyncio
async def test_apply_job_runs_retention_change_reprunes_existing(monkeypatch):
    # Already enabled + a retention change: the existing store is re-pruned, no new
    # store is created, no close.
    monkeypatch.setattr("src.reconfig.RunsStore", _FakeRunsStore)
    existing = _FakeRunsStore("/tmp/zakhar-runs-test/runs.db")
    _FakeRunsStore.instances.clear()    # ignore the one built directly above
    panel = types.SimpleNamespace(runs_store=existing)
    rt = _runs_runtime(enabled=True, retention_days=3, runs_store=existing, panel=panel)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())

    await reconf.apply_job({"core.runs.retention_days"})

    assert _FakeRunsStore.instances == []        # no new store created
    assert existing.closed is False
    assert existing.prunes and existing.prunes[-1][1] == 3
    assert rt.runs_store is existing             # unchanged


@pytest.mark.asyncio
async def test_apply_job_runs_enable_without_panel_does_not_crash(monkeypatch):
    # rt.panel is None: the runtime store is still set; no AttributeError on the panel.
    monkeypatch.setattr("src.reconfig.RunsStore", _FakeRunsStore)
    rt = _runs_runtime(enabled=True, retention_days=7, runs_store=None, panel=None)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())

    await reconf.apply_job({"core.runs.enabled"})

    assert rt.runs_store is _FakeRunsStore.instances[0]


# --- apply_job (Tier 3b: tool sources hot-swap) ------------------------------

class _FakeHub:
    """Records set_sources(sources) calls; can be told to raise to exercise the
    failure path that is logged and continues."""

    def __init__(self, *, fail=False):
        self._fail = fail
        self.set_calls = []      # list of source lists passed to set_sources

    async def set_sources(self, sources):
        self.set_calls.append(sources)
        if self._fail:
            raise RuntimeError("set_sources boom")


@pytest.mark.asyncio
async def test_apply_job_rebuild_tools_swaps_into_hub(monkeypatch):
    # build_sources is monkeypatched to a sentinel; assert hub.set_sources gets exactly
    # that list, built from rt.core/deps.http_cloud/rt.scheduler.
    sentinel = ["src-a", "src-b"]
    seen = {}

    def fake_build(core, http_cloud, scheduler):
        seen["args"] = (core, http_cloud, scheduler)
        return sentinel

    monkeypatch.setattr("src.reconfig.build_sources", fake_build)
    hub = _FakeHub()
    core = types.SimpleNamespace()
    scheduler = object()
    rt = types.SimpleNamespace(hub=hub, core=core, scheduler=scheduler)
    deps = types.SimpleNamespace(http_cloud="cloud-client", tts_timeout=30)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.openweathermap.api_key"})

    assert hub.set_calls == [sentinel]
    assert seen["args"] == (core, "cloud-client", scheduler)


@pytest.mark.asyncio
async def test_apply_job_rebuild_tools_failure_logs_and_continues(monkeypatch):
    # A failing set_sources() is logged and the job continues (apply-what-you-can).
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["x"])
    hub = _FakeHub(fail=True)
    rt = types.SimpleNamespace(hub=hub, core=types.SimpleNamespace(), scheduler=None)
    deps = types.SimpleNamespace(http_cloud="cloud-client", tts_timeout=30)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.mcp_servers"})

    assert hub.set_calls == [["x"]]


# --- apply_job (Tier 3b: external HTTP client rebuild) -----------------------

class _FakeHttpClient:
    """Stand-in for httpx.AsyncClient: records whether aclose() was called."""

    def __init__(self, tag):
        self.tag = tag
        self.closed = False

    async def aclose(self):
        self.closed = True


def _http_runtime(svc, hub, *, external_proxy=""):
    """Runtime stub for the http rebuild: holds svc (for backend cats), the hub (for the
    tools rebuild), a scheduler and a core exposing tts_timeout + network.external_proxy."""
    return types.SimpleNamespace(
        svc=svc,
        hub=hub,
        scheduler=None,
        core=types.SimpleNamespace(
            tts_timeout=42,
            network=types.SimpleNamespace(external_proxy=external_proxy),
        ),
        vad_backend="old-vad",
        stt_backend="old-stt",
        llm_backend="old-llm",
        tts_backend="old-tts",
    )


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_swaps_client_rebuilds_deps_and_closes_old(monkeypatch):
    # rebuild_http: builds a new client, swaps deps.http_cloud, rebuilds all cloud
    # backends + the tool sources off the new client, then closes the OLD client.
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr(
        "src.reconfig.httpx.AsyncClient", lambda **kw: new_client
    )
    old_client = _FakeHttpClient("old")
    svc = _StubSvc()
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.network.external_proxy"})

    # New client installed; old client closed.
    assert deps.http_cloud is new_client
    assert old_client.closed is True
    # All three cloud backends rebuilt off the new client (each exactly once).
    assert svc.calls == {"stt": 1, "llm": 1, "tts": 1}
    assert rt.stt_backend == ("backend", "stt", 42)   # tts_timeout pushed before create
    # Tool sources rebuilt against the new client.
    assert hub.set_calls == [["new-src"]]


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_build_failure_keeps_old_client(monkeypatch):
    # If constructing the new client raises, the old client stays in place and nothing
    # is rebuilt.
    def boom(**kw):
        raise RuntimeError("bad proxy")

    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", boom)
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["x"])
    old_client = _FakeHttpClient("old")
    svc = _StubSvc()
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.network.external_proxy"})

    assert deps.http_cloud is old_client      # unchanged
    assert old_client.closed is False         # never closed
    assert svc.calls == {}                    # no backend rebuild
    assert hub.set_calls == []                # no tools rebuild


@pytest.mark.asyncio
async def test_apply_job_http_covers_tools_and_backends_exactly_once(monkeypatch):
    # A coalesced job containing rebuild_http AND rebuild_tools/rebuild_backends must
    # rebuild backends + tools EXACTLY once (the http path covers them; the standalone
    # branches are skipped).
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", lambda **kw: new_client)
    old_client = _FakeHttpClient("old")
    svc = _StubSvc()
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    # external_proxy -> rebuild_http; stt.* -> rebuild_backends; mcp_servers -> rebuild_tools.
    await reconf.apply_job(
        {"core.network.external_proxy", "stt.selected", "core.mcp_servers"}
    )

    # Backends rebuilt once each (NOT twice), tools swapped once.
    assert svc.calls == {"stt": 1, "llm": 1, "tts": 1}
    assert hub.set_calls == [["new-src"]]
    assert deps.http_cloud is new_client
    assert old_client.closed is True


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_skips_offline_stt(monkeypatch):
    # Selected STT provider is OFFLINE (uses_http_cloud False) while llm/tts are cloud:
    # a proxy change must rebuild ONLY {"llm","tts"} (NOT stt -> no model reload), rebuild
    # tools, swap the client and close the old one.
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", lambda **kw: new_client)
    old_client = _FakeHttpClient("old")
    svc = _StubSvc(cloud_cats=("llm", "tts"))   # stt offline
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.network.external_proxy"})

    # Only the cloud stages rebuilt; the offline stt backend is left untouched.
    assert svc.calls == {"llm": 1, "tts": 1}
    assert rt.stt_backend == "old-stt"     # offline -> no reload
    assert rt.llm_backend == ("backend", "llm", 42)
    assert rt.tts_backend == ("backend", "tts", 42)   # tts_timeout pushed before create
    assert hub.set_calls == [["new-src"]]              # tools still rebuilt
    assert deps.http_cloud is new_client
    assert old_client.closed is True


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_also_rebuilds_offline_backend_changed_in_job(monkeypatch):
    # Coalesced job touching BOTH core.network.* (rebuild_http) AND a SELECTED OFFLINE
    # backend's own field (stt is offline here, llm/tts cloud). The http rebuild must
    # rebuild the UNION of the cloud stages (client changed) AND the offline stt (its
    # config changed) — so the offline change is NOT silently dropped — rebuild tools,
    # swap the client, and close the old one (nothing lost).
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", lambda **kw: new_client)
    old_client = _FakeHttpClient("old")
    svc = _StubSvc(cloud_cats=("llm", "tts"))   # stt offline, llm/tts cloud
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job(
        {"core.network.external_proxy", "stt.instances.vosk.model"}
    )

    # stt (offline, config changed) rebuilt despite NOT being cloud, plus the cloud stages.
    assert svc.calls == {"stt": 1, "llm": 1, "tts": 1}
    assert rt.stt_backend == ("backend", "stt", 42)   # offline stt change applied (not dropped)
    assert rt.llm_backend == ("backend", "llm", 42)
    assert rt.tts_backend == ("backend", "tts", 42)
    assert hub.set_calls == [["new-src"]]              # tools rebuilt
    assert deps.http_cloud is new_client               # client swapped
    assert old_client.closed is True                   # old client closed


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_also_rebuilds_vad_changed_in_job(monkeypatch):
    # Coalesced job touching BOTH core.network.* (rebuild_http) AND a vad instance
    # field. vad is OFFLINE (default _StubSvc cloud_cats has no vad), so it enters the
    # http rebuild only through the backend_categories(paths) half of the union — and
    # must be rebuilt EXACTLY once (not dropped, not doubled), alongside the cloud
    # stages, with tools rebuilt and the client swapped/closed as usual.
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", lambda **kw: new_client)
    old_client = _FakeHttpClient("old")
    svc = _StubSvc()   # default cloud_cats: stt/llm/tts cloud, vad offline
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job(
        {"core.network.external_proxy", "vad.instances.webrtc.auto_gain"}
    )

    # vad rebuilt exactly once via the union, plus each cloud stage exactly once.
    assert svc.calls == {"vad": 1, "stt": 1, "llm": 1, "tts": 1}
    assert rt.vad_backend == ("backend", "vad", 42)    # vad change applied (not dropped)
    assert rt.stt_backend == ("backend", "stt", 42)
    assert rt.llm_backend == ("backend", "llm", 42)
    assert rt.tts_backend == ("backend", "tts", 42)
    assert hub.set_calls == [["new-src"]]              # tools rebuilt
    assert deps.http_cloud is new_client               # client swapped
    assert old_client.closed is True                   # old client closed


@pytest.mark.asyncio
async def test_apply_job_rebuild_http_offline_only_reloads_no_model(monkeypatch):
    # ALL selected providers are offline (uses_http_cloud False): a proxy change rebuilds
    # NO backends (no svc.create calls -> no model reload), but tools are still rebuilt and
    # the client is still swapped + the old one closed.
    monkeypatch.setattr("src.reconfig.build_sources", lambda c, h, s: ["new-src"])
    new_client = _FakeHttpClient("new")
    monkeypatch.setattr("src.reconfig.httpx.AsyncClient", lambda **kw: new_client)
    old_client = _FakeHttpClient("old")
    svc = _StubSvc(cloud_cats=())   # every stage offline
    hub = _FakeHub()
    rt = _http_runtime(svc, hub)
    deps = types.SimpleNamespace(http_cloud=old_client, tts_timeout=30)
    svc.attach(deps)
    reconf = Reconfigurator(rt, deps, asyncio.Queue())

    await reconf.apply_job({"core.network.external_proxy"})

    # No model reload at all: not a single create() call across the stages.
    assert svc.calls == {}
    assert rt.stt_backend == "old-stt"
    assert rt.llm_backend == "old-llm"
    assert rt.tts_backend == "old-tts"
    assert deps.tts_timeout == 30          # tts not in cats -> timeout push skipped
    assert hub.set_calls == [["new-src"]]  # tools still rebuilt (OpenWeatherMap uses http_cloud)
    assert deps.http_cloud is new_client   # client still swapped
    assert old_client.closed is True       # old client still closed


# --- apply_job (Tier 3c: device reconcile) -----------------------------------

class _FakeManager:
    """Stand-in for DeviceManager: records reconfigure() calls; can be told to raise."""

    def __init__(self, *, fail=False):
        self._fail = fail
        self.reconfigures = 0

    async def reconfigure(self):
        self.reconfigures += 1
        if self._fail:
            raise RuntimeError("reconcile boom")


@pytest.mark.asyncio
async def test_apply_job_rebuild_devices_invokes_manager_reconfigure():
    mgr = _FakeManager()
    rt = types.SimpleNamespace(manager=mgr)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    await reconf.apply_job({"core.devices"})
    assert mgr.reconfigures == 1


@pytest.mark.asyncio
async def test_apply_job_rebuild_devices_failure_is_attempted():
    mgr = _FakeManager(fail=True)
    rt = types.SimpleNamespace(manager=mgr)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    await reconf.apply_job({"core.esphome.port"})
    assert mgr.reconfigures == 1


# --- apply_job (Tier 3c: reminders subsystem hot toggle) ---------------------

class _FakeReminderScheduler:
    """Stand-in for ReminderScheduler: records start/stop and carries .deliver/.store."""

    instances = []

    def __init__(self, store):
        self.store = store
        self.deliver = None
        self.started = False
        self.stopped = False
        _FakeReminderScheduler.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class _FakeRemindersStore:
    """Stand-in for RemindersStore: records the path and whether close() was called."""

    instances = []

    def __init__(self, path):
        self.path = path
        self.closed = False
        _FakeRemindersStore.instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_reminder_instances():
    _FakeReminderScheduler.instances.clear()
    _FakeRemindersStore.instances.clear()
    yield
    _FakeReminderScheduler.instances.clear()
    _FakeRemindersStore.instances.clear()


def _reminders_runtime(*, enabled, scheduler, reminders_store):
    """Runtime stub for _rebuild_reminders: core.reminders.enabled, a manager with an
    announce coroutine, and swappable scheduler/reminders_store/hub. The reminders.db
    path now derives from the hardcoded config_store.DATA_DIR, not core.context.dir."""
    async def _announce(device, text):  # the deliver callback the enable path wires
        return None

    return types.SimpleNamespace(
        core=types.SimpleNamespace(
            reminders=types.SimpleNamespace(enabled=enabled),
        ),
        manager=types.SimpleNamespace(announce=_announce),
        scheduler=scheduler,
        reminders_store=reminders_store,
    )


@pytest.mark.asyncio
async def test_apply_job_reminders_enable_starts_scheduler_and_rebuilds_tools(monkeypatch):
    # Enabling while scheduler is None: a store + scheduler are created, deliver is wired
    # to manager.announce, both refs are published, start() is awaited, and the tool set
    # is rebuilt (so build_sources can register the reminders source).
    monkeypatch.setattr("src.reminders.RemindersStore", _FakeRemindersStore)
    monkeypatch.setattr("src.reminders.ReminderScheduler", _FakeReminderScheduler)
    # reminders.db lives under the hardcoded DATA_DIR; pin it so we can assert the path.
    monkeypatch.setattr("src.config_store.DATA_DIR", "/tmp/zk-rem")
    rebuilt = []

    rt = _reminders_runtime(enabled=True, scheduler=None, reminders_store=None)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    monkeypatch.setattr(reconf, "_rebuild_tools",
                        lambda: rebuilt.append(True) or _async_none())

    await reconf.apply_job({"core.reminders.enabled"})

    assert len(_FakeRemindersStore.instances) == 1
    store = _FakeRemindersStore.instances[0]
    assert store.path == os.path.join("/tmp/zk-rem", "reminders.db")
    assert len(_FakeReminderScheduler.instances) == 1
    scheduler = _FakeReminderScheduler.instances[0]
    assert scheduler.store is store
    assert scheduler.deliver is rt.manager.announce   # delivery wired to the speaker route
    assert rt.scheduler is scheduler                  # published for build_sources gating
    assert rt.reminders_store is store
    assert scheduler.started is True                  # start() awaited
    assert rebuilt == [True]                           # tools rebuilt (source registered)


@pytest.mark.asyncio
async def test_apply_job_reminders_disable_stops_scheduler_and_closes_store(monkeypatch):
    # Disabling while a scheduler exists: scheduler stopped, rt.scheduler cleared BEFORE
    # the tools rebuild (so build_sources omits the source), tools rebuilt, store closed.
    monkeypatch.setattr("src.reminders.RemindersStore", _FakeRemindersStore)
    monkeypatch.setattr("src.reminders.ReminderScheduler", _FakeReminderScheduler)
    existing_store = _FakeRemindersStore("/tmp/zk-rem/reminders.db")
    existing_sched = _FakeReminderScheduler(existing_store)
    seen_scheduler_at_rebuild = {}

    rt = _reminders_runtime(enabled=False, scheduler=existing_sched,
                            reminders_store=existing_store)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())

    async def _fake_rebuild_tools():
        # The scheduler ref must already be None here so build_sources drops the source.
        seen_scheduler_at_rebuild["value"] = rt.scheduler

    monkeypatch.setattr(reconf, "_rebuild_tools", _fake_rebuild_tools)

    await reconf.apply_job({"core.reminders.enabled"})

    assert existing_sched.stopped is True
    assert seen_scheduler_at_rebuild["value"] is None   # cleared BEFORE the tools rebuild
    assert rt.scheduler is None
    assert existing_store.closed is True
    assert rt.reminders_store is None


@pytest.mark.asyncio
async def test_apply_job_reminders_noop_when_already_enabled(monkeypatch):
    # enabled -> enabled (no transition): nothing happens. No store/scheduler created,
    # the existing ones are left intact, no tools rebuild.
    monkeypatch.setattr("src.reminders.RemindersStore", _FakeRemindersStore)
    monkeypatch.setattr("src.reminders.ReminderScheduler", _FakeReminderScheduler)
    existing_store = _FakeRemindersStore("/tmp/zk-rem/reminders.db")
    existing_sched = _FakeReminderScheduler(existing_store)
    _FakeRemindersStore.instances.clear()
    _FakeReminderScheduler.instances.clear()
    rebuilt = []

    rt = _reminders_runtime(enabled=True, scheduler=existing_sched,
                            reminders_store=existing_store)
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    monkeypatch.setattr(reconf, "_rebuild_tools",
                        lambda: rebuilt.append(True) or _async_none())

    await reconf.apply_job({"core.reminders.enabled"})

    assert _FakeRemindersStore.instances == []     # no new store
    assert _FakeReminderScheduler.instances == []  # no new scheduler
    assert rt.scheduler is existing_sched          # unchanged
    assert rt.reminders_store is existing_store
    assert existing_sched.stopped is False
    assert existing_store.closed is False
    assert rebuilt == []                            # no tools rebuild


async def _async_none():
    """Awaitable returning None, used to stub the async _rebuild_tools via a lambda."""
    return None
