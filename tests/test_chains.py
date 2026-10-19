"""Recognition chains: the runner (src/chains.py) and its use by the pipeline."""

import asyncio
import types

import pytest

from src import chains, context
from src.core_config import ChainConfig
from src.llm import LlmResult
from src.pipeline_events import StageEvent
from src.plugins.llm.base import LlmConfig
from src.stage_errors import Declined, StageError
from tests.test_pipeline import (install_fake_vad, make_pipeline, set_small_vad_thresholds,
                                 types_of)


class Stt:
    """STT double. `text` is returned, or `error` raised; `gate` (an Event) holds the
    call until set. Records the vocabulary of each call, and whether it started or was
    cancelled."""

    def __init__(self, text="", *, error=None, gate=None, supports_vocabulary=False):
        self.text, self.error, self.gate = text, error, gate
        self.supports_vocabulary = supports_vocabulary
        self.calls = []
        self.started = asyncio.Event()
        self.cancelled = False

    async def transcribe(self, pcm, **kw):
        self.calls.append(kw.get("vocabulary"))
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return self.text


class Handler:
    """Handler backend double driven by FakeStage: `outcome` is a reply string or an
    exception to raise (Declined / StageError)."""

    def __init__(self, outcome, *, tool_used=False):
        self.outcome, self.tool_used = outcome, tool_used
        self.calls = []   # (user_text, has_next)


class FakeStage:
    def __init__(self, backend, hub, cfg):
        self.backend = backend

    async def respond(self, req, *, on_filler=None, has_next=False):
        b = self.backend
        b.calls.append((req.user_text, has_next))
        if isinstance(b.outcome, BaseException):
            raise b.outcome
        return LlmResult(reply=b.outcome, model="m", tokens=1, rounds=[],
                         request_debug={}, tool_used=b.tool_used)


class Svc:
    def __init__(self, vocabulary=()):
        self.cfg = LlmConfig()
        self._vocabulary = list(vocabulary)

    def instance_config(self, _category, _provider_id):
        return self.cfg

    def vocabulary(self, _handler_id):
        return list(self._vocabulary)


def chain(name, stt=None, handler=None, **kw):
    return ChainConfig(name=name, stt=stt or name, handler=handler or name, **kw)


def runtime(chain_list, stt=None, handlers=None, vocabulary=()):
    return types.SimpleNamespace(
        core=types.SimpleNamespace(chains=chain_list),
        svc=Svc(vocabulary),
        stt_backends=dict(stt or {}),
        llm_backends=dict(handlers or {}),
        hub=None,
    )


def request(text):
    return types.SimpleNamespace(user_text=text)


async def voice(rt, **kw):
    return await chains.run(rt, device="dev", request=request, pcm=b"\x01\x02", **kw)


@pytest.fixture(autouse=True)
def fake_stage(monkeypatch):
    monkeypatch.setattr("src.llm.LlmStage", FakeStage)


def outcomes(run):
    return [(a["chain"], a["outcome"]) for a in run.attempts]


# --- the runner -----------------------------------------------------------------------

async def test_declined_goes_to_next_chain_which_answers():
    nlu = Handler(Declined("Не понял команду", reason="no alias phrase matched"))
    llm = Handler("Сейчас 12 часов")
    rt = runtime([chain("nlu"), chain("llm")],
                 stt={"nlu": Stt("который час"), "llm": Stt("который час?")},
                 handlers={"nlu": nlu, "llm": llm})
    heard = []

    run = await voice(rt, on_text=lambda text, _t: heard.append(text))

    assert outcomes(run) == [("nlu", "declined"), ("llm", "answered")]
    assert (run.chain, run.reply, run.text, run.answered) == (
        "llm", "Сейчас 12 часов", "который час?", True)
    # has_next is True for every chain but the last enabled one.
    assert nlu.calls == [("который час", True)] and llm.calls == [("который час?", False)]
    assert heard == ["который час"]   # once, with the first non-empty transcript
    assert run.attempts[0]["stt"] == "nlu" and run.attempts[0]["handler"] == "nlu"
    # Why the handler declined rides on its attempt; an answer carries none.
    assert [a["detail"] for a in run.attempts] == ["no alias phrase matched", None]


@pytest.mark.parametrize("first", [
    Stt(""),                                   # heard nothing
    Stt("Продолжение следует..."),             # known Whisper hallucination
    Stt("[unk] [unk]"),                        # narrowed Vosk heard only out-of-grammar words
    Stt(error=StageError("stt", "groq down")),
])
async def test_empty_or_failed_stt_goes_to_next_chain(first):
    rt = runtime([chain("a"), chain("b")], stt={"a": first, "b": Stt("включи свет")},
                 handlers={"a": Handler("не должен"), "b": Handler("Готово")})
    run = await voice(rt)
    assert run.reply == "Готово" and run.error is None
    assert rt.llm_backends["a"].calls == []
    assert run.attempts[0]["outcome"] in ("empty", "stt_error")


async def test_handler_error_without_tool_goes_to_next_chain():
    rt = runtime([chain("a"), chain("b")], stt={"a": Stt("x"), "b": Stt("x")},
                 handlers={"a": Handler(StageError("llm", "500")), "b": Handler("ok")})
    run = await voice(rt)
    assert outcomes(run) == [("a", "handler_error"), ("b", "answered")]
    assert run.attempts[0]["error"] == "500"


async def test_handler_error_after_tool_stops_with_reply_error_after_tools():
    err = StageError("llm", "timeout on final round")
    err.partial = {"tool_used": True, "model": "m1", "rounds": [{"round": 1}]}
    later = Handler("не должен")
    rt = runtime([chain("a"), chain("b")], stt={"a": Stt("включи"), "b": Stt("включи")},
                 handlers={"a": Handler(err), "b": later})
    run = await voice(rt)
    assert outcomes(run) == [("a", "handler_error")]
    assert later.calls == []                    # the action is not repeated
    assert (run.chain, run.reply, run.answered) == ("a", "Сделал, но ответить не смог.", False)
    assert run.error is err and run.partial["model"] == "m1"


async def test_all_declined_speaks_last_decline_reply():
    rt = runtime([chain("a"), chain("b")], stt={"a": Stt("x"), "b": Stt("y")},
                 handlers={"a": Handler(Declined("первый")), "b": Handler(Declined("второй"))})
    run = await voice(rt)
    assert (run.reply, run.chain, run.error, run.answered) == ("второй", None, None, False)


async def test_all_failed_speaks_fallback_of_last_handler():
    rt = runtime([chain("a"), chain("b")], stt={"a": Stt("x"), "b": Stt("y")},
                 handlers={"a": Handler(StageError("llm", "500")),
                           "b": Handler(StageError("llm", "429", kind="rate_limit"))})
    run = await voice(rt)
    assert run.reply == LlmConfig().reply_rate_limit
    assert str(run.error) == "429" and run.chain is None


async def test_nothing_heard_runs_no_handler():
    rt = runtime([chain("a"), chain("b")], stt={"a": Stt(""), "b": Stt(" ")},
                 handlers={"a": Handler("x"), "b": Handler("y")})
    heard = []
    run = await voice(rt, on_text=lambda text, _t: heard.append(text))
    assert run.reply is None and heard == [] and run.text == ""
    assert outcomes(run) == [("a", "empty"), ("b", "empty")]


@pytest.mark.parametrize("accept", [True, False])
async def test_gate_is_asked_once_about_the_first_transcript_before_any_handler(accept):
    first = Handler(Declined("не понял"))
    rt = runtime([chain("a"), chain("b"), chain("c")],
                 stt={"a": Stt(""), "b": Stt("а в спальне"), "c": Stt("а в спальне")},
                 handlers={"a": Handler("x"), "b": first, "c": Handler("готово")})
    asked, heard = [], []

    async def gate(text):
        asked.append((text, list(heard), list(first.calls)))
        return accept

    run = await voice(rt, on_text=lambda text, _t: heard.append(text), gate=gate)
    assert asked == [("а в спальне", [], [])]
    if accept:
        assert heard == ["а в спальне"] and run.reply == "готово"
    else:
        assert heard == [] and first.calls == [] and run.reply is None
        assert outcomes(run) == [("a", "empty"), ("b", "rejected")]
        assert run.attempts[1]["text"] == "а в спальне"


async def test_disabled_chain_is_skipped_and_last_enabled_has_no_next():
    off = Handler("x")
    rt = runtime([chain("a"), chain("off", enabled=False)],
                 stt={"a": Stt("y"), "off": Stt("z")}, handlers={"a": Handler("ok"), "off": off})
    run = await voice(rt)
    assert rt.llm_backends["a"].calls == [("y", False)] and off.calls == []
    assert outcomes(run) == [("a", "answered")]


async def test_missing_backend_is_a_failed_attempt():
    # A chain whose backend is not built (e.g. its hot rebuild failed) is passed over.
    rt = runtime([chain("a"), chain("b")], stt={"b": Stt("x")}, handlers={"b": Handler("ok")})
    run = await voice(rt)
    assert outcomes(run) == [("a", "stt_error"), ("b", "answered")]


@pytest.mark.parametrize("narrow, supports, vocabulary, expected", [
    (True, True, ["свет"], ["свет"]),
    (False, True, ["свет"], None),     # the chain does not narrow
    (True, False, ["свет"], None),     # the STT cannot take a vocabulary
    (True, True, [], None),            # the handler has none to give
])
async def test_narrowing(narrow, supports, vocabulary, expected):
    stt = Stt("свет", supports_vocabulary=supports)
    rt = runtime([chain("a", narrow=narrow)], stt={"a": stt}, handlers={"a": Handler("ok")},
                 vocabulary=vocabulary)
    await voice(rt)
    assert stt.calls == [expected]


async def test_parallel_stt_starts_with_the_first_and_unused_result_is_cancelled():
    gate = asyncio.Event()
    first = Stt("включи свет", gate=gate)
    parallel = Stt("включи свет", gate=asyncio.Event())   # never released
    sequential = Stt("включи свет")
    rt = runtime([chain("a"), chain("b", parallel=True), chain("c")],
                 stt={"a": first, "b": parallel, "c": sequential},
                 handlers={"a": Handler("ok"), "b": Handler("x"), "c": Handler("y")})

    task = asyncio.create_task(voice(rt))
    await asyncio.wait_for(parallel.started.wait(), 5)
    assert not task.done() and sequential.calls == []   # c starts only when reached
    gate.set()
    run = await asyncio.wait_for(task, 5)

    assert outcomes(run) == [("a", "answered"), ("b", "cancelled")]
    assert parallel.cancelled and sequential.calls == []


async def test_parallel_result_is_used_when_reached():
    rt = runtime([chain("a"), chain("b", parallel=True)],
                 stt={"a": Stt(""), "b": Stt("включи свет")},
                 handlers={"a": Handler("x"), "b": Handler("ok")})
    run = await voice(rt)
    assert outcomes(run) == [("a", "empty"), ("b", "answered")]
    assert rt.stt_backends["b"].calls == [None]   # recognized once, at the start


async def test_text_request_enters_at_the_handler():
    stt = Stt("не должен")
    rt = runtime([chain("a"), chain("b")], stt={"a": stt, "b": stt},
                 handlers={"a": Handler(Declined("нет")), "b": Handler("ok")})
    run = await chains.run(rt, device="dev", request=request, text="который час")
    assert stt.calls == []
    assert [(a["stt"], a["text"], a["outcome"]) for a in run.attempts] == [
        (None, "который час", "declined"), (None, "который час", "answered")]
    assert run.t_stt == 0


def test_spoken_fallback_maps_kind_to_configured_phrase():
    # "rate_limit" gets its dedicated line; any other kind the generic error phrase.
    cfg = LlmConfig()
    assert chains.spoken_fallback(StageError("llm", "x", kind="rate_limit"), cfg) \
        == cfg.reply_rate_limit
    assert chains.spoken_fallback(StageError("llm", "x", kind="timeout"), cfg) == cfg.reply_error


def test_referenced_instances_and_starts_at_speech_follow_enabled_chains():
    cs = [chain("a", "vosk", "simple-nlu"), chain("b", "groq", "openrouter", enabled=False),
          chain("c", "groq", "simple-nlu", parallel=True), chain("d", "yandex", "groq")]
    assert chains.referenced_instances(cs) == {
        ("stt", "vosk"), ("stt", "groq"), ("stt", "yandex"),
        ("llm", "simple-nlu"), ("llm", "groq")}
    assert [c.name for c in chains.starts_at_speech(runtime(cs), cs)] == ["a", "c"]


async def test_chains_on_one_stt_and_vocabulary_share_one_recognition():
    # Commands -> Jev -> LLM (parallel), all on one cloud STT: one recognition, and
    # neither later chain opens a session of its own, the parallel one included.
    stt = Stt("включи свет")
    cs = [chain("cmd", stt="cloud"), chain("jev", stt="cloud"),
          chain("llm", stt="cloud", parallel=True)]
    rt = runtime(cs, stt={"cloud": stt},
                 handlers={"cmd": Handler(Declined("нет")), "jev": Handler(Declined("нет")),
                           "llm": Handler("ok")})
    assert [c.name for c in chains.starts_at_speech(rt, cs)] == ["cmd"]

    run = await voice(rt)

    assert stt.calls == [None]
    assert outcomes(run) == [("cmd", "declined"), ("jev", "declined"), ("llm", "answered")]
    assert [a["text"] for a in run.attempts] == ["включи свет"] * 3
    assert [a["t_stt"] for a in run.attempts[1:]] == [0, 0]


class FailsOnceStt(Stt):
    """Stt double whose first call raises `error`; later calls return `text`."""

    async def transcribe(self, pcm, **kw):
        try:
            return await super().transcribe(pcm, **kw)
        finally:
            self.error = None


async def test_failed_recognition_is_not_shared():
    stt = FailsOnceStt("включи свет", error=StageError("stt", "timeout"))
    rt = runtime([chain("a", stt="s"), chain("b", stt="s")], stt={"s": stt},
                 handlers={"a": Handler("не должен"), "b": Handler("ok")})
    run = await voice(rt)
    assert outcomes(run) == [("a", "stt_error"), ("b", "answered")]
    assert stt.calls == [None, None]


async def test_same_stt_with_another_vocabulary_recognizes_on_its_own():
    stt = Stt("свет", supports_vocabulary=True)
    cs = [chain("a", stt="s"), chain("b", stt="s", narrow=False, parallel=True)]
    rt = runtime(cs, stt={"s": stt}, handlers={"a": Handler(Declined("нет")), "b": Handler("ok")},
                 vocabulary=["свет"])
    assert [c.name for c in chains.starts_at_speech(rt, cs)] == ["a", "b"]
    run = await voice(rt)
    assert outcomes(run) == [("a", "declined"), ("b", "answered")]
    assert stt.calls == [["свет"], None]


# --- the pipeline over chains -------------------------------------------------------

def two_chain_pipeline(tmp_path, monkeypatch, first, second, **kw):
    pipeline, events = make_pipeline(tmp_path, monkeypatch, **kw)
    pipeline.rt.core.chains = [chain("nlu"), chain("llm")]
    pipeline.rt.stt_backends = {"nlu": first[0], "llm": second[0]}
    pipeline.rt.llm_backends = {"nlu": first[1], "llm": second[1]}
    return pipeline, events


async def test_pipeline_runs_chains_and_records_them(tmp_path, monkeypatch):
    from tests.test_pipeline import FakeRunsStore

    store = FakeRunsStore()
    pipeline, events = two_chain_pipeline(
        tmp_path, monkeypatch,
        (Stt("свет"), Handler(Declined("Не понял"))),
        (Stt("включи свет"), Handler("Готово", tool_used=True)),
        runs_store=store,
    )
    await pipeline.on_start("cid", None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    # One STT_END (the first non-empty transcript) and one INTENT_START, as before.
    assert types_of(events) == [
        StageEvent.RUN_START, StageEvent.STT_START, StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END, StageEvent.STT_END, StageEvent.INTENT_START,
        StageEvent.INTENT_END, StageEvent.TTS_START, StageEvent.RUN_END,
    ]
    assert dict(events)[StageEvent.STT_END] == {"text": "свет"}
    rec = store.records[0]
    assert (rec["chain"], rec["stt_text"], rec["llm_text"], rec["result"]) == (
        "llm", "включи свет", "Готово", "tool")
    # The chain attempts, then the TTS variant that spoke the reply.
    assert [a["outcome"] for a in rec["attempts"]] == ["declined", "answered", "spoke"]
    # The context history gets the serving chain's text and reply.
    history = context.load_context(pipeline._context_path, max_turns=5, ttl_seconds=300)
    assert [m["content"] for m in history] == ["включи свет", "Готово"]


async def test_pipeline_all_declined_speaks_decline_and_keeps_context_clean(
        tmp_path, monkeypatch):
    from tests.test_pipeline import FakeRunsStore

    store = FakeRunsStore()
    pipeline, events = two_chain_pipeline(
        tmp_path, monkeypatch,
        (Stt("свет"), Handler(Declined("первый"))),
        (Stt("свет"), Handler(Declined("Не понял команду"))),
        runs_store=store,
    )
    await pipeline.on_start("cid", None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert dict(events)[StageEvent.TTS_START] == {"text": "Не понял команду"}
    assert (store.records[0]["result"], store.records[0]["chain"]) == ("ok", None)
    assert context.load_context(pipeline._context_path, max_turns=5, ttl_seconds=300) == []


class StreamingStt(Stt):
    """Stt double that also opens a live session: records fed audio, finish() returns text."""

    def __init__(self, text):
        super().__init__(text)
        self.fed, self.finished, self.closed = [], False, False
        self.stream_vocabulary = "unset"

    def open_stream(self, **kw):
        self.stream_vocabulary = kw.get("vocabulary")
        outer = self

        class Session:
            def feed(self, pcm):
                outer.fed.append(pcm)

            async def finish(self):
                outer.finished = True
                return outer.text

            async def aclose(self):
                outer.closed = True

        return Session()


async def test_pipeline_opens_streams_for_first_and_parallel_chains_only(tmp_path, monkeypatch):
    a, b, c = StreamingStt("включи свет"), StreamingStt("включи свет"), StreamingStt("x")
    pipeline, _events = make_pipeline(tmp_path, monkeypatch)
    pipeline.rt.core.chains = [chain("a"), chain("b", parallel=True), chain("c")]
    pipeline.rt.stt_backends = {"a": a, "b": b, "c": c}
    pipeline.rt.llm_backends = {"a": Handler("ok"), "b": Handler("x"), "c": Handler("y")}

    await pipeline.on_start("cid", None)
    assert set(pipeline._stt_streams) == {"a", "b"}
    await pipeline.on_audio(b"\x01\x02" * 100)
    assert a.fed == b.fed == [b"\x01\x02" * 100] and c.fed == []
    await pipeline.on_stop(False)

    assert a.finished and not c.finished and c.calls == []
    assert b.closed                    # the unused parallel session is released
    assert pipeline._stt_streams == {}


class GatedStreamingStt(StreamingStt):
    """StreamingStt whose session finish() waits on `gate`, so a run can be caught mid-STT."""

    def __init__(self, text):
        super().__init__(text)
        self.finish_started, self.gate = asyncio.Event(), asyncio.Event()

    def open_stream(self, **kw):
        outer = self

        class Session:
            def feed(self, pcm):
                outer.fed.append(pcm)

            async def finish(self):
                outer.finish_started.set()
                await outer.gate.wait()
                return outer.text

            async def aclose(self):
                outer.closed = True

        return Session()


async def _end_utterance(pipeline):
    # Speech, then silence: the VAD end-points the utterance and the run task starts.
    for _ in range(12):
        await pipeline.on_audio(b"\x01\x02" * 320)
        if pipeline._run_task is not None:
            return pipeline._run_task
    raise AssertionError("the utterance was never end-pointed")


async def test_new_wake_mid_stt_closes_every_chain_session(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.LlmStage", FakeStage)
    pipeline, _events = make_pipeline(tmp_path, monkeypatch)
    set_small_vad_thresholds(pipeline)
    a, b = GatedStreamingStt("a"), GatedStreamingStt("b")
    pipeline.rt.core.chains = [chain("a"), chain("b", parallel=True)]
    pipeline.rt.stt_backends = {"a": a, "b": b}
    pipeline.rt.llm_backends = {"a": Handler("ok"), "b": Handler("ok")}
    install_fake_vad(pipeline, [True, True, True] + [False] * 20)

    await pipeline.on_start("c1", None)
    run1 = await _end_utterance(pipeline)
    await asyncio.wait_for(a.finish_started.wait(), 2)
    await pipeline.on_start("c2", None)
    await asyncio.gather(run1, return_exceptions=True)

    assert run1.cancelled()
    assert a.closed and b.closed


async def test_new_wake_while_the_run_waits_for_the_lock_closes_its_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.llm.LlmStage", FakeStage)
    pipeline, _events = make_pipeline(tmp_path, monkeypatch)
    set_small_vad_thresholds(pipeline)
    a = GatedStreamingStt("a")
    pipeline.rt.core.chains = [chain("a")]
    pipeline.rt.stt_backends = {"a": a}
    pipeline.rt.llm_backends = {"a": Handler("ok")}
    install_fake_vad(pipeline, [True, True, True] + [False] * 20)

    await pipeline._lock.acquire()  # another run on this speaker, e.g. an agent's `ask`
    try:
        await pipeline.on_start("c1", None)
        run1 = await _end_utterance(pipeline)
        await asyncio.sleep(0.05)  # the run took its sessions and now waits for the lock
        await pipeline.on_start("c2", None)
        await asyncio.gather(run1, return_exceptions=True)
    finally:
        pipeline._lock.release()

    assert run1.cancelled()
    assert a.closed
