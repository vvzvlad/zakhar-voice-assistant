"""Pipeline-level tests for the follow-up window (docs/device-protocol.md, "Follow-up").

After an answer that played out on a device with caps.follow_up and the followup stage
on, the run does not end: the pipeline listens again without a wake word. Silence ends
the session unrecorded; a phrase goes on only when the follow-up check accepts its
first transcript. These tests drive Pipeline directly with a fake check and the
existing pipeline test doubles.
"""

import asyncio

import pytest

from src import context
from src.followup import FollowupGate, FollowupVerdict, NoFollowup
from src.pipeline_events import StageEvent
from src.plugins.vad.webrtc import WebRtcVadBackend
from tests.test_pipeline import (
    FRAME,
    PHRASE,
    PHRASE_VAD,
    FakePlayer,
    FakeRunsStore,
    FakeVad,
    install_phrase_vad,
    make_pipeline,
    patch_llm,
    settle,
    types_of,
)

ANSWERED = [
    StageEvent.RUN_START, StageEvent.STT_START, StageEvent.STT_VAD_START,
    StageEvent.STT_VAD_END, StageEvent.STT_END, StageEvent.INTENT_START,
    StageEvent.INTENT_END, StageEvent.TTS_START,
]


class FakeGate(FollowupGate):
    """Follow-up check returning a fixed verdict; records (name, request, reply, utterance)."""

    def __init__(self, accepted=True, probability=0.9, window_ms=200):
        self.window_ms = window_ms
        self.verdict = FollowupVerdict(accepted, probability)
        self.calls = []

    async def check(self, name, request, reply, utterance):
        self.calls.append((name, request, reply, utterance))
        return self.verdict


class OutcomePlayer(FakePlayer):
    """The device's play hook reporting every stream with `outcome`."""

    def __init__(self, outcome="played"):
        super().__init__()
        self.outcome = outcome

    async def __call__(self, mime, audio):
        await super().__call__(mime, audio)
        return {"outcome": self.outcome, "dry": None, "gap_ms": None}


def _make(tmp_path, monkeypatch, *, gate=None, outcome="played", capable=True, **kw):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", **kw)
    pipeline.rt.followup_backend = FakeGate() if gate is None else gate
    pipeline.play = OutcomePlayer(outcome)
    pipeline.has_follow_up = capable
    install_phrase_vad(pipeline)
    return pipeline, events


async def _answer(pipeline):
    """One wake run, answered and played."""
    await pipeline.on_start("cid", "Захар")
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)


async def _say(pipeline, text):
    """A phrase in the open follow-up window, ended by the server's VAD."""
    pipeline.rt.stt_backends["fake"].text = text
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)


def _context(tmp_path):
    return context.load_context(str(tmp_path / "context_dev.txt"))


async def test_a_played_answer_opens_the_window_instead_of_ending(tmp_path, monkeypatch):
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    # No RUN_END: the device gets `listening` again in the same session.
    assert types_of(events) == [*ANSWERED, StageEvent.STT_VAD_START]
    # The next phrase is heard and checked against the turn that opened the window.
    await _say(pipeline, "а в спальне")
    assert pipeline.rt.followup_backend.calls == [("Захар", "включи свет", "готово", "а в спальне")]


@pytest.mark.parametrize("outcome, capable, gate", [
    ("dropped", True, FakeGate()),      # the answer was cut off (a wake, play_stop)
    ("played", False, FakeGate()),      # the device does not reopen its microphone
    ("played", True, NoFollowup()),     # the stage is off
    ("played", True, None),             # no backend (test fixtures)
])
async def test_no_window_unless_played_capable_and_enabled(tmp_path, monkeypatch,
                                                          outcome, capable, gate):
    pipeline, events = _make(tmp_path, monkeypatch, outcome=outcome, capable=capable)
    pipeline.rt.followup_backend = gate
    await _answer(pipeline)
    assert types_of(events) == [*ANSWERED, StageEvent.RUN_END]
    # Nothing listens any more: a phrase starts no run.
    await _say(pipeline, "а в спальне")
    assert types_of(events) == [*ANSWERED, StageEvent.RUN_END]


async def test_silence_in_the_window_ends_the_session_unrecorded(tmp_path, monkeypatch):
    store = FakeRunsStore()
    pipeline, events = _make(tmp_path, monkeypatch, runs_store=store)
    # The wake run hears its phrase; the window that opens after its answer, silence.
    scripts = iter([PHRASE_VAD, [False]])
    pipeline.rt.vad_backend = WebRtcVadBackend(vad_factory=lambda: FakeVad(next(scripts)))
    await _answer(pipeline)
    events.clear()

    # 240 ms of silence past the 200 ms window: the VAD ends the run with no_speech.
    await pipeline.on_audio(FRAME * 12)
    await settle(pipeline)
    assert events == [(StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {})]
    assert len(store.records) == 1          # the answered run only
    assert pipeline.rt.followup_backend.calls == []


@pytest.mark.parametrize("probability, reason, outcome", [
    (0.1, "followup_reject", "rejected"),
    (None, "followup_error", "error"),
])
async def test_a_rejected_phrase_is_dropped_silently(tmp_path, monkeypatch,
                                                     probability, reason, outcome):
    store = FakeRunsStore()
    gate = FakeGate(accepted=False, probability=probability)
    pipeline, events = _make(tmp_path, monkeypatch, gate=gate, runs_store=store)
    acks = []
    pipeline._schedule_ack = lambda: acks.append(1)
    await _answer(pipeline)
    events.clear()

    await _say(pipeline, "ну и погодка сегодня")
    assert gate.calls == [("Захар", "включи свет", "готово", "ну и погодка сегодня")]
    # No chime, no transcript, no handler, no answer: `transcribing`, then `idle`.
    assert acks == [1]                      # the wake run's only
    assert events == [(StageEvent.STT_VAD_END, {}), (StageEvent.STT_END, {"text": ""}),
                      (StageEvent.RUN_END, {})]
    assert len(pipeline.play.calls) == 1    # the first answer only
    assert _context(tmp_path) == [{"role": "user", "content": "включи свет"},
                                  {"role": "assistant", "content": "готово"}]
    rec = store.records[-1]
    # The log shows what was heard, though the device never got it.
    assert (rec["result"], rec["reason"], rec["stt_text"]) == ("rejected", reason,
                                                               "ну и погодка сегодня")
    check = rec["attempts"][0]
    assert (check["kind"], check["outcome"], check["probability"]) == ("followup", outcome, probability)
    assert check["t"] >= 0
    assert rec["attempts"][1]["outcome"] == "rejected"


async def test_an_accepted_phrase_chimes_runs_and_opens_the_window_again(tmp_path, monkeypatch):
    store = FakeRunsStore()
    gate = FakeGate(accepted=True, probability=0.8)
    pipeline, events = _make(tmp_path, monkeypatch, gate=gate, runs_store=store)
    pipeline._schedule_ack = lambda: events.append(("ack", None))
    await _answer(pipeline)
    events.clear()

    await _say(pipeline, "а в спальне")
    # The chime comes once the check accepted the phrase, before its transcript.
    assert types_of(events) == [
        StageEvent.STT_VAD_END, "ack", StageEvent.STT_END, StageEvent.INTENT_START,
        StageEvent.INTENT_END, StageEvent.TTS_START, StageEvent.STT_VAD_START,
    ]
    assert len(gate.calls) == 1
    assert dict(events)[StageEvent.STT_END] == {"text": "а в спальне"}
    assert _context(tmp_path)[-2:] == [{"role": "user", "content": "а в спальне"},
                                       {"role": "assistant", "content": "готово"}]
    rec = store.records[-1]
    assert (rec["result"], rec["stt_text"]) == ("ok", "а в спальне")
    assert [(a["kind"], a["outcome"]) for a in rec["attempts"]][:2] == [
        ("followup", "accepted"), ("chain", "answered")]
    # The name stays the session wake's across follow-up turns.
    await _say(pipeline, "и в коридоре")
    assert gate.calls[-1] == ("Захар", "а в спальне", "готово", "и в коридоре")


async def test_proactive_speech_closes_the_window_first(tmp_path, monkeypatch):
    # A reminder in the open window: the session ends before it plays, so the
    # speaker does not hear itself as a phrase.
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    events.clear()
    await pipeline.speak("напоминаю: выключить духовку")
    assert events == [(StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {})]
    await _say(pipeline, "напоминаю: выключить духовку")
    assert pipeline.rt.followup_backend.calls == []


async def test_no_window_while_proactive_speech_is_queued(tmp_path, monkeypatch):
    # A reminder queued on the device behind the answer (its play() still waits for
    # `played`): the window must not open into it. Once it has played, the next
    # answer opens the window again.
    pipeline, events = _make(tmp_path, monkeypatch)
    release = asyncio.Event()
    play = pipeline.play
    held = []

    async def hold_first(mime, audio):
        if not held:
            held.append(1)
            await release.wait()
        return await play(mime, audio)

    pipeline.play = hold_first
    reminder = asyncio.create_task(pipeline.speak("напоминаю"))
    while not held and not reminder.done():
        await asyncio.sleep(0)
    await _answer(pipeline)
    assert types_of(events) == [*ANSWERED, StageEvent.RUN_END]
    release.set()
    await reminder
    events.clear()
    await _answer(pipeline)
    assert types_of(events) == [*ANSWERED, StageEvent.STT_VAD_START]


async def test_a_cancelled_run_ends_the_window(tmp_path, monkeypatch):
    # The stop word cancels the session (the device server sends `idle`): audio that
    # still arrives in the window is no phrase.
    pipeline, _ = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    pipeline.cancel_run()
    await _say(pipeline, "стоп")
    assert pipeline.rt.followup_backend.calls == []


async def test_a_follow_up_phrase_keeps_its_start(tmp_path, monkeypatch):
    # trim_start_ms cuts a wake word's tail; a follow-up phrase has none to cut.
    pipeline, _ = _make(tmp_path, monkeypatch)
    pipeline.rt.core.vad.trim_start_ms = 10
    await _answer(pipeline)
    heard = []
    pipeline.rt.stt_backends["fake"].transcribe = lambda pcm: heard.append(pcm) or _text()
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert heard == [PHRASE]


async def _text():
    return "а в спальне"


async def test_a_wake_in_the_window_starts_a_normal_run(tmp_path, monkeypatch):
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    assert types_of(events)[-1] == StageEvent.STT_VAD_START  # the window is open

    pipeline.play.outcome = "dropped"
    events.clear()
    await _answer(pipeline)
    assert pipeline.rt.followup_backend.calls == []   # no follow-up check of a wake run
    assert types_of(events) == [*ANSWERED, StageEvent.RUN_END]
