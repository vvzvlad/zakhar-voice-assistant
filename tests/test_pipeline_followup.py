"""Pipeline-level tests for the follow-up window (docs/device-protocol.md, "Follow-up").

After an answer that played out on a device with caps.follow_up and the followup stage
on, the run does not end: the pipeline listens again without a wake word. Silence ends
the session unrecorded; a phrase goes on only when the follow-up check accepts its
first transcript. These tests drive Pipeline directly with a fake check and the
existing pipeline test doubles.
"""

import pytest

from src import context
from src.followup import FollowupGate, FollowupVerdict, NoFollowup
from src.pipeline_events import StageEvent
from tests.test_pipeline import (
    FRAME,
    FakePlayer,
    FakeRunsStore,
    install_fake_vad,
    make_pipeline,
    patch_llm,
    set_small_vad_thresholds,
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
        return self.outcome


def _make(tmp_path, monkeypatch, *, gate=None, outcome="played", capable=True, **kw):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", **kw)
    pipeline.rt.followup_backend = FakeGate() if gate is None else gate
    pipeline.play = OutcomePlayer(outcome)
    pipeline.has_follow_up = capable
    return pipeline, events


async def _answer(pipeline):
    """One wake run, answered and played."""
    await pipeline.on_start("cid", "Захар")
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)


async def _say(pipeline, text):
    """A phrase in the open follow-up window, finalized by a device stop."""
    pipeline.rt.stt_backends["fake"].text = text
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)


def _context(tmp_path):
    return context.load_context(str(tmp_path / "context_dev.txt"))


async def test_a_played_answer_opens_the_window_instead_of_ending(tmp_path, monkeypatch):
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    # No RUN_END: the device gets `listening` again in the same session.
    assert types_of(events) == [*ANSWERED, StageEvent.STT_VAD_START]
    assert pipeline._followup == ("Захар", "включи свет", "готово")
    assert pipeline._finalized is False


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
    assert pipeline._followup is None


async def test_silence_in_the_window_ends_the_session_unrecorded(tmp_path, monkeypatch):
    store = FakeRunsStore()
    pipeline, events = _make(tmp_path, monkeypatch, runs_store=store)
    set_small_vad_thresholds(pipeline)
    install_fake_vad(pipeline, [False])
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
    # The name stays the session wake's across follow-up turns.
    assert pipeline._followup == ("Захар", "а в спальне", "готово")
    assert _context(tmp_path)[-2:] == [{"role": "user", "content": "а в спальне"},
                                       {"role": "assistant", "content": "готово"}]
    rec = store.records[-1]
    assert (rec["result"], rec["stt_text"]) == ("ok", "а в спальне")
    assert [(a["kind"], a["outcome"]) for a in rec["attempts"]][:2] == [
        ("followup", "accepted"), ("chain", "answered")]


async def test_proactive_speech_closes_the_window_first(tmp_path, monkeypatch):
    # A reminder in the open window: the session ends before it plays, so the
    # speaker does not hear itself as a phrase.
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    events.clear()
    await pipeline.speak("напоминаю: выключить духовку")
    assert events == [(StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {})]
    assert pipeline._followup is None
    await _say(pipeline, "напоминаю: выключить духовку")
    assert pipeline.rt.followup_backend.calls == []


async def test_a_follow_up_phrase_keeps_its_start(tmp_path, monkeypatch):
    # trim_start_ms cuts a wake word's tail; a follow-up phrase has none to cut.
    pipeline, _ = _make(tmp_path, monkeypatch)
    pipeline.rt.core.vad.trim_start_ms = 10
    await _answer(pipeline)
    heard = []
    pipeline.rt.stt_backends["fake"].transcribe = lambda pcm: heard.append(pcm) or _text()
    fed = b"\x01\x02" * 400
    await pipeline.on_audio(fed)
    await pipeline.on_stop(False)
    assert heard == [fed]


async def _text():
    return "а в спальне"


async def test_a_wake_in_the_window_starts_a_normal_run(tmp_path, monkeypatch):
    pipeline, events = _make(tmp_path, monkeypatch)
    await _answer(pipeline)
    assert pipeline._followup is not None

    pipeline.play.outcome = "dropped"
    events.clear()
    await _answer(pipeline)
    assert pipeline.rt.followup_backend.calls == []   # no follow-up check of a wake run
    assert types_of(events) == [*ANSWERED, StageEvent.RUN_END]
    assert pipeline._followup is None
