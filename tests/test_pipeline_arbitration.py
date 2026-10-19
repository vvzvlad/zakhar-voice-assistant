"""Pipeline-level tests for the cross-device wake arbitration hold: a wake run with an
arbiter hook waits for its verdict like the wake-word gate waits for the verifier —
nothing reaches STT and the device sees no `listening` until it is let through; a lost
run streams on silently to its end-point and is logged as a duplicate with its
transcript. The arbiter itself is a recording double in most tests; the verdicts are
delivered by the test."""

import pytest

from src.pipeline_events import StageEvent
from tests.test_pipeline import (
    FRAME,
    FakeRunsStore,
    FakeSttBackend,
    install_fake_vad,
    make_pipeline,
    patch_llm,
    set_small_vad_thresholds,
    settle,
    types_of,
)
from src.wake_arbiter import WakeArbiter
from tests.test_pipeline_wakeword_gate import WAKE, FakeWakewordVerifier, _drain_verify
from tests.test_wake_arbiter import preroll


class RecordingArbiter:
    """The arbiter double: records each registered token, rules nothing by itself."""

    def __init__(self):
        self.tokens = []

    def register(self, pipeline, token):
        self.tokens.append(token)

    def preroll_in(self, pipeline, token):
        pass

    def run_changed(self, pipeline, token):
        pass

    def run_event(self, pipeline, event, data):
        pass


def make_arbitrated(tmp_path, monkeypatch, *, verifier=None, **kw):
    pipeline, events = make_pipeline(tmp_path, monkeypatch, **kw)
    pipeline.rt.wakeword_backend = verifier
    arbiter = pipeline.arbiter = RecordingArbiter()
    return pipeline, events, arbiter


async def test_the_cleaner_wake_wins_though_its_pre_roll_comes_after_its_wake(tmp_path, monkeypatch):
    # Two speakers; the second is the last online device to wake, so the group is
    # complete at its wake, but its cleaner pre-roll streams in only afterwards.
    patch_llm(monkeypatch)
    first, first_events = make_pipeline(tmp_path, monkeypatch, name="first")
    second, second_events = make_pipeline(tmp_path, monkeypatch, name="second")
    first.arbiter = second.arbiter = WakeArbiter(first.rt, online_count=lambda: 2)

    await first.on_start("a", WAKE)
    await first.on_audio(preroll(1000, 3162))  # 10 dB
    await second.on_start("b", WAKE)
    await second.on_audio(preroll(100, 1000))  # 20 dB
    assert StageEvent.STT_VAD_START in types_of(second_events)
    # The loser stays held, silently, until its own phrase ends.
    assert types_of(first_events) == [StageEvent.RUN_START, StageEvent.STT_START]
    await settle(second)


async def test_a_winner_the_verifier_rejects_leaves_the_phrase_to_the_other_device(
        tmp_path, monkeypatch):
    # The speaker would win, but its verifier (a window longer than the pre-roll, as
    # in production) rejects it: the panel's run goes on instead of being dropped.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker", runs_store=store)
    panel, panel_events = make_pipeline(tmp_path, monkeypatch, name="panel")
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=False, window_ms=2500)
    panel.rt.wakeword_backend = FakeWakewordVerifier(accepted=True, window_ms=2500)
    panel.has_speaker = False
    speaker.arbiter = panel.arbiter = WakeArbiter(speaker.rt, online_count=lambda: 2)

    for p in (panel, speaker):
        await p.on_start("cid", WAKE)
        await p.on_audio(preroll(100, 1000))
    for p in (speaker, panel):
        await p.on_audio(bytes(16000))  # 500 ms of live audio fills the verify window
        await _drain_verify(p)
    assert StageEvent.STT_VAD_START in types_of(panel_events)
    assert (store.records[-1]["result"], store.records[-1]["reason"]) == ("rejected", "wakeword_reject")


async def test_a_lone_false_wake_leaves_no_cooldown_for_another_device(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker")
    panel, panel_events = make_pipeline(tmp_path, monkeypatch, name="panel")
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=False)
    speaker.arbiter = panel.arbiter = WakeArbiter(speaker.rt, online_count=lambda: 1)

    await speaker.on_start("cid", WAKE)
    await speaker.on_audio(b"\x01\x02" * 30000)  # > the verify window
    await _drain_verify(speaker)
    await panel.on_start("cid", WAKE)
    assert StageEvent.STT_VAD_START in types_of(panel_events)


async def test_a_held_run_starts_listening_only_once_it_won(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await pipeline.on_start("cid", WAKE)
    await pipeline.on_audio(b"\x01\x02" * 1000)
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START]

    pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    assert types_of(events)[-1] == StageEvent.STT_VAD_START
    await pipeline.on_stop(False)
    assert store.records[-1]["stt_text"] == "включи свет"


def held_phrase(pipeline):
    """Small VAD thresholds and a phrase the fake VAD end-points: three speech frames,
    then silence."""
    set_small_vad_thresholds(pipeline)
    install_fake_vad(pipeline, [True, True, True, False, False, False, False, False, False])


async def test_a_lost_run_streams_on_to_its_end_point_and_is_logged_with_its_transcript(
        tmp_path, monkeypatch):
    # Lost at its wake (as in the cooldown): no `listening`, no transcript to the
    # device, no handler, no ack — the phrase is only recognized for the run log.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, ack=True)
    ack_calls = []
    pipeline._schedule_ack = lambda: ack_calls.append(1)
    held_phrase(pipeline)

    await pipeline.on_start("cid", WAKE)
    pipeline.arbitration_verdict(arbiter.tokens[-1], False)
    await pipeline.on_audio(FRAME * 3)  # the phrase goes on
    await settle(pipeline)
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START]
    await pipeline.on_audio(FRAME * 6)  # ...and ends
    await settle(pipeline)

    assert events == [(StageEvent.RUN_START, {}), (StageEvent.STT_START, {}),
                      (StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {})]
    assert ack_calls == []
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"], rec["llm_text"]) == (
        "rejected", "duplicate", "включи свет", "")


async def test_a_speakerless_device_the_cooldown_dropped_is_shown_the_winners_answer(
        tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    sent = []
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker", stt_text="свет")
    panel, _ = make_pipeline(tmp_path, monkeypatch, name="panel", stt_text="свет")
    panel.has_speaker = False
    speaker.arbiter = panel.arbiter = WakeArbiter(
        speaker.rt, online_count=lambda: 1,
        send_reply=lambda p, text: sent.append((p.name, text)))
    for p in (speaker, panel):
        held_phrase(p)

    await speaker.on_start("a", WAKE)  # the lone online device wins at once
    await panel.on_start("b", WAKE)  # in the cooldown: dropped
    for p in (panel, speaker):
        await p.on_audio(FRAME * 9)
        await settle(p)
    assert sent == [("panel", "готово")]


async def test_a_duplicate_whose_stt_fails_is_still_logged(tmp_path, monkeypatch):
    # The phrase had already ended when the verdict came: the run ends at once.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    stt = FakeSttBackend()

    async def down(pcm):
        raise RuntimeError("stt down")

    stt.transcribe = down
    pipeline, _, arbiter = make_arbitrated(tmp_path, monkeypatch, stt_backend=stt, runs_store=store)
    held_phrase(pipeline)

    await pipeline.on_start("cid", WAKE)
    await pipeline.on_audio(FRAME * 9)
    pipeline.arbitration_verdict(arbiter.tokens[-1], False)
    await settle(pipeline)
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"]) == ("rejected", "duplicate", "")


async def test_a_duplicate_without_speech_is_not_transcribed(tmp_path, monkeypatch):
    # The STT would make words up for the noise, as for any no-speech run.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    stt = FakeSttBackend(text="продолжение следует")
    calls = []
    real = stt.transcribe

    async def counted(pcm):
        calls.append(len(pcm))
        return await real(pcm)

    stt.transcribe = counted
    pipeline, _, arbiter = make_arbitrated(tmp_path, monkeypatch, stt_backend=stt, runs_store=store)
    set_small_vad_thresholds(pipeline)
    install_fake_vad(pipeline, [False] * 400)

    await pipeline.on_start("cid", WAKE)
    pipeline.arbitration_verdict(arbiter.tokens[-1], False)
    for _ in range(300):
        await pipeline.on_audio(FRAME)
    await settle(pipeline)
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"], calls) == ("rejected", "duplicate", "", [])


@pytest.mark.parametrize("verifier_first", [True, False])
async def test_a_won_run_with_a_verifier_is_released_only_after_both(
        tmp_path, monkeypatch, verifier_first):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, verifier=FakeWakewordVerifier(accepted=True))

    await pipeline.on_start("cid", WAKE)
    if verifier_first:
        await pipeline.on_audio(b"\x01\x02" * 30000)  # > the verify window
        await _drain_verify(pipeline)
    else:
        pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    assert StageEvent.STT_VAD_START not in types_of(events)

    if verifier_first:
        pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    else:
        await pipeline.on_audio(b"\x01\x02" * 30000)
        await _drain_verify(pipeline)
    assert types_of(events).count(StageEvent.STT_VAD_START) == 1


async def test_an_utterance_that_ended_during_the_hold_is_finalized_on_release(
        tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="ок")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="свет", runs_store=store)
    held_phrase(pipeline)

    await pipeline.on_start("cid", WAKE)
    await pipeline.on_audio(FRAME * 9)  # speech then silence: an end-point
    assert not pipeline._finalized

    pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    await settle(pipeline)
    rec = store.records[-1]
    assert (rec["reason"], rec["stt_text"]) == ("endpoint", "свет")
    assert types_of(events)[-1] == StageEvent.RUN_END
