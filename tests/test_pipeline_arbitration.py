"""Pipeline-level tests for the cross-device wake arbitration hold: a wake run with an
arbiter hook waits for its verdict like the wake-word gate waits for the verifier —
nothing reaches STT until it is let through, while the device is shown `listening` as
soon as its wake is confirmed; a lost run streams on to its end-point and is logged as a
duplicate with its transcript. The arbiter itself is a recording double in most tests; the verdicts are
delivered by the test."""

import asyncio

import pytest

from src.pipeline_events import StageEvent
from tests.test_pipeline import (
    FRAME,
    PHRASE,
    FakeRunsStore,
    FakeSttBackend,
    install_fake_vad,
    install_phrase_vad,
    make_pipeline,
    patch_llm,
    set_small_vad_thresholds,
    settle,
    types_of,
)
from src.wake_arbiter import COOLDOWN_MS, WakeArbiter
from tests.test_pipeline_followup import TAIL, _answer, _make
from tests.test_pipeline_wakeword_gate import HEAD_BYTES, WAKE, FakeWakewordVerifier, _make_text_checked
from tests.test_wake_arbiter import preroll


class RecordingArbiter:
    """The arbiter double: records each registered token; rules nothing by itself, or,
    with `rule_on_register`, rules that verdict inside register() as the real arbiter
    does for a wake in a cooldown."""

    def __init__(self, rule_on_register=None):
        self.tokens = []
        self.rule_on_register = rule_on_register

    def register(self, pipeline, token):
        self.tokens.append(token)
        if self.rule_on_register is not None:
            pipeline.arbitration_verdict(token, self.rule_on_register)

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


class _RecordingStreamSession:
    """Streaming STT session double recording every fed chunk + finish() result."""

    def __init__(self, text):
        self.fed = []
        self._text = text

    def feed(self, pcm):
        self.fed.append(pcm)

    async def finish(self):
        return self._text

    async def aclose(self):
        pass


class _RecordingStreamSttBackend:
    def __init__(self, text):
        self._text = text
        self.session = _RecordingStreamSession(text)

    async def transcribe(self, pcm):
        return self._text  # unused: the streaming path takes finish()

    def open_stream(self):
        return self.session


async def test_a_held_run_feeds_its_streaming_stt_the_backlog_once_on_release(
        tmp_path, monkeypatch):
    # Streaming STT under the hold: the stream is not opened or fed while the run is
    # held; on release the whole buffered backlog is fed exactly once, then later chunks
    # feed incrementally (no double-feed, no lost audio).
    patch_llm(monkeypatch, reply="готово")
    stt = _RecordingStreamSttBackend("включи свет")
    pipeline, events, arbiter = make_arbitrated(tmp_path, monkeypatch, stt_backend=stt)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE)
    backlog = PHRASE[:HEAD_BYTES]  # does not end the phrase yet
    await pipeline.on_audio(backlog)
    assert stt.session.fed == []
    pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    assert stt.session.fed == [backlog]

    tail = PHRASE[HEAD_BYTES:]  # ends the phrase
    await pipeline.on_audio(tail)
    assert stt.session.fed == [backlog, tail]
    await settle(pipeline)
    assert StageEvent.STT_END in types_of(events)


async def test_the_cleaner_wake_wins_though_its_pre_roll_comes_after_its_wake(tmp_path, monkeypatch):
    # Two speakers; the second is the last online device to wake, so the group is
    # complete at its wake, but its cleaner pre-roll streams in only afterwards.
    patch_llm(monkeypatch)
    first, first_events = make_pipeline(tmp_path, monkeypatch, name="first")
    second, second_events = make_pipeline(tmp_path, monkeypatch, name="second")
    first.arbiter = second.arbiter = WakeArbiter(online_count=lambda: 2)

    first.on_wake("a", WAKE)
    await first.on_audio(preroll(1000, 3162))  # 10 dB
    second.on_wake("b", WAKE)
    await second.on_audio(preroll(100, 1000))  # 20 dB
    assert StageEvent.STT_VAD_START in types_of(second_events)
    # The loser keeps the `listening` it was shown at its wake and stays held until its
    # own phrase ends.
    assert types_of(first_events) == [
        StageEvent.RUN_START, StageEvent.STT_START, StageEvent.STT_VAD_START]
    await settle(second)


async def test_a_winner_the_verifier_rejects_leaves_the_phrase_to_the_other_device(
        tmp_path, monkeypatch):
    # The speaker, answering, would win, but its verifier (a window longer than the
    # pre-roll, as in production) rejects its wake over the answer: the panel's run goes
    # on instead of being dropped.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker", runs_store=store)
    panel, panel_events = make_pipeline(tmp_path, monkeypatch, name="panel")
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=False, window_ms=2500)
    panel.has_speaker = False
    speaker.arbiter = panel.arbiter = WakeArbiter(online_count=lambda: 2)

    for p in (panel, speaker):
        p.on_wake("cid", WAKE, over_answer=p is speaker)
        await p.on_audio(preroll(100, 1000))
    for p in (speaker, panel):
        await p.on_audio(bytes(16000))  # 500 ms of live audio fills the verify window
        await settle(p)
    assert StageEvent.STT_VAD_START in types_of(panel_events)
    assert (store.records[-1]["result"], store.records[-1]["reason"]) == ("rejected", "wakeword_reject")


async def test_a_device_that_loses_by_its_own_confirmation_is_not_shown_listening(
        tmp_path, monkeypatch):
    # Two answering speakers with equal pre-rolls: the first wake's is confirmed first and
    # shown at once; the second's confirmation completes the group, which the earlier
    # wake wins: the second never lights up.
    patch_llm(monkeypatch)
    first, first_events = make_pipeline(tmp_path, monkeypatch, name="first")
    second, second_events = make_pipeline(tmp_path, monkeypatch, name="second")
    for p in (first, second):
        p.rt.wakeword_backend = FakeWakewordVerifier(accepted=True, window_ms=2500)
        install_fake_vad(p, [True] * 400)  # no end-point: each check waits for its window
    first.arbiter = second.arbiter = WakeArbiter(online_count=lambda: 2)
    verdicts = []
    ruled = second.arbitration_verdict
    second.arbitration_verdict = lambda token, won, **kw: (verdicts.append(won), ruled(token, won, **kw))

    for p in (first, second):
        p.on_wake("cid", WAKE, over_answer=True)
        await p.on_audio(preroll(100, 1000))
    await first.on_audio(bytes(16000))  # 500 ms of live audio fills the verify window
    await settle(first)
    assert StageEvent.STT_VAD_START in types_of(first_events)
    assert verdicts == []  # its check still holds the group open
    await second.on_audio(bytes(16000))
    await settle(second)
    assert verdicts == [False]
    assert StageEvent.STT_VAD_START not in types_of(second_events)


async def test_a_lone_false_wake_over_an_answer_leaves_no_cooldown_for_another_device(
        tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker")
    panel, panel_events = make_pipeline(tmp_path, monkeypatch, name="panel")
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=False)
    speaker.arbiter = panel.arbiter = WakeArbiter(online_count=lambda: 1)

    speaker.on_wake("cid", WAKE, over_answer=True)
    await speaker.on_audio(b"\x01\x02" * 30000)  # > the verify window
    await settle(speaker)
    panel.on_wake("cid", WAKE)
    assert StageEvent.STT_VAD_START in types_of(panel_events)


async def test_a_held_run_shows_listening_at_once_and_is_released_once_it_won(
        tmp_path, monkeypatch):
    # No verifier: the wake needs no check, so the device is shown `listening` at once,
    # and only once, though the run waits for the arbitration.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="захар включи свет", runs_store=store)
    held_phrase(pipeline)

    pipeline.on_wake("cid", WAKE)
    await pipeline.on_audio(FRAME * 3)  # the phrase goes on
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START, StageEvent.STT_VAD_START]

    pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    assert types_of(events).count(StageEvent.STT_VAD_START) == 1
    await pipeline.on_audio(FRAME * 6)  # ...and ends
    await settle(pipeline)
    assert store.records[-1]["stt_text"] == "захар включи свет"


def held_phrase(pipeline):
    """Small VAD thresholds and a phrase the fake VAD end-points: three speech frames,
    then silence."""
    set_small_vad_thresholds(pipeline)
    install_fake_vad(pipeline, [True, True, True, False, False, False, False, False, False])


async def test_a_lost_run_streams_on_to_its_end_point_and_is_logged_with_its_transcript(
        tmp_path, monkeypatch):
    # Lost inside its registration (as in the cooldown): never taken — no `listening`, no
    # transcript to the device, no handler, no ack — the phrase is only recognized for the
    # run log, and at its end the device server ends the wake (end_wake).
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, ack=True)
    pipeline.arbiter = RecordingArbiter(rule_on_register=False)
    pipeline.end_wake = lambda cid: events.append(("end_wake", cid))
    ack_calls = []
    pipeline._schedule_ack = lambda: ack_calls.append(1)
    held_phrase(pipeline)

    pipeline.on_wake("cid", WAKE)
    await pipeline.on_audio(FRAME * 3)  # the phrase goes on
    await settle(pipeline)
    assert events == []
    await pipeline.on_audio(FRAME * 6)  # ...and ends
    await settle(pipeline)

    assert events == [("end_wake", "cid")]
    assert ack_calls == []
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"], rec["llm_text"]) == (
        "rejected", "duplicate", "включи свет", "")


async def test_a_speakerless_device_the_cooldown_dropped_is_shown_the_winners_answer(
        tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    sent = []
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker", stt_text="захар свет")
    panel, _ = make_pipeline(tmp_path, monkeypatch, name="panel", stt_text="захар свет")
    panel.has_speaker = False
    speaker.arbiter = panel.arbiter = WakeArbiter(
        online_count=lambda: 1,
        send_reply=lambda p, text: sent.append((p.name, text)))
    for p in (speaker, panel):
        held_phrase(p)

    speaker.on_wake("a", WAKE)  # the lone online device wins at once
    panel.on_wake("b", WAKE)  # in the cooldown: dropped
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

    pipeline.on_wake("cid", WAKE)
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

    pipeline.on_wake("cid", WAKE)
    pipeline.arbitration_verdict(arbiter.tokens[-1], False)
    for _ in range(300):
        await pipeline.on_audio(FRAME)
    await settle(pipeline)
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"], calls) == ("rejected", "duplicate", "", [])


@pytest.mark.parametrize("verifier_first", [True, False])
async def test_a_won_run_with_a_verifier_shows_listening_once_the_verifier_confirms(
        tmp_path, monkeypatch, verifier_first):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, verifier=FakeWakewordVerifier(accepted=True))

    pipeline.on_wake("cid", WAKE, over_answer=True)
    if verifier_first:
        await pipeline.on_audio(b"\x01\x02" * 30000)  # > the verify window
        await settle(pipeline)
        # Confirmed while the arbitration still holds the run: `listening` at once.
        assert types_of(events).count(StageEvent.STT_VAD_START) == 1
        pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    else:
        pipeline.arbitration_verdict(arbiter.tokens[-1], True)
        # Won, but the wake word is not confirmed yet: nothing shown.
        assert StageEvent.STT_VAD_START not in types_of(events)
        await pipeline.on_audio(b"\x01\x02" * 30000)
        await settle(pipeline)
    assert types_of(events).count(StageEvent.STT_VAD_START) == 1


async def test_an_utterance_that_ended_during_the_hold_is_finalized_on_release(
        tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="ок")
    store = FakeRunsStore()
    pipeline, events, arbiter = make_arbitrated(
        tmp_path, monkeypatch, stt_text="захар свет", runs_store=store)
    held_phrase(pipeline)

    pipeline.on_wake("cid", WAKE)
    await pipeline.on_audio(FRAME * 9)  # speech then silence: an end-point
    assert StageEvent.STT_VAD_END not in types_of(events) and store.records == []

    pipeline.arbitration_verdict(arbiter.tokens[-1], True)
    await settle(pipeline)
    rec = store.records[-1]
    assert (rec["reason"], rec["stt_text"]) == ("endpoint", "захар свет")
    assert types_of(events)[-1] == StageEvent.RUN_END


async def test_a_follow_up_window_notes_when_its_vad_first_hears_speech(tmp_path, monkeypatch):
    speaker, _ = _make(tmp_path, monkeypatch)
    await _answer(speaker)
    await speaker.on_audio(TAIL + PHRASE[:640])  # one speech frame: under min_speech_ms
    assert speaker.followup_speech() is None
    await speaker.on_audio(PHRASE[640:1280])
    assert speaker.followup_speech() is not None


async def speaker_window_and_panel(tmp_path, monkeypatch, age=0.5):
    """A speaker whose follow-up window heard a phrase start `age` seconds ago (its next
    answer opens no window) and a speakerless panel on one real arbiter: (speaker,
    panel, the panel's runs store, the replies sent to the panel)."""
    monkeypatch.setattr("src.wake_arbiter.WINDOW_MS", 20)
    speaker, _ = _make(tmp_path, monkeypatch, name="speaker")
    await _answer(speaker)
    speaker.play.outcome = "dropped"
    await speaker.on_audio(TAIL + PHRASE[:1280])
    speaker._current.speech_at -= age
    store = FakeRunsStore()
    panel, _ = make_pipeline(tmp_path, monkeypatch, name="panel", stt_text="захар свет",
                             runs_store=store)
    panel.has_speaker = False
    held_phrase(panel)
    sent = []
    speaker.arbiter = panel.arbiter = WakeArbiter(
        online_count=lambda: 2, online_pipelines=lambda: [speaker, panel],
        send_reply=lambda p, text: sent.append((p.name, text)))
    return speaker, panel, store, sent


@pytest.mark.parametrize("age, ended, kept", [
    (0.5, False, True),
    (COOLDOWN_MS / 1000 + 0.5, False, False),  # the speech of an earlier phrase
    (0.5, True, False),                        # its phrase ended before the wake
])
async def test_a_follow_up_window_hearing_the_phrase_keeps_it_from_a_panels_wake(
        tmp_path, monkeypatch, age, ended, kept):
    # The speaker sends no wake inside its window's phrase; the panel wakes on it.
    speaker, panel, store, sent = await speaker_window_and_panel(tmp_path, monkeypatch, age)
    if ended:
        await speaker.on_audio(PHRASE[1280:])
        await settle(speaker)
    panel.on_wake("b", WAKE)
    await asyncio.sleep(0.1)  # the group's window runs out
    await panel.on_audio(FRAME * 9)
    await speaker.on_audio(PHRASE[1280:])  # the window's phrase ends and is answered
    for p in (panel, speaker):
        await settle(p)
    rec = store.records[-1]
    assert (rec["result"], rec["reason"]) == (("rejected", "duplicate") if kept else ("ok", "endpoint"))
    assert sent == ([("panel", "готово")] if kept else [])


async def test_a_window_whose_device_woke_in_the_group_does_not_keep_the_phrase(
        tmp_path, monkeypatch):
    # The speaker's own wake over its window joined the group and its check rejected
    # it: the panel keeps the phrase.
    speaker, panel, store, _ = await speaker_window_and_panel(tmp_path, monkeypatch)
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=False)
    speaker.end_wake = lambda cid: None
    speaker.on_wake("s2", WAKE, over_answer=True)
    panel.on_wake("b", WAKE)
    await speaker.on_audio(b"\x01\x02" * 30000)  # past the check's window: rejected
    await asyncio.sleep(0.1)
    await panel.on_audio(FRAME * 9)
    for p in (panel, speaker):
        await settle(p)
    rec = store.records[-1]
    assert (rec["result"], rec["reason"]) == ("ok", "endpoint")


async def test_a_wake_over_the_playing_answer_stops_it_inside_another_devices_cooldown(
        tmp_path, monkeypatch):
    # The panel's lone wake has just won its group; the speaker's wake over its own playing
    # answer arrives inside that group's cooldown. It keeps no phrase, so it takes no part
    # in the arbitration: once its check confirms it, it stops the answer.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    speaker, _ = make_pipeline(tmp_path, monkeypatch, name="speaker", runs_store=store)
    panel, _ = make_pipeline(tmp_path, monkeypatch, name="panel")
    panel.has_speaker = False
    speaker.rt.wakeword_backend = FakeWakewordVerifier(accepted=True, window_ms=100)
    speaker.arbiter = panel.arbiter = WakeArbiter(online_count=lambda: 1)
    install_phrase_vad(speaker)

    panel.on_wake("cid", WAKE)
    speaker.on_wake("cid", WAKE, over_answer=True, over_playback=True)
    await speaker.on_audio(PHRASE[:HEAD_BYTES])
    await settle(speaker)
    await speaker.on_audio(PHRASE[HEAD_BYTES:])
    await settle(speaker)
    assert (store.records[-1]["result"], store.records[-1]["reason"]) == ("ok", "wake_stop")


@pytest.mark.parametrize("together,playing,runs", [
    (True, False, True), (False, False, False), (True, True, False)])
async def test_a_winner_another_device_woke_with_needs_no_wake_word_in_its_transcript(
        tmp_path, monkeypatch, together, playing, runs):
    # Run 4527: the speaker kept the phrase, but its transcript lost «Захар», which both
    # panels' transcripts carried. A winner of a group in which another device woke too is
    # confirmed by that, so its phrase runs; a wake alone in its group still needs the word,
    # and so does one ruled while a device plays (run 4480: the panels heard the speaker's
    # answer ahead of «Захар, стоп»).
    monkeypatch.setattr("src.wake_arbiter.WINDOW_MS", 50)
    speaker, _, store = _make_text_checked(tmp_path, monkeypatch, "сколько времени")
    panel, _ = make_pipeline(tmp_path, monkeypatch, name="panel")
    panel.has_speaker = False
    install_phrase_vad(panel)
    speaker.arbiter = panel.arbiter = WakeArbiter(online_count=lambda: 2,
                                                  any_playing=lambda: playing)

    devices = (speaker, panel) if together else (speaker,)
    for p in devices:
        p.on_wake("cid", WAKE)
    for p in devices:
        await p.on_audio(PHRASE)
    await asyncio.sleep(0.15)  # the window runs out: the group is ruled
    for p in devices:
        await settle(p)
    rec = store.records[-1]
    assert (rec["result"], rec["llm_text"]) == (("ok", "готово") if runs else ("rejected", ""))
