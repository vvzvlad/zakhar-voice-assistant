"""Pipeline-level tests for the server-side wake-word checks.

An ESP32 detects «Захар» locally and streams a pre-roll + live audio. A wake over the
device's own running answer (over_answer) goes through the quick check: a
second-stage verifier runs on the pre-roll head, and until it rules the wake is held
and nothing reaches the device. On accept the run is taken. Over the device's playing
stream it only stops that stream and ends at once — its phrase carries the end of the
answer, so no STT/LLM/TTS — recorded as 'ok', reason 'wake_stop'; over a silent session it
proceeds as a normal run. On reject it is never taken: it ends silently (no events, no
STT/LLM/TTS, no ack chime), the device server ends the wake (end_wake) and the run is
recorded as 'rejected'. Any other wake is taken at once and confirmed by the wake word
in its recognised phrase. These tests drive Pipeline directly with a controllable fake
verifier (verdict/score/timeout/fail_open) and the existing pipeline test doubles.

The verify runs in a background task scheduled (synchronously) from on_audio, so
each test settles the pipeline before asserting on the post-verdict state.
"""

import asyncio
import io
import wave

import pytest

from src.pipeline_events import StageEvent
from src.wakeword import PassthroughVerifier, WakewordVerdict, WakewordVerifier

# Reuse the existing pipeline test harness (fakes + helpers) verbatim.
from tests.test_pipeline import (
    FRAME,
    PHRASE,
    FakeRunEvents,
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

# A real wake-word run carries a non-empty wake_word_phrase; a None phrase is a
# manual button start (never gated). The gate keys off this phrase.
WAKE = "Захар"

# A verify window that fills inside a PHRASE, and the first chunk of a phrase that runs
# one frame past it without end-pointing yet (4 frames of trailing silence, 5 end it).
VERIFY_WINDOW_MS = 100
VERIFY_WINDOW_BYTES = VERIFY_WINDOW_MS * 16000 * 2 // 1000
HEAD_BYTES = VERIFY_WINDOW_BYTES + 640


class FakeWakewordVerifier(WakewordVerifier):
    """Controllable second-stage verifier.

    `verify()` returns a fixed verdict (accept/reject + score), optionally after a
    delay, and records the PCM it was handed. `hang=True` exceeds the pipeline's
    timeout so the gate falls back to the fail policy. Like the real backend,
    verify() never raises (a raising verifier would still be caught by the
    pipeline's except, but the contract is no-raise)."""

    def __init__(self, *, accepted=True, score=0.9, timeout_ms=300, fail_open=True,
                 hang=False, window_ms=1500):
        self.timeout_ms = timeout_ms
        self.fail_open = fail_open
        self.window_ms = window_ms
        self._accepted = accepted
        self._score = score
        self._hang = hang
        self.calls = []  # PCM handed to each verify()

    async def verify(self, pcm: bytes, over_answer: bool = False) -> WakewordVerdict:
        self.calls.append(pcm)
        if self._hang:
            # Sleep past the pipeline's timeout so wait_for cancels us.
            await asyncio.sleep(self.timeout_ms / 1000 + 1.0)
        return WakewordVerdict(accepted=self._accepted, score=self._score)


def _make_gated(tmp_path, monkeypatch, *, verifier, **kw):
    """Build a pipeline whose runtime carries `verifier` as the wakeword backend; the
    end_wake hook lands in the events as ("end_wake", cid)."""
    pipeline, events = make_pipeline(tmp_path, monkeypatch, **kw)
    pipeline.rt.wakeword_backend = verifier
    pipeline.end_wake = lambda cid: events.append(("end_wake", cid))
    return pipeline, events


# --- (a) accept -> over the playback the answer stops, else the run proceeds --


async def test_gate_accept_only_stops_the_answer(tmp_path, monkeypatch):
    # An accepted wake over the playing answer stops that answer and goes no further: its
    # audio carries the end of the answer it cut. It is taken (`listening`, which makes it
    # the device's session) and ends at once — an empty STT_END and RUN_END, no STT, no
    # handler, no ack — and the rest of its phrase is dropped. The record is `ok`,
    # reason `wake_stop`, with the check's latency and score.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    stt = FakeSttBackend("вторник захар сто")
    stt_calls = []
    orig = stt.transcribe

    async def spy(pcm):
        stt_calls.append(pcm)
        return await orig(pcm)

    stt.transcribe = spy
    verifier = FakeWakewordVerifier(accepted=True, score=0.91, window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier,
        stt_backend=stt, runs_store=store, ack=True,
    )
    ack_calls = []
    pipeline._schedule_ack = lambda: ack_calls.append(1)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE, over_answer=True, over_playback=True)
    # Before the verdict: nothing at all.
    assert events == []

    # Feed a chunk past the verify window (100 ms = 3200 bytes); the gate schedules
    # verify on the first chunk that crosses it.
    await pipeline.on_audio(PHRASE[:HEAD_BYTES])
    await settle(pipeline)
    # The verifier saw the pre-roll head sized to its window (3200 bytes), not the
    # whole chunk.
    assert len(verifier.calls) == 1
    assert len(verifier.calls[0]) == VERIFY_WINDOW_BYTES < HEAD_BYTES

    await pipeline.on_audio(PHRASE[HEAD_BYTES:])
    await settle(pipeline)
    assert events == [
        (StageEvent.RUN_START, {}), (StageEvent.STT_START, {}), (StageEvent.STT_VAD_START, {}),
        (StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {}),
    ]
    assert (stt_calls, ack_calls, pipeline.play.calls) == ([], [], [])
    [rec] = store.records
    assert (rec["result"], rec["reason"], rec["stt_text"], rec["llm_text"]) == ("ok", "wake_stop", "", "")
    assert rec["t_wakeword"] >= 0
    assert rec["wakeword_score"] == 0.91


async def test_gate_accept_over_a_silent_session_runs_the_phrase(tmp_path, monkeypatch):
    # Over the open session with nothing playing (its run still thinking) an accepted
    # wake runs its phrase through STT/LLM/TTS; its phrase is not checked for the wake
    # word again.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=True, window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier,
                                   stt_text="включи свет", runs_store=store)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE, over_answer=True)
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert StageEvent.TTS_START in types_of(events)
    rec = store.records[-1]
    assert (rec["result"], rec["stt_text"], rec["llm_text"]) == ("ok", "включи свет", "готово")


@pytest.mark.parametrize("text, outcome", [
    ("алиса включи свет", ("ok", "готово")),
    ("включи свет", ("rejected", "")),
])
async def test_a_wake_of_a_word_the_check_does_not_hear_is_decided_by_its_phrase(
        tmp_path, monkeypatch, text, outcome):
    # The Yandex Station spots «Алиса» itself and the check listens only for «захар», so it
    # could only reject the wake. Over an answer the wake is taken at once without asking the
    # verifier, and its phrase is checked for its word as that of any wake taken at once.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=False, window_ms=VERIFY_WINDOW_MS)
    verifier.hears = lambda word: word.lower() == "захар"
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier,
                                   stt_text=text, runs_store=store)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", "Алиса", over_answer=True)
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START,
                                StageEvent.STT_VAD_START]
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert verifier.calls == []
    rec = store.records[-1]
    assert (rec["result"], rec["llm_text"]) == outcome


@pytest.mark.parametrize("accepted", [False, True])
async def test_a_wake_over_the_music_is_decided_by_its_phrase(tmp_path, monkeypatch, accepted):
    # Over the device's music the check decides only whether the music ducks: the wake is
    # taken either way, and a phrase without the wake word ends it silently, with no
    # handler and no answer.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=accepted, score=0.4, window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier,
                                   stt_text="включи свет", runs_store=store)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE, over_music=True)
    assert events == []  # held for the check
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert len(verifier.calls) == 1
    assert (StageEvent.STT_VAD_START, {} if accepted else {"duck": "False"}) in events
    assert StageEvent.TTS_START not in types_of(events)
    assert pipeline.play.calls == []
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["llm_text"]) == ("rejected", "wakeword_reject", "")
    # The record is that of a wake decided by its phrase: no score of the ducking check.
    assert (rec["wakeword_score"], rec["t_wakeword"]) == (None, 0)


# --- (b) reject -> silent abort, recorded ------------------------------------


async def test_gate_reject_aborts_silently_and_records(tmp_path, monkeypatch):
    # A rejecting verifier ends the wake untaken: NO events (a RUN_END would end the
    # session of an answer the device may be playing), NO STT/LLM/TTS, NO ack; the
    # device server ends the wake (end_wake). The run is recorded as
    # result='rejected', reason='wakeword_reject', and its audio is still stored.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    # STT spy: must NEVER be called on a reject.
    stt = FakeSttBackend("должно остаться нераспознанным")
    stt_calls = []
    orig = stt.transcribe

    async def spy(pcm):
        stt_calls.append(pcm)
        return await orig(pcm)

    stt.transcribe = spy
    verifier = FakeWakewordVerifier(accepted=False, score=0.05, window_ms=VERIFY_WINDOW_MS)
    hub = FakeRunEvents()
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, run_events=hub,
        stt_backend=stt, runs_store=store, ack=True,  # ack ON: prove it stays silent
    )
    # Spy on the ack scheduler so we can prove a reject never beeps.
    ack_calls = []
    pipeline._schedule_ack = lambda: ack_calls.append(1)

    pipeline.on_wake("cid", WAKE, over_answer=True)
    # 1 s on each mic channel, past the window -> verify scheduled.
    await pipeline.on_audio(b"\x01\x02" * 16000, b"\x03\x04" * 16000)
    await settle(pipeline)

    assert stt_calls == []                 # STT never ran
    assert ack_calls == []                 # no ack chime
    assert events == [("end_wake", "cid")]

    # Recorded as rejected, audio still stored.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_reject"
    assert rec["stt_text"] == ""
    assert rec["wakeword_score"] == 0.05
    assert rec["t_vad"] == 800             # the lead-in is trimmed (trim_start_ms 200)
    # Both mic channels are stored, trimmed like any run.
    [(run_id, wav, _keep)] = store.audio_calls
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getnframes()) == (2, 12800)
    # The run log panel sees the rejected row live.
    [msg] = hub.broadcasts
    assert (msg["run"]["id"], msg["run"]["result"], msg["run"]["has_audio"]) == (run_id, "rejected", 1)


async def test_gate_reject_records_audio_even_when_utterance_ended(tmp_path, monkeypatch):
    # Reject path where the utterance had already end-pointed while verifying: the
    # deferred finalize is irrelevant (reject claims the run itself); still recorded.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=False, score=0.0)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, runs_store=store,
    )
    set_small_vad_thresholds(pipeline)
    install_fake_vad(
        pipeline, [True, True, True, False, False, False, False, False, False]
    )

    pipeline.on_wake("cid", WAKE, over_answer=True)
    await pipeline.on_audio(FRAME * 9)
    await settle(pipeline)

    rec = store.records[-1]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_reject"


# --- (c) timeout / error -> fail policy --------------------------------------


async def test_gate_timeout_fail_open_stops_the_answer(tmp_path, monkeypatch):
    # A verify that exceeds the timeout with fail_open=True is treated as ACCEPT: the
    # wake stops the answer as a confirmed one does.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=20, fail_open=True,
                                    window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier, runs_store=store)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE, over_answer=True, over_playback=True)
    await pipeline.on_audio(PHRASE[:VERIFY_WINDOW_BYTES])
    await settle(pipeline)

    assert types_of(events)[-1] == StageEvent.RUN_END
    rec = store.records[-1]
    # Timeout -> no verdict -> score is None.
    assert (rec["result"], rec["reason"], rec["wakeword_score"]) == ("ok", "wake_stop", None)


async def test_gate_timeout_fail_closed_rejects_with_error(tmp_path, monkeypatch):
    # A verify that exceeds the timeout with fail_open=False is treated as REJECT,
    # recorded with reason='wakeword_error' (distinguishing it from a clean reject).
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=20, fail_open=False)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, runs_store=store,
    )

    pipeline.on_wake("cid", WAKE, over_answer=True)
    await pipeline.on_audio(b"\x01\x02" * 30000)
    await settle(pipeline)

    rec = store.records[-1]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_error"
    assert rec["wakeword_score"] is None
    assert events == [("end_wake", "cid")]


# --- (d) None / Passthrough backend -> identical to today --------------------


async def test_no_gate_when_backend_none(tmp_path, monkeypatch):
    # Default backend is None (warm-up not done): NO deferral — STT_VAD_START fires
    # in on_wake exactly as today, even for a real wake-word phrase.
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="захар включи свет")
    assert pipeline.rt.wakeword_backend is None
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE)
    # STT_VAD_START emitted immediately (no gate).
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
    ]
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert StageEvent.RUN_END in types_of(events)


async def test_no_gate_when_passthrough(tmp_path, monkeypatch):
    # A PassthroughVerifier (stage disabled) skips the gate entirely, the phrase's
    # text check included: a phrase without the wake word (a model of another word,
    # Hey Jarvis) still runs.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=PassthroughVerifier(),
        stt_text="эй джарвис включи свет", runs_store=store,
    )
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE)
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
    ]
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    rec = store.records[-1]
    # Ungated run: gate fields stay at their 0/None defaults.
    assert rec["t_wakeword"] == 0
    assert rec["wakeword_score"] is None
    assert (rec["result"], rec["llm_text"]) == ("ok", "готово")


async def test_no_gate_for_phraseless_start_with_real_verifier(tmp_path, monkeypatch):
    # A phraseless (manual button) start NEVER gates, even with a real verifier
    # loaded: on_wake confirms immediately.
    patch_llm(monkeypatch, reply="готово")
    verifier = FakeWakewordVerifier(accepted=False)  # would reject if it ran
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, stt_text="включи свет",
    )

    pipeline.on_wake("cid", None)  # no wake phrase
    assert StageEvent.STT_VAD_START in types_of(events)
    await pipeline.on_audio(b"\x01\x02" * 30000)
    assert verifier.calls == []  # verifier never engaged


# --- (e) a wake not over an answer: taken at once, confirmed by its phrase ----


async def test_a_wake_not_over_an_answer_is_taken_without_the_quick_check(tmp_path, monkeypatch):
    # Nothing of the device's own to tell it from: shown `listening` at the wake, and
    # the verifier (which would reject) is never asked.
    patch_llm(monkeypatch)
    verifier = FakeWakewordVerifier(accepted=False, window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier)
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", WAKE)
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START,
                                StageEvent.STT_VAD_START]
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    assert verifier.calls == []


def _make_text_checked(tmp_path, monkeypatch, text):
    """A pipeline with a loaded verifier whose STT hears `text`; the end-of-phrase
    chime lands in the events as ("ack", None)."""
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=FakeWakewordVerifier(),
                                   stt_text=text, runs_store=store)
    pipeline._schedule_ack = lambda: events.append(("ack", None))
    install_phrase_vad(pipeline)
    return pipeline, events, store


@pytest.mark.parametrize("text", [
    "захар включи свет",
    "Захаркоторый час?",          # fused with the next word
    "ну вот захар, включи свет",  # the third word
])
async def test_a_phrase_with_the_wake_word_runs_and_chimes_once_checked(tmp_path, monkeypatch,
                                                                        text):
    pipeline, events, store = _make_text_checked(tmp_path, monkeypatch, text)

    pipeline.on_wake("cid", WAKE)
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    # The chime comes once the wake word is found, not at the end of the phrase.
    assert types_of(events) == [
        StageEvent.RUN_START, StageEvent.STT_START, StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END, "ack", StageEvent.STT_END, StageEvent.INTENT_START,
        StageEvent.INTENT_END, StageEvent.TTS_START, StageEvent.RUN_END,
    ]
    rec = store.records[-1]
    assert (rec["result"], rec["stt_text"], rec["llm_text"]) == ("ok", text, "готово")


@pytest.mark.parametrize("text", [
    "заказ пришёл",
    "а где же наш захар",  # past the first three words
])
async def test_a_phrase_without_the_wake_word_ends_silently(tmp_path, monkeypatch, text):
    pipeline, events, store = _make_text_checked(tmp_path, monkeypatch, text)

    pipeline.on_wake("cid", WAKE)
    await pipeline.on_audio(PHRASE)
    await settle(pipeline)
    # No chime, no transcript, no handler, no answer: `transcribing`, then `idle`.
    assert events == [
        (StageEvent.RUN_START, {}), (StageEvent.STT_START, {}), (StageEvent.STT_VAD_START, {}),
        (StageEvent.STT_VAD_END, {}), (StageEvent.STT_END, {"text": ""}), (StageEvent.RUN_END, {}),
    ]
    assert pipeline.play.calls == []
    assert not (tmp_path / "context_dev.txt").exists()
    rec = store.records[-1]
    assert (rec["result"], rec["reason"], rec["stt_text"], rec["llm_text"]) == (
        "rejected", "wakeword_reject", text, "")
    assert [a["outcome"] for a in rec["attempts"]] == ["rejected"]


# --- disconnect cancels an in-flight verify ----------------------------------


async def test_on_disconnect_cancels_pending_verify(tmp_path, monkeypatch):
    # The speaker vanished mid-verify: on_disconnect cancels the verify task so a
    # late verdict can't act on the dead run.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    # Long-running; its timeout would accept (fail_open) after 5 s.
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=5000)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier, runs_store=store)

    pipeline.on_wake("cid", WAKE, over_answer=True)
    await pipeline.on_audio(b"\x01\x02" * 30000)  # schedules verify
    while not verifier.calls:
        await asyncio.sleep(0)

    await pipeline.on_disconnect()
    # The check ends at once, and the dead run is neither confirmed nor rejected.
    await asyncio.wait_for(settle(pipeline), 1)
    assert events == []
    assert store.records == []


@pytest.mark.parametrize("abandon", ["disconnect", "cancel"])
async def test_an_abandoned_run_takes_no_audio_after_a_rejected_wake(tmp_path, monkeypatch, abandon):
    # A run cut off mid-phrase (the device dropped off, or the run was cancelled)
    # never answers: the frames that still arrive after a later wake is rejected do
    # not complete its phrase and run the old command.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=True, window_ms=VERIFY_WINDOW_MS)
    pipeline, events = _make_gated(tmp_path, monkeypatch, verifier=verifier, runs_store=store)
    set_small_vad_thresholds(pipeline)
    # Three frames of speech, then silence; five silent frames end a phrase.
    install_fake_vad(pipeline, [True, True, True, False])

    pipeline.on_wake("s1", WAKE)
    await pipeline.on_audio(FRAME * 6)  # taken; three silent frames, no end yet
    await settle(pipeline)
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START,
                                StageEvent.STT_VAD_START]
    if abandon == "disconnect":
        await pipeline.on_disconnect()
    else:
        pipeline.cancel_run()
    del events[:]

    verifier._accepted = False
    pipeline.on_wake("s2", WAKE, over_answer=True)
    await pipeline.on_audio(FRAME * 6)  # rejected at its verify window
    await settle(pipeline)
    await pipeline.on_audio(FRAME * 6)  # the device has not seen the end of s2 yet
    await settle(pipeline)

    assert events == [("end_wake", "s2")]
    assert [rec["reason"] for rec in store.records] == ["wakeword_reject"]
