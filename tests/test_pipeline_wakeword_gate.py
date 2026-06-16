"""Pipeline-level tests for the server-side wake-word gate.

The gate is purely server-side: an ESP32 detects «Захар» locally and streams a
pre-roll + live audio; a second-stage verifier runs on the pre-roll head and
decides whether to PROCESS the rest of the pipeline. On accept the run proceeds
exactly as a normal wake-word run; on reject it aborts silently (no STT/LLM/TTS,
no ack chime), records the run as 'rejected', and emits RUN_END to stop the
device. These tests drive Pipeline directly with a controllable fake verifier
(verdict/score/timeout/fail_open) and the existing pipeline test doubles.

The verify runs in a background task scheduled (synchronously) from on_audio, so
each test awaits that task before asserting on the post-verdict state.
"""

import asyncio

import pytest

from src.pipeline_events import StageEvent
from src.wakeword import PassthroughVerifier, WakewordVerdict, WakewordVerifier

# Reuse the existing pipeline test harness (fakes + helpers) verbatim.
from tests.test_pipeline import (
    FRAME,
    FakeRunsStore,
    FakeSttBackend,
    install_fake_vad,
    make_pipeline,
    patch_llm,
    set_small_vad_thresholds,
    types_of,
)

# A real wake-word run carries a non-empty wake_word_phrase; a None phrase is a
# manual button start (never gated). The gate keys off this phrase.
WAKE = "Захар"


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

    async def verify(self, pcm: bytes) -> WakewordVerdict:
        self.calls.append(pcm)
        if self._hang:
            # Sleep past the pipeline's timeout so wait_for cancels us.
            await asyncio.sleep(self.timeout_ms / 1000 + 1.0)
        return WakewordVerdict(accepted=self._accepted, score=self._score)


async def _drain_verify(pipeline):
    """Await the in-flight wake-word verify task (and the finalize it may trigger)."""
    task = pipeline._wakeword_task
    if task is not None:
        await task


def _make_gated(tmp_path, monkeypatch, *, verifier, **kw):
    """Build a pipeline whose runtime carries `verifier` as the wakeword backend."""
    pipeline, events = make_pipeline(tmp_path, monkeypatch, **kw)
    pipeline.rt.wakeword_backend = verifier
    return pipeline, events


# --- (a) accept -> run proceeds normally ------------------------------------


async def test_gate_accept_proceeds_to_stt_and_records(tmp_path, monkeypatch):
    # An accepting verifier lets the run proceed EXACTLY as a normal wake-word run:
    # STT_VAD_START is emitted (deferred until confirm), STT/LLM/TTS run, and the
    # record carries the verify latency + score.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=True, score=0.91)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier,
        stt_text="включи свет", runs_store=store,
    )

    assert await pipeline.on_start("cid", 0, None, WAKE) == 0
    # Before the verdict: gate pending, STT_VAD_START NOT emitted, stream not opened.
    assert pipeline._wakeword_pending is True
    assert pipeline._wakeword_confirmed is False
    assert types_of(events) == [StageEvent.RUN_START, StageEvent.STT_START]

    # Feed enough PCM to cross the verify window (1500 ms = 48000 bytes); the gate
    # schedules verify on the first chunk that crosses it.
    await pipeline.on_audio(b"\x01\x02" * 30000)  # 120000 bytes > window
    assert pipeline._wakeword_task is not None
    await _drain_verify(pipeline)

    # Verdict accepted -> confirmed, STT_VAD_START now emitted, stream/STT continue.
    assert pipeline._wakeword_confirmed is True
    assert pipeline._wakeword_pending is False
    assert StageEvent.STT_VAD_START in types_of(events)

    # The verifier saw the pre-roll head sized to its window (48000 bytes).
    assert len(verifier.calls) == 1
    assert len(verifier.calls[0]) == verifier.window_ms * 16000 * 2 // 1000

    await pipeline.on_stop(False)
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END,
        StageEvent.STT_END,
        StageEvent.INTENT_START,
        StageEvent.INTENT_END,
        StageEvent.TTS_START,
        StageEvent.TTS_END,
        StageEvent.RUN_END,
    ]
    rec = store.records[-1]
    assert rec["result"] in ("ok", "tool")
    assert rec["stt_text"] == "включи свет"
    assert rec["t_wakeword"] >= 0
    assert rec["wakeword_score"] == 0.91


async def test_gate_accept_short_utterance_ends_during_verify(tmp_path, monkeypatch):
    # The utterance ends (VAD endpoint) BEFORE the verify window fills: the gate
    # fires verify on the whole short buffer, and confirm replays the deferred
    # finalize so the run still completes through STT.
    patch_llm(monkeypatch, reply="ок")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(accepted=True, score=0.8)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier,
        stt_text="свет", runs_store=store,
    )
    set_small_vad_thresholds(pipeline)
    # 3 speech frames then enough silence -> endpoint well before the 1500 ms window.
    install_fake_vad(
        pipeline, [True, True, True, False, False, False, False, False, False]
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    # One short chunk of several frames triggers the VAD endpoint while gating.
    await pipeline.on_audio(FRAME * 9)
    # The endpoint was deferred, not finalized, and verify was scheduled on the
    # short buffer (the window wasn't reached).
    assert pipeline._vad_finalize_pending == "endpoint"
    assert not pipeline._finalized
    assert pipeline._wakeword_task is not None
    await _drain_verify(pipeline)

    # Confirm replayed the deferred finalize -> the run completed through STT.
    assert pipeline._finalized is True
    assert StageEvent.RUN_END in types_of(events)
    assert StageEvent.STT_END in types_of(events)
    rec = store.records[-1]
    assert rec["reason"] == "endpoint"
    assert rec["stt_text"] == "свет"
    # verify saw the short buffer (whole buffer, < window).
    assert len(verifier.calls) == 1
    assert len(verifier.calls[0]) == len(FRAME) * 9


# --- (b) reject -> silent abort, recorded ------------------------------------


async def test_gate_reject_aborts_silently_and_records(tmp_path, monkeypatch):
    # A rejecting verifier aborts the run: NO STT/LLM/TTS, NO ack, NO STT_VAD_START,
    # NO STT_VAD_END; STT_END (empty) + RUN_END are emitted (RUN_END stops the
    # device). The run is recorded as result='rejected', reason='wakeword_reject',
    # and its audio is still stored.
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
    verifier = FakeWakewordVerifier(accepted=False, score=0.05)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier,
        stt_backend=stt, runs_store=store, ack=True,  # ack ON: prove it stays silent
    )
    # Spy on the ack scheduler so we can prove a reject never beeps.
    ack_calls = []
    pipeline._schedule_ack = lambda: ack_calls.append(1)
    pipeline.send_announcement = lambda **kw: None

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(b"\x01\x02" * 30000)  # > window -> verify scheduled
    await _drain_verify(pipeline)

    assert stt_calls == []                 # STT never ran
    assert ack_calls == []                 # no ack chime
    assert pipeline._finalized is True
    seq = types_of(events)
    assert seq == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]
    # No "thinking"/"listening" indicators on a reject.
    assert StageEvent.STT_VAD_START not in seq
    assert StageEvent.STT_VAD_END not in seq
    assert StageEvent.TTS_START not in seq
    data = dict(events)
    assert data[StageEvent.STT_END] == {"text": ""}

    # Recorded as rejected, audio still stored.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_reject"
    assert rec["stt_text"] == ""
    assert rec["wakeword_score"] == 0.05
    assert len(store.audio_calls) == 1     # utterance audio stored for the reject


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

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(FRAME * 9)
    await _drain_verify(pipeline)

    assert pipeline._finalized is True
    rec = store.records[-1]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_reject"


# --- (c) timeout / error -> fail policy --------------------------------------


async def test_gate_timeout_fail_open_proceeds(tmp_path, monkeypatch):
    # A verify that exceeds the timeout with fail_open=True is treated as ACCEPT:
    # the run proceeds normally through STT.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=20, fail_open=True)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier,
        stt_text="включи свет", runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(b"\x01\x02" * 30000)
    await _drain_verify(pipeline)

    assert pipeline._wakeword_confirmed is True
    await pipeline.on_stop(False)
    rec = store.records[-1]
    assert rec["result"] in ("ok", "tool")
    assert rec["stt_text"] == "включи свет"
    # Timeout -> no verdict -> score is None.
    assert rec["wakeword_score"] is None


async def test_gate_timeout_fail_closed_rejects_with_error(tmp_path, monkeypatch):
    # A verify that exceeds the timeout with fail_open=False is treated as REJECT,
    # recorded with reason='wakeword_error' (distinguishing it from a clean reject).
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=20, fail_open=False)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(b"\x01\x02" * 30000)
    await _drain_verify(pipeline)

    assert pipeline._finalized is True
    rec = store.records[-1]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_error"
    assert rec["wakeword_score"] is None
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]


# --- (d) None / Passthrough backend -> identical to today --------------------


async def test_no_gate_when_backend_none(tmp_path, monkeypatch):
    # Default backend is None (warm-up not done): NO deferral — STT_VAD_START fires
    # in on_start exactly as today, even for a real wake-word phrase.
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    assert pipeline.rt.wakeword_backend is None

    await pipeline.on_start("cid", 0, None, WAKE)
    assert pipeline._wakeword_confirmed is True
    assert pipeline._wakeword_pending is False
    # STT_VAD_START emitted immediately (no gate), no verify task scheduled.
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
    ]
    await pipeline.on_audio(b"\x01\x02" * 100)
    assert pipeline._wakeword_task is None
    await pipeline.on_stop(False)
    assert StageEvent.RUN_END in types_of(events)


async def test_no_gate_when_passthrough(tmp_path, monkeypatch):
    # A PassthroughVerifier (stage disabled) skips the gate entirely: behavior is
    # byte-for-byte the same as an ungated run.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=PassthroughVerifier(),
        stt_text="включи свет", runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    assert pipeline._wakeword_confirmed is True
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
    ]
    await pipeline.on_audio(b"\x01\x02" * 100)
    assert pipeline._wakeword_task is None
    await pipeline.on_stop(False)
    rec = store.records[-1]
    # Ungated run: gate fields stay at their 0/None defaults.
    assert rec["t_wakeword"] == 0
    assert rec["wakeword_score"] is None


async def test_no_gate_for_phraseless_start_with_real_verifier(tmp_path, monkeypatch):
    # A phraseless (manual button) start NEVER gates, even with a real verifier
    # loaded: on_start confirms immediately.
    patch_llm(monkeypatch, reply="готово")
    verifier = FakeWakewordVerifier(accepted=False)  # would reject if it ran
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, stt_text="включи свет",
    )

    await pipeline.on_start("cid", 0, None, None)  # no wake phrase
    assert pipeline._wakeword_confirmed is True
    assert pipeline._wakeword_pending is False
    assert StageEvent.STT_VAD_START in types_of(events)
    await pipeline.on_audio(b"\x01\x02" * 30000)
    assert pipeline._wakeword_task is None  # verifier never engaged
    assert verifier.calls == []


# --- streaming STT: backlog is fed once on confirm, then normally ------------


class _RecordingStreamSession:
    """Streaming STT session double recording every fed chunk + finish() result."""

    def __init__(self, text="команда"):
        self.fed = []
        self.closed = False
        self._text = text

    def feed(self, pcm):
        self.fed.append(pcm)

    async def finish(self):
        return self._text

    async def aclose(self):
        self.closed = True


class _RecordingStreamSttBackend:
    def __init__(self, text="команда"):
        self._text = text
        self.session = _RecordingStreamSession(text)

    async def transcribe(self, pcm):
        return self._text  # unused: the streaming path takes finish()

    def open_stream(self):
        return self.session


async def test_gate_accept_streaming_feeds_backlog_once(tmp_path, monkeypatch):
    # Streaming STT under the gate: the stream is NOT opened/fed while pending; on
    # confirm the WHOLE buffered backlog is fed exactly once, then later chunks feed
    # incrementally (no double-feed, no lost audio).
    patch_llm(monkeypatch, reply="готово")
    stt = _RecordingStreamSttBackend(text="включи свет")
    verifier = FakeWakewordVerifier(accepted=True, score=0.9, window_ms=1500)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, stt_backend=stt,
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    # No stream opened while gating.
    assert pipeline._stt_stream is None
    assert stt.session.fed == []

    backlog = b"\x01\x02" * 30000  # > 48000-byte window
    await pipeline.on_audio(backlog)
    await _drain_verify(pipeline)

    # On confirm the stream opened and the whole backlog was fed once.
    assert pipeline._stt_stream is stt.session
    assert stt.session.fed == [backlog]

    # A further chunk after confirm feeds incrementally (not the backlog again).
    tail = b"\x03\x04" * 10
    await pipeline.on_audio(tail)
    assert stt.session.fed == [backlog, tail]

    await pipeline.on_stop(False)
    assert StageEvent.STT_END in types_of(events)


# --- disconnect cancels an in-flight verify ----------------------------------


async def test_on_stop_while_pending_rejects_not_full_pipeline(tmp_path, monkeypatch):
    # A device stop that arrives while the verdict is still pending must NOT bypass
    # the gate into STT/LLM/TTS. It cancels the in-flight verify and routes the stop
    # through the reject path: result='rejected', NO STT/LLM/TTS, and NO STT_VAD_END
    # (which would have wrongly triggered the thinking indicator).
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    # STT spy: must NEVER be called when the stop short-circuits to a reject.
    stt = FakeSttBackend("должно остаться нераспознанным")
    stt_calls = []
    orig = stt.transcribe

    async def spy(pcm):
        stt_calls.append(pcm)
        return await orig(pcm)

    stt.transcribe = spy
    # Long-running verify so the gate is still pending when on_stop arrives.
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=5000, accepted=True)
    pipeline, events = _make_gated(
        tmp_path, monkeypatch, verifier=verifier, stt_backend=stt, runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(b"\x01\x02" * 30000)  # schedules verify
    task = pipeline._wakeword_task
    assert task is not None and not task.done()
    assert pipeline._wakeword_pending is True
    assert pipeline._wakeword_confirmed is False

    # Device stop BEFORE the verdict -> reject, not the full pipeline.
    await pipeline.on_stop(False)
    # The in-flight verify was cancelled.
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    assert stt_calls == []                 # STT never ran
    assert pipeline._finalized is True
    seq = types_of(events)
    # No STT_VAD_START/STT_VAD_END (no thinking indicator), only the reject sequence.
    assert seq == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]
    assert StageEvent.STT_VAD_END not in seq
    assert StageEvent.STT_VAD_START not in seq
    assert StageEvent.TTS_START not in seq

    rec = store.records[-1]
    assert rec["result"] == "rejected"
    assert rec["reason"] == "wakeword_reject"
    assert rec["stt_text"] == ""


async def test_on_disconnect_cancels_pending_verify(tmp_path, monkeypatch):
    # The speaker vanished mid-verify: on_disconnect cancels the verify task so a
    # late verdict can't act on the dead run.
    patch_llm(monkeypatch)
    verifier = FakeWakewordVerifier(hang=True, timeout_ms=5000)  # long-running
    pipeline, _events = _make_gated(tmp_path, monkeypatch, verifier=verifier)

    await pipeline.on_start("cid", 0, None, WAKE)
    await pipeline.on_audio(b"\x01\x02" * 30000)  # schedules verify
    task = pipeline._wakeword_task
    assert task is not None and not task.done()

    await pipeline.on_disconnect()
    # cancel() was requested; let the loop process it, then confirm the task ended
    # in cancellation so no late verdict acted on the dead run.
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    # The dead run was never confirmed/rejected by a late verdict.
    assert pipeline._wakeword_confirmed is False
    assert pipeline._finalized is False
