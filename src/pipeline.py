"""Per-device voice pipeline: mic PCM -> recognition chains (STT -> LLM) -> TTS -> device playback.

One Pipeline instance per device name. It owns its own audio buffer and lock; the
brain and TTS backend are stateless shared dependencies, and the device server
injects the transport hooks (send_event, play, has_speaker, has_follow_up).

Server-side VAD end-pointing
----------------------------
The device fires its wake word on-device and then streams mic PCM to us
continuously (silence included). It has no end-pointing of its own
(docs/device-protocol.md), so on_stop has no caller on the wire (the device's `stop`, the stop
word, cancels the run through cancel_run). To know when the user has
finished speaking we run the configured VAD stage backend over the incoming PCM
(one VadSession per run, see src/vad.py) and finalize the utterance once it
reports an endpoint (speech followed by enough trailing silence), a hard max
length, or a no-speech timeout. The explicit device-stop path still finalizes
too — both routes funnel through a synchronous _claim() (which atomically
marks the run finalized and snapshots the audio) followed by the async _run(), so a
run is finalized exactly once even under concurrent eager on_audio tasks.
"""

import asyncio
import os
import time

import numpy as np
from loguru import logger

from src import chains, config_store, context, llm
from src.audio_prep import (
    highpass,
    normalize_peak,
    pcm_to_wav_bytes,
    trim_start_pcm,
    vad_boost,
    write_wav,
)
from src.tts import TtsBackend, pcm_rate
from src.pipeline_events import StageEvent
from src.llm_text import clean_llm_output
from src.prompt import build_system_prompt
from src.runs_store import live_row, summary_row
from src.chime import build_ack_clip
from src.followup import NoFollowup
from src.stage_errors import StageError
from src.wakeword import PassthroughVerifier
# SAMPLE_RATE now lives in the VAD stage contract module; re-exported here so the
# many existing `from src.pipeline import SAMPLE_RATE` users keep working.
from src.vad import SAMPLE_RATE, EndpointPolicy  # noqa: F401  (SAMPLE_RATE re-export)

# Generous hard memory cap (~60s of 16 kHz / 16-bit mono PCM). VAD should end the
# utterance long before this; the cap only guards against unbounded growth if VAD
# misbehaves, and also forces a finalize when reached.
HARD_CAP_BYTES = SAMPLE_RATE * 2 * 60

# A wake session's stream opens with this much pre-roll, ending where the device
# detected the wake word (docs/device-protocol.md, "Voice session").
PREROLL_MS = 2000
PREROLL_BYTES = PREROLL_MS * SAMPLE_RATE * 2 // 1000

# Run reasons that end a wake run before its handler: the wake-word verifier rejected
# it (or timed out under fail-closed), or another device's wake of the same phrase won
# the cross-device arbitration (only a duplicate's phrase is still recognized, for the
# run log). Such a run is recorded as "rejected", without the ack chime or a capture
# file.
REJECT_REASONS = ("wakeword_reject", "wakeword_error", "duplicate")


class _PrefetchedChunks:
    """A TTS chunk stream whose first chunk was already pulled: iterating yields that
    chunk, then the rest of the stream. aclose() closes the underlying stream even
    before the first iteration, so the backend's open response is always released."""

    def __init__(self, first, rest):
        self._first = first
        self._rest = rest

    @classmethod
    async def start(cls, chunks):
        """Pull the first chunk of `chunks` (an empty stream yields nothing). A
        failure while pulling it closes the stream and propagates."""
        rest = chunks.__aiter__()
        try:
            first = await anext(rest, None)
        except BaseException:
            await Pipeline._close_stream(rest)
            raise
        return cls(first, rest)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._first is not None:
            chunk, self._first = self._first, None
            return chunk
        return await anext(self._rest)

    async def aclose(self):
        await Pipeline._close_stream(self._rest)


class Pipeline:
    """Drives one voice run for a single speaker."""

    def __init__(self, name, runtime):
        self.name = name
        # The mutable Runtime holder: all config and backends are read THROUGH it
        # per request (no frozen copies), so reconfiguration takes effect live.
        self.rt = runtime
        self._buffer = bytearray()
        # The other (non-selected) mic channel, buffered only for the stored
        # stereo diagnostic WAV — never fed to VAD/STT.
        self._buffer2 = bytearray()
        self._lock = asyncio.Lock()
        self._conversation_id = ""

        # Per-run VAD session, opened by _reset_run (a wake, a follow-up window) from the
        # swappable vad stage backend (self.rt.vad_backend). The session owns ALL
        # end-pointing state (framing, counters); the pipeline only feeds it
        # chunks and acts on the returned reason.
        self._vad_session = None
        # Live streaming STT sessions of the chains whose STT starts at speech, keyed
        # by chain name (only for streaming-capable backends).
        self._stt_streams: dict = {}

        # Server-side second-stage wake-word gate (see _run_wakeword_verify). While
        # the gate is pending we hold back STT_VAD_START + the STT stream and don't
        # finalize the utterance: a grammar-Vosk keyword spotter verifies the pre-roll
        # head and either confirms (proceed exactly as normal) or rejects (abort the
        # run silently — no STT/LLM/TTS, no ack — and stop the device). Inactive for
        # non-wake runs and a None/passthrough backend.
        self._wakeword_pending = False     # gate active, verdict not yet in
        self._wakeword_confirmed = False   # verdict accepted OR gate inactive -> normal streaming
        self._wakeword_task: asyncio.Task | None = None
        self._wakeword_t = 0               # verify latency (ms), into the run record
        self._wakeword_score: float | None = None
        # VAD endpoint reason seen WHILE the gate is still verifying: the utterance
        # ended before the verdict landed, so the finalize is deferred until confirm.
        self._vad_finalize_pending: str | None = None
        # Cross-device wake arbitration (src/wake_arbiter.py) holds a wake run the
        # same way as the gate until the arbiter's verdict (arbitration_verdict). The
        # token is bumped for every registered run, so a verdict meant for an older
        # run of this device is told apart and ignored.
        self._arbitration_pending = False
        self._arbitration_token = 0
        # The run lost the arbitration: it stays held, silently, until its VAD
        # end-points, and is then finalized as a "duplicate" (_finalize_duplicate).
        self._duplicate = False
        # Running utterance peak driving the optional decision-only VAD boost
        # (core.vad.mic_auto_gain); reset per run in on_start.
        self._vad_peak = 0
        # Identity of the VAD backend that opened this run's session, for the
        # finalize log line.
        self._vad_desc = ""
        self._finalized = False
        # Logging-only flag: log "receiving audio" once per run, not per chunk.
        self._audio_logged = False
        # Logging-only flag: warn once per run if mic.channel=1 but no 2nd channel.
        self._mic_fallback_logged = False

        # The finalized run (STT -> LLM -> TTS) started from on_audio; a new wake
        # cancels it (see on_start).
        self._run_task: asyncio.Task | None = None

        # The assistant's name for the follow-up check: the word of the session's wake.
        self._wake_word = ""
        # (name, recognized request, un-accented spoken reply) while a follow-up
        # window is open (see _arm_followup); the run of the next utterance takes it,
        # a wake clears it.
        self._followup: tuple[str, str, str] | None = None

        # Transport hooks, injected by the device server: send_event(event, data)
        # (sync), play(mime, bytes | async chunks) (returns {"outcome", "dry",
        # "gap_ms"} once the device reports the stream, see Connection.play),
        # has_speaker (a device without one gets no TTS at all) and has_follow_up
        # (the device reopens its microphone on a follow-up `listening`). arbiter is
        # the cross-device WakeArbiter
        # (src/wake_arbiter.py) a wake run registers with and reports to — its
        # pre-roll is in, its verifier ruled, it stopped waiting (sync calls); None
        # runs every wake unarbitrated.
        self.send_event = None
        self.play = None
        self.has_speaker = False
        self.has_follow_up = False
        self.arbiter = None
        # speak() calls not yet played out (see speak()).
        self._speaking = 0

        # Background "filler" play tasks (early "I'll go check it" lines). Kept in a
        # set so a still-playing filler isn't garbage-collected mid-flight; each task
        # removes itself on completion. Fire-and-forget — never awaited by the run.
        self._filler_tasks = set()

        # Background end-of-phrase "ack" chime tasks (the server-side «блям» played the
        # instant VAD end-points the utterance). Tracked in a set for the same GC reason
        # as the fillers — fire-and-forget, never awaited by the run so STT isn't delayed.
        self._ack_tasks = set()
        # Lazily-built cache of the ack clip: (source_key, mime, audio_bytes). The
        # generated chime is synthesized once; a configured sound_path is loaded and
        # re-loaded only when the path changes (the cache key is the resolved source).
        self._ack_clip: tuple[str, str, bytes] | None = None

        # Tail of the chained background tasks that broadcast live (in-progress) run
        # snapshots. Scheduling these off the run coroutine keeps a slow WS client from
        # backpressuring `self._lock`; `_run`'s outer finally awaits this tail (outside
        # the lock) so the finalized broadcast never overtakes a partial. None = idle.
        self._live_send_tail = None

    # Read-through convenience properties: external callers (the device server,
    # tests) and the run logic below all reach config/backends THROUGH the runtime,
    # so a reconfiguration (live field change or backend swap) takes effect without
    # rebuilding the pipeline.
    @staticmethod
    def _backend_desc(backend) -> str:
        """Human-readable 'provider/model' identity stamped by ConfigService.create();
        falls back to the class name for unstamped backends (tests/fakes)."""
        return getattr(backend, "backend_desc", None) or type(backend).__name__

    @property
    def core(self):
        return self.rt.core

    @property
    def wakeword_backend(self):
        return self.rt.wakeword_backend

    @property
    def stress_backend(self):
        return self.rt.stress_backend

    @property
    def hub(self):
        return self.rt.hub

    @property
    def runs_store(self):
        return self.rt.runs_store

    @property
    def prompt_store(self):
        return self.rt.prompt_store

    @property
    def run_events(self):
        return self.rt.run_events

    @property
    def _context_path(self):
        # Data dir is hardcoded in config_store; accessed as a module attribute (not a
        # bound copy) so tests can monkeypatch config_store.DATA_DIR.
        return os.path.join(config_store.DATA_DIR, f"context_{self.name}.txt")

    def _emit(self, event_type, data=None):
        """Emit a transport-neutral StageEvent with a flat dict[str, str] payload."""
        if self.send_event is not None:
            self.send_event(
                event_type, {str(k): str(v) for k, v in (data or {}).items()}
            )

    async def on_start(self, conversation_id, wake_word_phrase) -> None:
        """Handle a wake: cancel the older run, reset ALL per-run state, announce the run.

        Everything before the first await is synchronous, so the older run (its run
        task, a pending wake-word verify, its ack/filler plays) is cancelled before
        any of it can run again."""
        if any(t is not None and not t.done() for t in (self._run_task, self._wakeword_task)):
            logger.info(f"{self.name}: new wake cancels the running run")
        # Also ends a follow-up window the previous answer opened.
        self.cancel_run()
        self._wake_word = wake_word_phrase or ""
        await self._reset_run(
            conversation_id,
            preroll_ms=PREROLL_MS if wake_word_phrase else 0,
            no_speech_timeout_ms=self.core.vad.no_speech_timeout_ms,
        )
        logger.info(f"{self.name}: ▶️ run started (cid={conversation_id})")
        self._emit(StageEvent.RUN_START, {})
        self._emit(StageEvent.STT_START, {})
        # Server-side wake-word gate decision. Gate only a REAL wake-word run
        # (non-empty phrase), and only when a real verifier is
        # loaded — a None backend (warm-up not done) or a PassthroughVerifier (stage
        # disabled) skips the gate entirely so behavior is unchanged. RUN_START /
        # STT_START above fire for BOTH paths (they don't light the LED).
        wb = self.rt.wakeword_backend
        gate = (
            bool(wake_word_phrase)
            and wb is not None
            and not isinstance(wb, PassthroughVerifier)
        )
        if gate:
            # Hold the run: do NOT emit STT_VAD_START and do NOT open the STT stream
            # until the verdict lands. The VAD session opened above still tracks the
            # end-point; on_audio buffers + defers any finalize until confirm/reject.
            self._wakeword_pending = True
            self._wakeword_confirmed = False
            self._stt_streams = {}
            logger.info(f"{self.name}: 🛡️ wake-word gate pending (cid={conversation_id})")
        else:
            # Ungated (disabled/None/passthrough/non-wake): the verifier's part of
            # the hold is satisfied at once.
            self._wakeword_confirmed = True
        if wake_word_phrase and self.arbiter is not None:
            # Cross-device arbitration holds a wake run the same way until its
            # verdict (arbitration_verdict). Registering is the last synchronous
            # step: the arbiter may rule inside the call.
            self._arbitration_pending = True
            self._arbitration_token += 1
            self.arbiter.register(self, self._arbitration_token)
        elif not gate:
            # Speech is already in-stream (pre-roll captured the command), so the
            # device should show the active "listening for command" LED right away;
            # the matching STT_VAD_END (thinking) is emitted in _run once VAD
            # finalizes the utterance.
            self._emit(StageEvent.STT_VAD_START, {})
            self._open_stt_streams()

    async def _reset_run(self, conversation_id, *, preroll_ms: int,
                         no_speech_timeout_ms: int) -> None:
        """Reset ALL per-run state and open the run's VAD session: a wake (on_start)
        and a follow-up listen (_arm_followup) start a run the same way. Cancels
        nothing; synchronous up to closing the stale STT sessions."""
        # The streaming STT sessions of a run that was never finalized; a finalized
        # run owns (and closes) its own sessions.
        stale_streams, self._stt_streams = self._stt_streams, {}
        self._conversation_id = conversation_id or ""
        self._buffer.clear()
        self._buffer2.clear()
        self._wakeword_pending = False
        self._wakeword_confirmed = False
        self._wakeword_task = None
        self._wakeword_t = 0
        self._wakeword_score = None
        self._vad_finalize_pending = None
        self._duplicate = False
        self._vad_session = None
        self._vad_peak = 0
        self._finalized = False
        self._audio_logged = False
        self._mic_fallback_logged = False
        await self._close_stt_streams(stale_streams)
        # Open the per-run VAD session. The end-pointing POLICY (generic thresholds)
        # is read live off core.vad here, while the speech classifier itself is the
        # swappable vad stage backend. One session per run, so a hot-swapped backend
        # or a changed policy applies on the next run.
        vad_cfg = self.core.vad
        # Capture the identity from the SAME backend ref used to open the session,
        # so a hot-swap mid-run can't mislabel the finalize log line.
        vad_backend = self.rt.vad_backend
        self._vad_desc = self._backend_desc(vad_backend)
        self._vad_session = vad_backend.open(EndpointPolicy(
            silence_ms=vad_cfg.silence_ms,
            min_speech_ms=vad_cfg.min_speech_ms,
            max_utterance_ms=vad_cfg.max_utterance_ms,
            no_speech_timeout_ms=no_speech_timeout_ms,
            preroll_ms=preroll_ms,
            wake_pause_ms=vad_cfg.wake_pause_ms,
        ))

    async def _arm_followup(self, conversation_id, request: str, reply: str) -> None:
        """Keep the session open after its answer played out and listen again without
        a wake word (docs/device-protocol.md, "Follow-up"). The per-run reset of a
        wake, but it cancels nothing (it runs inside the answering run) and has no
        wake word: no pre-roll, no wake-word gate, and the VAD ends the run with
        no_speech when speech does not start within the stage's window. The next
        utterance's run takes the last turn for the follow-up check."""
        window_ms = self.rt.followup_backend.window_ms
        await self._reset_run(conversation_id, preroll_ms=0, no_speech_timeout_ms=window_ms)
        self._followup = (self._wake_word, request, reply)
        self._wakeword_confirmed = True
        logger.info(f"{self.name}: 👂 listening for a follow-up ({window_ms} ms)")
        self._emit(StageEvent.STT_VAD_START, {})
        self._open_stt_streams()

    def cancel_run(self) -> None:
        """Cancel the running run: its run task, the wake-word verify (and a run it
        started) and its ack/filler plays, and end an open follow-up window: audio
        that still arrives is dropped. Synchronous; the cancelled run emits no
        RUN_END (its stages catch Exception, which a cancellation is not). The run
        stops waiting for the arbitration, so the arbiter skips it."""
        # Cancellation of an already-finished task is a no-op. Not awaited: each
        # cancelled task unwinds on its own (the ack/filler sets self-clean via their
        # done-callbacks); snapshot the sets first since those callbacks mutate them.
        for task in (self._run_task, self._wakeword_task,
                     *self._ack_tasks, *self._filler_tasks):
            if task is not None:
                task.cancel()
        self._run_task = None
        if self._followup is not None:
            self._followup = None
            self._finalized = True
        self._leave_arbitration()

    def _leave_arbitration(self) -> None:
        """Stop waiting for the arbitration verdict (the run was cancelled or its
        device vanished, which ends a pending verifier gate too): the arbiter
        re-checks the group without this run."""
        if self._arbitration_pending:
            self._arbitration_pending = False
            self.arbiter.run_changed(self, self._arbitration_token)

    def _open_stt_streams(self) -> None:
        """Open the live streaming STT sessions for this run (synchronous): one per
        chain whose STT starts at speech (chains.starts_at_speech), so on_audio can
        feed them chunks while speech is still arriving (VAD still owns end-pointing;
        on finalize the chain runner calls finish()).

        A batch-only backend (no open_stream(), or one returning None) gets no
        session — NOT an error — and is transcribed in one shot at VAD end; only a
        genuine open failure warns and falls back to that batch route. Shared by
        on_start's ungated path, _wakeword_confirm and _arm_followup so the open
        logic stays identical.
        """
        self._stt_streams = {}
        for chain in chains.starts_at_speech(self.rt, self.core.chains):
            try:
                stream = chains.open_stream(self.rt, chain)
            except Exception as e:  # noqa: BLE001 - never let STT setup break the run; batch-fallback
                logger.warning(
                    f"{self.name}: streaming STT open failed for chain {chain.name!r}, "
                    f"using batch: {e}"
                )
                continue
            if stream is not None:
                self._stt_streams[chain.name] = stream

    async def on_audio(self, data: bytes, data2=None) -> None:
        """Accumulate mic PCM and run VAD end-pointing on the configured channel.

        The Voice PE streams two mic channels: channel 0 (`data`) is the more-
        processed XMOS AGC output, channel 1 (`data2`) the less-processed XMOS
        noise-suppression output (cleaner but quieter). Only the channel is selected
        HERE, per chunk, by `core.vad.mic_channel` (read live off `self.core`, so panel
        changes apply on the next utterance — no restart). The optional high-pass +
        peak-normalization conditioning (`core.vad.mic_highpass` / `mic_normalize`) is
        applied LATER, once to the WHOLE utterance, in `_run` before STT — never per
        chunk (per-chunk normalization would destroy dynamics and amplify silence).
        The NON-selected channel is buffered separately (`_buffer2`) only so the
        stored diagnostic WAV can carry both channels in stereo — it never reaches
        VAD or STT.

        The device streams continuously and never signals end-of-speech, so we
        detect it here: VAD over the PCM finalizes the utterance once speech is
        followed by enough trailing silence (or a max length / no-speech timeout).

        Never suspends: the whole update is synchronous and a finalized run is
        started as its own task (_run_task), so the device server's read loop can
        await this per frame without ever waiting on a run.
        """
        if self._finalized:
            return  # Ignore late audio for an already-finalized run.

        # Select the mic channel for the WHOLE pipeline (VAD + STT) from
        # config (0 = processed, 1 = less-processed/quieter). Channel 1 uses the
        # device's second stream when present; if it's missing, fall back to channel 0
        # and warn once per run. Read live off self.core, so panel changes take effect
        # on the next utterance. Any high-pass / normalize conditioning happens later,
        # once on the whole utterance in _run (never per chunk).
        other = None  # the non-selected channel, kept only for the stored diagnostic WAV
        if self.core.vad.mic_channel == 1:
            if data2:
                data, other = data2, data
            elif not self._mic_fallback_logged:
                self._mic_fallback_logged = True
                logger.debug(
                    f"{self.name}: mic_channel=1 but device sent no second channel; "
                    f"using channel 0"
                )
        else:
            other = data2  # may be None when the device streams a single channel

        # Held: the wake-word verifier or the cross-device arbitration hasn't ruled
        # yet, or the run lost the arbitration. Keep accumulating the buffer
        # (channel-select below feeds _buffer/_buffer2) but DON'T feed the STT streams
        # (there are none yet) and DON'T finalize the utterance — if VAD end-points,
        # stash the reason and stop feeding VAD; the finalize is deferred until the
        # hold is lifted (_release_hold replays it). A lost run is never lifted: its
        # end-point finalizes it as a duplicate at once. While the verifier gate is
        # active, trigger the verify task EXACTLY once, when enough pre-roll is
        # buffered (window_bytes) or the utterance already ended. While the
        # arbitration is pending, tell the arbiter once when the whole pre-roll is in.
        # This branch is fully SYNCHRONOUS (every await lives inside the task) so the
        # eager-on_audio atomicity invariant holds.
        if ((self._wakeword_pending and not self._wakeword_confirmed)
                or self._arbitration_pending or self._duplicate):
            if not self._audio_logged:
                self._audio_logged = True
                logger.info(f"{self.name}: 🎤 receiving audio (held)...")
            buffered = len(self._buffer)
            self._buffer.extend(data)
            if other:
                self._buffer2.extend(other)
            elif self._buffer2:
                self._buffer2.extend(b"\x00" * len(data))
            if len(self._buffer2) >= HARD_CAP_BYTES:
                del self._buffer2[HARD_CAP_BYTES:]
            if len(self._buffer) >= HARD_CAP_BYTES:
                del self._buffer[HARD_CAP_BYTES:]
            # Track the end-point while gating, but DON'T finalize here: stash the
            # reason and stop feeding VAD (the utterance end is replayed on confirm).
            if self._vad_finalize_pending is None and self._vad_session is not None:
                reason = self._vad_session.feed(data)
                if reason is not None:
                    self._vad_finalize_pending = reason
            if self._duplicate and self._vad_finalize_pending is not None:
                self._finalize_duplicate()
                return
            wb = self.rt.wakeword_backend
            if self._wakeword_pending and self._wakeword_task is None and wb is not None:
                window_bytes = wb.window_ms * SAMPLE_RATE * 2 // 1000
                if len(self._buffer) >= window_bytes or self._vad_finalize_pending is not None:
                    # Snapshot the pre-roll head SYNCHRONOUSLY; the task then awaits.
                    pcm = bytes(self._buffer[:window_bytes])
                    self._wakeword_task = asyncio.create_task(
                        self._run_wakeword_verify(pcm)
                    )
            # Last: the buffer just reached the whole pre-roll. The arbiter may rule
            # inside the call, on this run too, so nothing follows it.
            if self._arbitration_pending and buffered < PREROLL_BYTES <= len(self._buffer):
                self.arbiter.preroll_in(self, self._arbitration_token)
            return

        # Log once per run on the first chunk; the speaker streams many chunks.
        if not self._audio_logged:
            self._audio_logged = True
            logger.info(f"{self.name}: 🎤 receiving audio...")

        # The full utterance audio (everything streamed, silence included) is what
        # we send to STT.
        self._buffer.extend(data)
        # Feed the live streaming STT sessions (if any) the same selected-channel bytes
        # that go into the buffer. feed() is synchronous and never raises.
        for stream in self._stt_streams.values():
            stream.feed(data)
        if other:
            self._buffer2.extend(other)
        elif self._buffer2:
            # Keep channels time-aligned if the second stream drops out mid-run.
            self._buffer2.extend(b"\x00" * len(data))
        if len(self._buffer2) >= HARD_CAP_BYTES:
            del self._buffer2[HARD_CAP_BYTES:]
        reason = None
        if len(self._buffer) >= HARD_CAP_BYTES:
            del self._buffer[HARD_CAP_BYTES:]
            reason = "maxlen"
        elif self._vad_session is not None:
            # Feed the SELECTED channel — the same bytes that went into the buffer —
            # to the per-run VAD session. The session does its own framing/counters
            # and returns a finalize reason ("endpoint" | "maxlen" | "no_speech") or
            # None to keep listening. feed() is SYNCHRONOUS, so the claim logic's
            # atomicity guarantees hold (no await before _claim below). When HARD_CAP
            # already set reason="maxlen" above, the session is NOT fed: we're about
            # to finalize this chunk anyway, and the buffer was truncated, so there's
            # no point feeding (possibly truncated) bytes into the VAD counters.
            vdata = data
            if self.core.vad.mic_auto_gain:
                # Track the running utterance peak and lift the chunk for the
                # speech/no-speech decision only — the buffer/STT path above keeps
                # the raw bytes (per-utterance normalization happens pre-STT in _run).
                m = len(data) - (len(data) % 2)
                if m:
                    self._vad_peak = max(
                        self._vad_peak,
                        int(np.abs(np.frombuffer(data[:m], dtype="<i2")).max()),
                    )
                vdata = vad_boost(data, self._vad_peak)
            reason = self._vad_session.feed(vdata)

        if reason is not None:
            state = self._vad_session.debug_state() if self._vad_session else {}
            logger.debug(
                f"{self.name}: VAD finalize reason={reason} peak={self._vad_peak} "
                + " ".join(f"{k}={v}" for k, v in state.items())
            )

        # All synchronous state is updated above; claim last. _claim() is atomic (no
        # await), so a run is finalized exactly once and no audio is lost.
        if reason is not None:
            claimed = self._claim()
            if claimed is not None:
                # Snapshot the conversation id synchronously with the claim so a
                # re-triggered wake word can't relabel this run's events.
                self._run_task = asyncio.create_task(
                    self._run(reason, claimed[0], self._conversation_id, pcm2=claimed[1])
                )

    def _claim(self) -> tuple[bytes, bytes] | None:
        """Atomically claim this run for finalization and snapshot the audio.

        Runs with no await, so concurrent eager on_audio tasks can't double-claim
        or lose audio. Returns (pcm, pcm2) — the selected-channel PCM to process
        and the other-channel PCM kept for the stored stereo diagnostic WAV
        (empty when the device streamed a single channel) — or None if already
        finalized.
        """
        if self._finalized:
            return None
        self._finalized = True
        pcm = bytes(self._buffer)
        pcm2 = bytes(self._buffer2)
        self._buffer.clear()
        self._buffer2.clear()
        return pcm, pcm2

    async def _run_wakeword_verify(self, pcm: bytes) -> None:
        """Run the second-stage wake-word verifier on the pre-roll head, then gate.

        Spawned exactly once per gated run (from on_audio). All the awaits live here,
        so the on_audio gating branch that schedules this stays synchronous and the
        eager-on_audio atomicity invariant holds. verify() never raises, but we still
        bound it with the backend's timeout and treat a timeout/None as the configured
        fail policy (fail_open accepts, fail_closed rejects). Decides confirm vs reject.
        """
        t0 = time.perf_counter()
        wb = self.rt.wakeword_backend
        if wb is None or self._finalized:
            return  # backend vanished, or the run is already gone.
        try:
            verdict = await asyncio.wait_for(wb.verify(pcm), timeout=wb.timeout_ms / 1000)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - timeout/error -> fail policy
            verdict = None
        self._wakeword_t = int((time.perf_counter() - t0) * 1000)
        self._wakeword_score = verdict.score if verdict is not None else None
        # A timeout / internal error has no verdict -> apply the fail policy.
        accepted = verdict.accepted if verdict is not None else wb.fail_open
        if verdict is None:
            logger.warning(
                f"{self.name}: 🛡️ wake-word verify timed out/errored "
                f"(fail_open={wb.fail_open}) -> {'accept' if accepted else 'reject'} "
                f"({self._wakeword_t} ms)"
            )
        elif accepted:
            logger.info(
                f"{self.name}: ✅ wake word confirmed (score={self._wakeword_score}, "
                f"{self._wakeword_t} ms)"
            )
        else:
            logger.info(
                f"{self.name}: 🚫 wake word rejected (score={self._wakeword_score}, "
                f"{self._wakeword_t} ms)"
            )
        if self._finalized:
            return  # the run was finalized while we were verifying.
        if accepted:
            await self._wakeword_confirm()
        else:
            await self._wakeword_reject(error=(verdict is None))

    async def _wakeword_confirm(self) -> None:
        """Accept verdict: the verifier lets the run through.

        Synchronously flips the gate. A run that already lost the arbitration stays
        held until its end-point. While the cross-device arbitration is still
        pending, the arbiter is told the verifier has ruled — it may rule inside the
        call, on this run too — and arbitration_verdict releases the run. Otherwise
        the hold is lifted here (_release_hold) and, if the utterance already ended
        while we were verifying, its finalize is awaited now; otherwise subsequent
        on_audio drives the normal VAD finalize.
        """
        if self._finalized:
            return
        self._wakeword_pending = False
        self._wakeword_confirmed = True
        if self._duplicate:
            return
        if self._arbitration_pending:
            self.arbiter.run_changed(self, self._arbitration_token)
            return
        finalize = self._release_hold()
        if finalize is not None:
            await finalize

    def _release_hold(self):
        """Lift the hold on a wake run once both the verifier and the arbitration let
        it through; shared by _wakeword_confirm and arbitration_verdict.

        Synchronously emits the held-back STT_VAD_START, opens the STT streams and
        replays the whole buffered backlog into each once — no await between the
        caller clearing its pending flag, opening, and feeding, so a concurrent
        on_audio can't double-feed or lose the chunk. If the utterance already ended
        during the hold (_vad_finalize_pending), the run is claimed here as well and
        the un-awaited coroutine of its _run is returned for the caller to await or
        schedule; otherwise None.
        """
        self._emit(StageEvent.STT_VAD_START, {})
        self._open_stt_streams()
        # Feed the whole backlog buffered during the hold once, so streaming STT sees
        # the same audio the batch path would.
        backlog = bytes(self._buffer)
        for stream in self._stt_streams.values():
            stream.feed(backlog)
        if self._vad_finalize_pending is None:
            return None
        claimed = self._claim()
        if claimed is None:
            return None
        return self._run(
            self._vad_finalize_pending, claimed[0], self._conversation_id, pcm2=claimed[1]
        )

    @property
    def preroll_pcm(self) -> bytes:
        """The pre-roll of the current wake run received so far (the selected mic
        channel), for the arbiter's signal-to-noise comparison."""
        return bytes(self._buffer[:PREROLL_BYTES])

    @property
    def verifier_pending(self) -> bool:
        """Whether the wake-word verifier has yet to rule on this run; the arbiter
        rules on a run only after its verifier."""
        return self._wakeword_pending

    def awaits_arbitration(self, token: int) -> bool:
        """Whether this run still waits for the arbitration verdict registered under
        `token`: not an older run's token, not finalized, still held for it."""
        return (
            token == self._arbitration_token
            and self._arbitration_pending
            and not self._finalized
        )

    def arbitration_verdict(self, token: int, won: bool) -> None:
        """The cross-device arbiter's verdict on the run registered under `token`.

        Synchronous, and possibly called inside a call this or another pipeline made
        to the arbiter, or from one of its timers — the module docstring of
        src/wake_arbiter.py lists them. A verdict the run no longer awaits is ignored.
        Lost: the run stays held — no STT_VAD_START, so the device gets no
        `listening` and streams the rest of the phrase on silently — and on_audio
        finalizes it as a duplicate at its VAD end-point, or it is finalized at once
        when the end-point was already stashed. Won: the arbitration hold is lifted
        and the run released, unless the wake-word verifier is still pending — its
        _wakeword_confirm then releases it.
        """
        if not self.awaits_arbitration(token):
            return
        if not won:
            self._arbitration_pending = False
            self._duplicate = True
            if self._vad_finalize_pending is not None:
                self._finalize_duplicate()
            return
        self._arbitration_pending = False
        if self._wakeword_pending:
            return
        finalize = self._release_hold()
        if finalize is not None:
            self._run_task = asyncio.create_task(finalize)

    def _finalize_duplicate(self) -> None:
        """Claim the lost run and start its _run as a duplicate (as _run_task, like
        on_audio's finalize): its phrase is recognized for the run log only."""
        claimed = self._claim()
        if claimed is not None:
            # A phrase the VAD found no speech in is not transcribed, as for any run:
            # the STT would make up words for the noise.
            self._run_task = asyncio.create_task(self._run(
                "duplicate", claimed[0], self._conversation_id, pcm2=claimed[1],
                transcribe=self._vad_finalize_pending != "no_speech",
            ))

    async def _wakeword_reject(self, error: bool) -> None:
        """Reject verdict: abort the run silently and stop the device.

        Claims the run (so a concurrent device-stop / VAD finalize can't also run it)
        and funnels through _run with a reject reason. _run records the run as
        'rejected', skips STT/LLM/TTS and the ack chime, and emits RUN_END — which the
        device server sends as `state idle`, so the device stops streaming the mic and
        ends its session. `error` distinguishes a timeout/internal error (wakeword_error)
        from a genuine non-match (wakeword_reject).
        """
        if self._finalized:
            return
        reason = "wakeword_error" if error else "wakeword_reject"
        claimed = self._claim()
        if claimed is None:
            return  # already finalized by a concurrent path.
        if self._arbitration_pending:
            # The rejected run drops out of its arbitration group (awaits_arbitration
            # is now False); the arbiter may rule on the other candidates.
            self.arbiter.run_changed(self, self._arbitration_token)
        await self._run(reason, claimed[0], self._conversation_id, pcm2=claimed[1])

    def _emit_live(self, record) -> None:
        """Schedule a best-effort broadcast of the current in-progress run snapshot.

        The snapshot is built SYNCHRONOUSLY here (so a later stage mutating `record`
        can't alter an already-scheduled message), then sent from a background task
        chained after this pipeline's previous live send. Scheduling instead of
        awaiting keeps a slow/stuck WS client from backpressuring the per-device
        `self._lock` that `_run` holds across the whole run — the same reason the
        finalized broadcast is deferred outside the lock. Chaining preserves per-stage
        delivery order; `_run`'s outer `finally` drains the chain (outside the lock)
        before the finalized broadcast so the final never overtakes a partial.
        """
        if self.run_events is None:
            return
        row = live_row(record)  # snapshot now; `record` keeps mutating across stages
        prev = self._live_send_tail

        async def _send():
            # Preserve order: wait for the previous live send first. `prev` swallows
            # its own errors, so awaiting it never raises here.
            if prev is not None:
                try:
                    await prev
                except Exception:  # noqa: BLE001 - defensive; _send never re-raises
                    pass
            try:
                await self.run_events.broadcast({"type": "run", "run": row})
            except Exception as e:  # noqa: BLE001 - live updates must never break a run
                logger.debug(f"{self.name}: live run update failed: {e}")

        self._live_send_tail = asyncio.create_task(_send())

    async def _close_stt_streams(self, streams: dict) -> None:
        """Close the given streaming STT sessions (keyed by chain name); a failure is
        logged, never raised."""
        for stream in streams.values():
            try:
                await stream.aclose()
            except Exception as e:  # noqa: BLE001 - cleanup must never break the run
                logger.debug(f"{self.name}: streaming STT close failed: {e}")

    def _llm_request(self, text: str) -> llm.LlmRequest:
        """A handler's input for `text`: the assembled system prompt (a SQLite read
        of the active profile), the context history and the originating device."""
        return llm.LlmRequest(
            system_prompt=build_system_prompt(self.core, self.prompt_store),
            history=context.load_context(
                self._context_path,
                max_turns=self.core.context.max_turns,
                ttl_seconds=self.core.context.ttl_seconds,
            ),
            user_text=text,
            device=self.name,
        )

    @staticmethod
    def _record_chain_run(record: dict, run) -> None:
        """Fill the run record from what the recognition chains produced."""
        record["chain"] = run.chain
        record["attempts"] = run.attempts
        record["t_stt"] = run.t_stt
        record["t_llm"] = run.t_llm
        record["stt_text"] = run.text
        if run.reply is not None:
            record["llm_text"] = run.reply
        if run.result is not None:
            record["model"] = run.result.model
            record["tokens"] = run.result.tokens
            record["rounds"] = run.result.rounds
            record["request"] = run.result.request_debug
        else:
            # A failed handler attaches what it accumulated before the failure.
            record["model"] = run.partial.get("model")
            record["tokens"] = run.partial.get("tokens")
            record["rounds"] = run.partial.get("rounds") or []
            record["request"] = run.partial.get("request")
        if run.answered:
            # Did the handler actually run any tool this run?
            record["result"] = "tool" if run.result.tool_used else "ok"
        elif run.error is not None:
            record["result"] = "error"
            record["error_stage"] = run.error.stage.upper()
            record["error_text"] = str(run.error)
        elif run.reply is not None:
            # Every handler that heard the request declined: its reply is the answer.
            record["result"] = "ok"

    async def on_disconnect(self) -> None:
        """Device dropped its connection. If streaming STT sessions are open for a
        run that was never finalized (the speaker vanished mid-utterance), close them
        so the gRPC streams/channels don't linger until the next run. A FINALIZED run
        owns the sessions and closes them in _run's finally, so skip it here to avoid
        racing finish() against aclose()."""
        # Best-effort cancel an in-flight wake-word verify for the vanished speaker so
        # a late verdict can't act on (confirm/reject) a dead run. No-op if already done.
        if self._wakeword_task is not None:
            self._wakeword_task.cancel()
        # A vanished device no longer waits for the arbitration: the arbiter skips it.
        self._leave_arbitration()
        if not self._finalized:
            streams, self._stt_streams = self._stt_streams, {}
            await self._close_stt_streams(streams)

    async def _run(self, reason, pcm, conversation_id, pcm2: bytes = b"",
                   transcribe: bool = True) -> None:
        """Run the recognition chains -> TTS -> events on the already-claimed audio, once.

        The caller claims the run via _claim() (which sets _finalized, snapshots
        and clears the buffer) and passes the snapshotted pcm here. The lock only
        provides defensive serialization; _claim() already guarantees single entry.
        RUN_END is always sent, unless the played answer opens a follow-up window
        (_arm_followup). The run takes over the streaming STT sessions of the
        claimed utterance, so a cancelled run's cleanup never touches the next one.
        `transcribe` is read for a duplicate only: whether its phrase is recognized for
        the run log.
        """
        stt_streams, self._stt_streams = self._stt_streams, {}
        # The last turn when this utterance was heard in a follow-up window, else
        # None; taken with the claimed utterance, like its STT sessions.
        followup, self._followup = self._followup, None
        pending_run = None
        try:
            async with self._lock:
                # Total-run timer (logging only); started after the claim/lock.
                t0 = time.perf_counter()
                logger.info(
                    f"{self.name}: 🎙️ captured {len(pcm)} bytes "
                    f"(~{len(pcm) / (SAMPLE_RATE * 2):.1f}s), reason={reason}, "
                    f"vad={self._vad_desc}"
                )
                if not pcm:
                    # Truly-empty audio: nothing to transcribe and nothing to record.
                    logger.info(f"{self.name}: empty audio, ending run")
                    await self._close_stt_streams(stt_streams)
                    self._emit(StageEvent.RUN_END, {})
                    return

                # End-of-phrase ack ("блям"): we've just finalized the utterance, so play
                # a short confirmation chime to the speaker NOW as immediate "got it"
                # feedback, before STT/LLM/TTS run. Fire-and-forget (scheduled, never
                # awaited) so it never delays STT; tracked in a set so it isn't GC'd
                # mid-playback. The ack, a filler and the reply are separate play()
                # streams, queued in call order on the device.
                #
                # A rejected wake (verifier, duplicate) aborts the run silently — no
                # ack chime — so a mis-trigger or a second device never beeps. A
                # follow-up phrase gets its chime only once the follow-up check accepts
                # it (_followup_gate).
                if reason not in REJECT_REASONS and followup is None:
                    self._schedule_ack()

                # Trim the configured lead-in (wake-word tail / button-press click) off the
                # start of the captured sample. The trim is applied ONCE here, so every
                # downstream consumer in this run uses the trimmed sample: STT, the t_vad
                # metric, the capture-session WAV and the stored diagnostic audio. Read live
                # off core.vad so it hot-applies. trim_start_pcm never returns empty audio,
                # so the non-empty guard above still holds afterwards. A follow-up phrase
                # has no lead-in: its audio starts at the `listening`, before the phrase.
                trim_ms = self.core.vad.trim_start_ms
                if trim_ms > 0 and followup is None:
                    trimmed = trim_start_pcm(pcm, trim_ms)
                    if len(trimmed) != len(pcm):
                        logger.info(
                            f"{self.name}: ✂️ trimmed {len(pcm) - len(trimmed)} bytes "
                            f"(~{trim_ms} ms) off sample start"
                        )
                    pcm = trimmed
                    # Trim the other (non-selected) channel by the same amount so
                    # the two channels of the stored stereo WAV stay time-aligned.
                    if pcm2:
                        pcm2 = trim_start_pcm(pcm2, trim_ms)

                # Optional pre-STT conditioning of the FULL utterance (read live off
                # core.vad, so the toggles hot-apply). High-pass first to strip
                # DC/rumble, then peak-normalize so the quieter less-processed channel
                # reaches a consistent level without clipping. Applied ONCE to the whole
                # sample (never per chunk), so the capture WAV, the stored diagnostic
                # audio and STT all get the exact audio Whisper sees.
                # `pcm2` (the other channel) is deliberately kept raw here, so the
                # stored stereo WAV allows comparing conditioned vs raw channels.
                # NOTE: mic_normalize gates ONLY this pre-STT normalization; the
                # decision-only VAD boost applied per chunk in on_audio is gated by
                # `core.vad.mic_auto_gain`.
                if self.core.vad.mic_highpass:
                    pcm = highpass(pcm)
                if self.core.vad.mic_normalize:
                    pcm = normalize_peak(pcm)

                # Optional raw audio capture: save the finalized utterance PCM (already
                # trimmed by core.vad.trim_start_ms above) as a 16 kHz / mono / 16-bit WAV.
                # Off by default; enabled per capture session. A capture failure must NEVER
                # break the run, so it is fully wrapped. A follow-up window that heard no
                # speech is not an utterance and leaves nothing, like in the run log.
                silent_followup = followup is not None and reason == "no_speech"
                if (self.core.capture.enabled and not silent_followup
                        and reason not in REJECT_REASONS):
                    try:
                        capture_dir = self.core.capture.dir
                        safe_name = "".join(
                            c if c.isascii() and (c.isalnum() or c in "._-") else "_"
                            for c in self.name
                        )
                        fname = f"{safe_name}_{time.time_ns()}_{reason}.wav"
                        out_path = os.path.join(capture_dir, fname)

                        def _capture(d=capture_dir, p=out_path, data=pcm):
                            os.makedirs(d, exist_ok=True)
                            write_wav(p, data)

                        await asyncio.to_thread(_capture)
                        logger.info(
                            f"{self.name}: 💾 captured raw audio -> {out_path} "
                            f"({len(pcm)} bytes)"
                        )
                    except Exception as e:
                        logger.error(f"{self.name}: raw audio capture failed: {e}")

                # Observability record for this run. Accumulated across all terminal
                # paths below and inserted exactly once in the finally; the rest of
                # the timings fill in as stages complete.
                # t_vad is the captured-utterance duration in ms (the "VAD capture"
                # segment shown in the UI waterfall), derived from the PCM length —
                # NOT the CPU time spent running the VAD.
                # audio_ms is intentionally left None for now: the reply's playback
                # duration is not measured yet.
                # Native TTS reply audio captured during synthesis (see the TTS section) and
                # stored after the run is inserted, mirroring the utterance-audio store. Stays
                # None unless synthesis produced bytes, so a failed/empty TTS stores nothing.
                tts_audio = None
                tts_mime = None
                # False only for a follow-up window that heard no speech: no request,
                # so no run in the log.
                keep_record = True
                record = {
                    "ts": time.time(),
                    "device": self.name,
                    "reason": reason,
                    "result": "empty",
                    "t_vad": int(len(pcm) / (SAMPLE_RATE * 2) * 1000),
                    "t_stt": 0, "t_llm": 0, "t_stress": 0, "t_tts": 0,
                    "stt_text": "", "llm_text": "", "stress_text": "",
                    "filler_text": "", "t_filler": None,
                    "model": None, "tokens": None,
                    "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
                    "error_stage": None, "error_text": None,
                    "rounds": [],
                    "request": None,
                    # Recognition chains: the one that served the request and every
                    # chain tried (see chains.ChainRun).
                    "chain": None,
                    "attempts": [],
                    # Second-stage wake-word gate: verify latency + score. Stay 0/None
                    # for ungated (non-wake / disabled / passthrough) runs.
                    "t_wakeword": self._wakeword_t,
                    "wakeword_score": self._wakeword_score,
                    # The reply's playback as the device reports it: times it ran dry
                    # and the ms of silence played meanwhile. None when it does not count.
                    "play_dry": None, "play_gap_ms": None,
                }
                try:
                    # REJECT: the second-stage verifier did not confirm the wake word,
                    # or another device's wake of the same phrase won the arbitration.
                    # Abort the run silently — NO handler/TTS, NO ack (guarded above),
                    # NO STT_VAD_END (that would trigger the thinking indicator). Emit
                    # an empty STT_END (to balance STT_START) and RUN_END (which stops
                    # the device streaming). A duplicate's phrase is then recognized by
                    # the first enabled chain's STT for the run log only; the device
                    # never gets that transcript. Placed INSIDE the inner try so the
                    # finally still records the run + stores its audio.
                    if reason in REJECT_REASONS:
                        if reason == "duplicate":
                            logger.info(
                                f"{self.name}: 🚫 wake dropped as a duplicate of another "
                                f"device's wake"
                            )
                        else:
                            logger.info(
                                f"{self.name}: 🚫 wake word rejected by verifier "
                                f"(reason={reason}, score={self._wakeword_score})"
                            )
                        record["result"] = "rejected"
                        record["reason"] = reason
                        await self._close_stt_streams(stt_streams)
                        stt_streams = {}
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit(StageEvent.RUN_END, {})
                        if reason == "duplicate" and transcribe:
                            stt_t = time.perf_counter()
                            try:
                                first = chains.enabled_chains(self.core.chains)[0]
                                record["stt_text"] = await chains.transcribe(self.rt, first, pcm)
                            except Exception as e:  # noqa: BLE001 - the run is recorded without it
                                logger.error(f"{self.name}: duplicate STT failed: {e}")
                            record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
                        return
                    # VAD found no speech in the whole window — this is silence/noise,
                    # not an utterance. Skip STT entirely: Whisper hallucinates stray
                    # phrases on non-speech audio (e.g. "Продолжение следует..."), which
                    # would then pollute the run log or flow into the LLM/TTS. Balance the
                    # STT_START emitted in on_start with an empty STT_END, end the run, and
                    # let the finally below record it as an empty run (result="empty",
                    # stt_text="", t_stt=0 — all left at their defaults).
                    if reason == "no_speech":
                        logger.info(
                            f"{self.name}: 😶 no speech detected by VAD, skipping STT"
                        )
                        # Discard any streaming sessions WITHOUT reading their results —
                        # we never want a transcript for silence.
                        await self._close_stt_streams(stt_streams)
                        stt_streams = {}
                        keep_record = followup is None
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit(StageEvent.RUN_END, {})
                        return
                    # Utterance ended and there IS speech to process: tell the device to
                    # enter the "thinking" LED phase now. It stays in thinking through STT
                    # and the LLM until TTS_START switches it to "replying" — this is the
                    # indicator shown to the user between their question and the answer.
                    self._emit(StageEvent.STT_VAD_END, {})
                    filler_fired = False  # at most one early filler per run

                    async def _speak_filler(text: str, tool_names: list[str]) -> None:
                        # Policy lives here (llm.py is policy-free): speak at most once per
                        # run, and only for a SLOW tool (so instant smart-home actions don't
                        # double-talk). The slow KNOWLEDGE comes from the tool source via
                        # hub.is_slow (each source declares whether its tools are slow).
                        # Schedule synthesis+play as a fire-and-forget task
                        # so the slow tool is NOT delayed by filler TTS. Synchronous
                        # gate/dedup first (no await before we decide), then schedule.
                        nonlocal filler_fired
                        if filler_fired or not self.has_speaker:
                            return
                        if not any(self.hub.is_slow(n) for n in tool_names):
                            return  # fast action: the final reply alone is enough
                        spoken = clean_llm_output(text)  # tag cleanup only; engine post-processing happens inside the TTS backend
                        if not spoken:
                            return
                        filler_fired = True
                        record["filler_text"] = spoken
                        record["t_filler"] = int((time.perf_counter() - t0) * 1000)
                        task = asyncio.create_task(self._deliver_filler(spoken))
                        self._filler_tasks.add(task)
                        task.add_done_callback(self._filler_tasks.discard)

                    def _on_text(text: str, t_stt: int) -> None:
                        # The first non-empty transcript of any chain: shown on the device
                        # (STT_END) and in the panel (live STT partial), then the device
                        # enters the intent phase. Fires at most once per run.
                        record["stt_text"] = text
                        record["t_stt"] = t_stt
                        self._emit(StageEvent.STT_END, {"text": text})
                        self._emit_live(record)
                        self._emit(StageEvent.INTENT_START, {})

                    # A phrase heard in a follow-up window: the follow-up check of its
                    # first transcript, asked before anything reaches the device. An
                    # accepted phrase gets the end-of-phrase chime now and runs like any
                    # phrase; a rejected one ends the run silently (below). The check
                    # lands in the run's attempts.
                    followup_check = None

                    async def _followup_gate(text: str) -> bool:
                        nonlocal followup_check
                        gate = self.rt.followup_backend
                        t = time.perf_counter()
                        verdict = await gate.check(*followup, text)
                        if verdict.accepted:
                            outcome = "accepted"
                            self._schedule_ack()
                        else:
                            outcome = "rejected" if verdict.probability is not None else "error"
                        followup_check = {
                            "kind": "followup", "provider": self._backend_desc(gate),
                            "text": text, "outcome": outcome, "probability": verdict.probability,
                            "t": int((time.perf_counter() - t) * 1000),
                        }
                        return verdict.accepted

                    # The recognition chains: each enabled chain's STT then handler, in
                    # order, until one answers. Streaming sessions opened at speech were
                    # fed in on_audio; the runner finishes them. A handler failure is
                    # mapped to a configured spoken phrase — the raw API error text never
                    # reaches the user's ears.
                    run = await chains.run(
                        self.rt,
                        device=self.name,
                        request=self._llm_request,
                        pcm=pcm,
                        streams=dict(stt_streams),
                        on_text=_on_text,
                        on_filler=_speak_filler,
                        gate=_followup_gate if followup is not None else None,
                    )
                    self._record_chain_run(record, run)
                    if followup_check is not None:
                        record["attempts"].insert(0, followup_check)
                        if followup_check["outcome"] != "accepted":
                            # Not a continuation of the dialog, or the check failed: the
                            # phrase is dropped without a sound — no handler ran, no
                            # transcript goes to the device, nothing joins the context.
                            # The log still shows what was heard.
                            record["stt_text"] = followup_check["text"]
                            record["result"] = "rejected"
                            record["reason"] = ("followup_reject"
                                                if followup_check["outcome"] == "rejected"
                                                else "followup_error")
                            logger.info(f"{self.name}: 🚫 follow-up dropped "
                                        f"({record['reason']}, "
                                        f"probability={followup_check['probability']})")
                            self._emit(StageEvent.STT_END, {"text": ""})
                            self._emit(StageEvent.RUN_END, {})
                            return
                    if run.reply is None:
                        # No chain heard anything (or every STT failed): nothing is spoken.
                        # Balance the STT_START emitted in on_start with an empty STT_END
                        # and end the run; the finally records it ("empty" or an STT error).
                        logger.info(f"{self.name}: no chain heard a request, ending run")
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit_live(record)
                        self._emit(StageEvent.RUN_END, {})
                        return
                    reply = run.reply
                    # Live LLM partial: record now carries llm_text, model, tokens,
                    # rounds, t_llm, the chain attempts and the updated result.
                    self._emit_live(record)

                    # Only an answered turn is a conversation turn: don't pollute the
                    # context history with a spoken fallback or decline phrase.
                    if run.answered:
                        try:
                            context.append_context(
                                self._context_path,
                                run.text,
                                reply,
                                max_turns=self.core.context.max_turns,
                                ttl_seconds=self.core.context.ttl_seconds,
                            )
                        except Exception as e:
                            # Context persistence failure must not break the run.
                            logger.error(f"{self.name}: context append failed: {e}")

                    self._emit(StageEvent.INTENT_END, {"conversation_id": conversation_id})

                    # Accent (stress) stage: place '+' stress marks on the reply so every TTS
                    # backend pronounces Russian correctly. RuAccent's output IS the canonical
                    # '+vowel' LLM->TTS contract, so the per-backend adaptation handles it
                    # unchanged. record["llm_text"] keeps the original (mark-free) reply for the
                    # panel; only the text sent to TTS is accented. Isolated: a failure falls back
                    # to the un-accented reply so it can never break the run.
                    stress = self.stress_backend
                    if stress is not None:
                        ra_t = time.perf_counter()
                        accented = reply
                        accent_failed = False
                        try:
                            accented = await stress.accentize(reply)
                        except Exception as e:
                            accent_failed = True
                            logger.error(f"{self.name}: accent stage [{self._backend_desc(stress)}] failed: {e}")
                        record["t_stress"] = int((time.perf_counter() - ra_t) * 1000)
                        # Store the accent-stage output (the text actually sent to TTS) ONLY when it
                        # differs from the LLM reply, so the panel shows it without cluttering no-op
                        # (disabled / nothing-to-stress) runs. record["llm_text"] keeps the original.
                        if accented != reply:
                            record["stress_text"] = accented
                        reply = accented
                        # Log the accent stage on the success path only; failures are
                        # already surfaced by the logger.error above. `reply` now holds
                        # the accented text actually sent to TTS.
                        if not accent_failed:
                            logger.info(
                                f"{self.name}: ✒️ Accent [{self._backend_desc(stress)}] "
                                f"({record['t_stress'] / 1000:.2f}s): {reply!r}"
                            )
                        self._emit_live(record)

                    self._emit(StageEvent.TTS_START, {"text": reply})
                    played = None  # what the play hook returned for the answer stream
                    # A device without a speaker gets the reply text only: no synthesis.
                    if self.has_speaker:
                        # The backend of the variant that started: it both synthesizes
                        # and labels the log lines, so they can never diverge. None until
                        # a variant starts.
                        tts = None
                        try:
                            tts_t = time.perf_counter()
                            # The TTS variants in order until one yields its first chunk;
                            # every variant tried lands in the run's attempts. All failing
                            # raises the last error into the except below.
                            tts, mime, audio = await self._start_tts(reply, record["attempts"])
                            # t_tts = time-to-stream-start: the user-perceived TTS
                            # latency (the device plays while synthesis continues).
                            record["t_tts"] = int((time.perf_counter() - tts_t) * 1000)
                            logger.info(
                                f"{self.name}: 🔊 TTS [{self._backend_desc(tts)}] "
                                f"stream start ({time.perf_counter() - tts_t:.2f}s)"
                            )
                            tts_chunks = []
                            try:
                                played = await self.play(mime, self._tee_chunks(audio, tts_chunks))
                            finally:
                                # The tee only closes `audio` once it has started.
                                await self._close_stream(audio)
                            if played is not None:
                                record["play_dry"] = played["dry"]
                                record["play_gap_ms"] = played["gap_ms"]
                            clip = b"".join(tts_chunks)
                            record["audio_bytes"] = len(clip)
                            record["audio_fmt"] = mime.partition(";")[0].rpartition("/")[2]
                            # Keep the full native clip for the run's stored TTS audio (see
                            # finally); raw PCM is stored as WAV so the panel can play it.
                            rate = pcm_rate(mime)
                            if rate is not None:
                                tts_audio, tts_mime = pcm_to_wav_bytes(clip, rate=rate), "audio/wav"
                            else:
                                tts_audio, tts_mime = clip, mime
                            logger.info(
                                f"{self.name}: 🔊 TTS played "
                                f"({time.perf_counter() - tts_t:.2f}s total, {len(clip)} bytes)"
                            )
                        except Exception as e:
                            # The run still ends cleanly. result="error" unconditionally,
                            # but only claim the stage/text if nothing earlier set them:
                            # when the LLM already reported an error (error_stage="LLM") we
                            # continued into TTS anyway, so a TTS failure here must not
                            # overwrite the LLM root cause.
                            record["result"] = "error"
                            if record["error_stage"] is None:
                                record["error_stage"] = "TTS"
                                record["error_text"] = str(e)
                            # No backend when no variant started (each failure is
                            # already logged by _start_tts).
                            where = f" [{self._backend_desc(tts)}]" if tts is not None else ""
                            logger.error(f"{self.name}: TTS{where} failed: {e}")

                    logger.info(
                        f"{self.name}: ✅ run complete in {time.perf_counter() - t0:.2f}s"
                    )
                    # The answer played out (not dropped by a new wake or play_stop) on
                    # a device that reopens its microphone, with the stage on and no
                    # proactive speech queued behind it: the session stays open for a
                    # follow-up phrase instead of ending.
                    fb = self.rt.followup_backend
                    if (played is not None and played["outcome"] == "played"
                            and self.has_follow_up and not self._speaking
                            and fb is not None and not isinstance(fb, NoFollowup)):
                        await self._arm_followup(conversation_id, run.text, run.reply)
                    else:
                        self._emit(StageEvent.RUN_END, {})
                except Exception as e:
                    record["result"] = "error"
                    record["error_stage"] = "pipeline"
                    record["error_text"] = str(e)
                    logger.exception(f"{self.name}: pipeline run failed: {e}")
                    self._emit(
                        StageEvent.ERROR,
                        {"code": "server_error", "message": str(e)},
                    )
                    self._emit(StageEvent.RUN_END, {})
                finally:
                    # Single cleanup point for the streaming STT sessions covering every
                    # path out of the inner try (no_speech, STT error, empty/non-empty
                    # success, later-stage exceptions). finish() already drains/closes
                    # the gRPC call, so this is a no-op there; for the error/early-exit
                    # paths and for parallel sessions whose result went unused it
                    # actually closes the session.
                    await self._close_stt_streams(stt_streams)
                    # Record the run on every non-empty-pcm path (empty-STT return,
                    # success, TTS-fail, exception) but a silent follow-up window. A
                    # recording failure must never break the run or swallow RUN_END,
                    # so it is fully wrapped.
                    if self.runs_store is not None and keep_record:
                        record["t_total"] = int((time.perf_counter() - t0) * 1000)
                        try:
                            run_id = await asyncio.to_thread(self.runs_store.insert, record)
                        except Exception as e:
                            logger.error(f"{self.name}: run record failed: {e}")
                        else:
                            # Store the finalized utterance audio in a rolling window of
                            # the last runs.audio_keep, so it can be downloaded/played from
                            # the log to diagnose mis-triggers (e.g. a wake-word tail
                            # reaching STT). When the device streamed both mic channels the
                            # WAV is STEREO: left = the pipeline/STT channel (trimmed +
                            # conditioned — exactly what STT received), right = the other
                            # raw channel (same trim, no conditioning); mono when the
                            # device streams a single channel. Best-effort: a storage
                            # failure must never break the run or swallow the broadcast.
                            stored_audio = False
                            if self.core.runs.store_audio and pcm:
                                try:
                                    wav = pcm_to_wav_bytes(pcm, pcm2)
                                    await asyncio.to_thread(
                                        self.runs_store.put_audio,
                                        run_id, wav, self.core.runs.audio_keep,
                                    )
                                    stored_audio = True
                                except Exception as e:
                                    logger.error(
                                        f"{self.name}: utterance audio store failed: {e}"
                                    )
                            # Store the generated TTS reply audio (native backend format) in the same
                            # rolling window as the utterance audio (runs.audio_keep) so operators can play
                            # back exactly what the assistant spoke — shown right after the Accents text in
                            # the log detail view. Best-effort: a storage failure must never break the run.
                            # tts_audio is set only when synthesis produced bytes (see the TTS section).
                            if self.core.runs.store_audio and tts_audio:
                                try:
                                    await asyncio.to_thread(
                                        self.runs_store.put_tts_audio,
                                        run_id, tts_audio, tts_mime, self.core.runs.audio_keep,
                                    )
                                except Exception as e:
                                    logger.error(f"{self.name}: TTS audio store failed: {e}")
                            # Defer the live broadcast until the lock is released so a slow
                            # WebSocket consumer can't backpressure the next run on this speaker.
                            if self.run_events is not None:
                                pending_run = summary_row(
                                    record, run_id, has_audio=stored_audio
                                )
        finally:
            # Outside the lock: first drain any in-flight live stage broadcasts, then
            # push the finalized run. The live partials are scheduled as background
            # tasks (see _emit_live) so a slow WS client can't backpressure the
            # per-device lock held during the run; draining the chain HERE — outside
            # the lock — both guarantees the finalized broadcast never overtakes a
            # partial and keeps a stuck client from delaying the next run on this
            # speaker. This `finally` (not merely after the `async with`) also ensures
            # the finalized broadcast still fires on the early-return paths inside the
            # lock (no_speech / empty-transcription set `pending_run` then `return`).
            # Fully isolated — a broadcast failure or slow client must never affect the
            # run or the lock.
            # The run owns its STT sessions from the claim on, so a cancel (a new wake)
            # that lands while it waits for the lock, before the inner try, closes them
            # here; aclose() is idempotent, so the inner close is harmless.
            await self._close_stt_streams(stt_streams)
            tail = self._live_send_tail
            self._live_send_tail = None
            if tail is not None:
                try:
                    await tail
                except Exception as e:  # noqa: BLE001 - draining must never break the run
                    logger.debug(f"{self.name}: live run update drain failed: {e}")
            if pending_run is not None:
                try:
                    await self.run_events.broadcast({"type": "run", "run": pending_run})
                except Exception as e:
                    logger.error(f"{self.name}: run broadcast failed: {e}")

    async def run_text(self, text: str, speak: bool = True) -> dict:
        """Public entry for TEXT requests (agent MCP `ask`): a full handler turn
        through the recognition chains (with tools and conversation context),
        optionally spoken on the speaker.

        A simplified mirror of the chains/TTS portion of `_run`: no audio, no VAD/STT
        and no stage events of its own (there is no voice session for a text run, so
        they would reach the device out of band; speak() first ends an open follow-up
        window, see _close_followup). Serialized with voice runs via the same per-device
        lock. The run is recorded to the runs store with reason="text" and the
        request in stt_text, so the panel run log shows it in the usual column.

        Returns {"reply", "result", "error_stage", "error_text"}.
        """
        pending_run = None
        reply = ""
        try:
            async with self._lock:
                t0 = time.perf_counter()
                # Same record shape as `_run`'s: audio-stage fields stay at their
                # text-run values (t_vad/t_stt = 0, audio_* = None).
                record = {
                    "ts": time.time(),
                    "device": self.name,
                    "reason": "text",
                    "result": "empty",
                    "t_vad": 0,
                    "t_stt": 0, "t_llm": 0, "t_stress": 0, "t_tts": 0,
                    "stt_text": text, "llm_text": "", "stress_text": "",
                    "filler_text": "", "t_filler": None,
                    "model": None, "tokens": None,
                    "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
                    "error_stage": None, "error_text": None,
                    "rounds": [],
                    "request": None,
                    "chain": None,
                    "attempts": [],
                    "play_dry": None, "play_gap_ms": None,
                }
                try:
                    # The recognition chains, entered at the handler step (no STT): each
                    # enabled chain's handler in order until one answers. No on_filler:
                    # a text run has no listener waiting in silence, so no early filler
                    # line is spoken.
                    run = await chains.run(
                        self.rt, device=self.name, request=self._llm_request, text=text,
                    )
                    self._record_chain_run(record, run)
                    reply = run.reply

                    # Only an answered turn is a conversation turn: don't pollute the
                    # context history with a spoken fallback or decline phrase.
                    if run.answered:
                        try:
                            context.append_context(
                                self._context_path,
                                text,
                                reply,
                                max_turns=self.core.context.max_turns,
                                ttl_seconds=self.core.context.ttl_seconds,
                            )
                        except Exception as e:
                            # Context persistence failure must not break the run.
                            logger.error(f"{self.name}: context append failed: {e}")

                    # Accent (stress) stage: only accent the text actually spoken; the
                    # value RETURNED to the caller stays the plain `reply`. RuAccent's
                    # output is the canonical '+vowel' contract every TTS backend adapts.
                    # Isolated: a failure falls back to the un-accented text.
                    spoken = reply
                    stress = self.stress_backend
                    if speak and reply and stress is not None:
                        ra_t = time.perf_counter()
                        accent_failed = False
                        try:
                            spoken = await stress.accentize(reply)
                        except Exception as e:
                            accent_failed = True
                            logger.error(f"{self.name}: accent stage [{self._backend_desc(stress)}] failed: {e}")
                        record["t_stress"] = int((time.perf_counter() - ra_t) * 1000)
                        # Store the accent-stage output ONLY when it differs from the
                        # reply (mirror the _run rule); record["llm_text"] keeps the
                        # original. The value returned to the caller stays `reply`.
                        if spoken != reply:
                            record["stress_text"] = spoken
                        # Mirror _run: log the accent stage on success only; `spoken`
                        # holds the accented text actually sent to TTS.
                        if not accent_failed:
                            logger.info(
                                f"{self.name}: ✒️ Accent [{self._backend_desc(stress)}] "
                                f"({record['t_stress'] / 1000:.2f}s): {spoken!r}"
                            )

                    if speak and reply:
                        try:
                            played = await self.speak(spoken, record["attempts"])
                            # The variants were tried back to back, so their times add
                            # up to the time from the first attempt to the start.
                            record["t_tts"] = sum(
                                a["t"] for a in record["attempts"] if a["kind"] == "tts")
                            if played is not None:
                                record["play_dry"] = played["dry"]
                                record["play_gap_ms"] = played["gap_ms"]
                        except Exception as e:
                            # Same precedence rule as `_run`: a TTS failure must
                            # not overwrite an earlier (LLM) root cause. The reply
                            # is still returned to the caller.
                            record["result"] = "error"
                            if record["error_stage"] is None:
                                record["error_stage"] = "TTS"
                                record["error_text"] = str(e)
                            logger.error(f"{self.name}: text-run TTS failed: {e}")
                except Exception as e:
                    # Mirror `_run`'s catch-all: an unexpected failure is recorded
                    # honestly and never propagates to the MCP caller.
                    record["result"] = "error"
                    record["error_stage"] = "pipeline"
                    record["error_text"] = str(e)
                    logger.exception(f"{self.name}: text run failed: {e}")
                finally:
                    # Record the run; a store failure must never break the call.
                    if self.runs_store is not None:
                        record["t_total"] = int((time.perf_counter() - t0) * 1000)
                        try:
                            run_id = await asyncio.to_thread(self.runs_store.insert, record)
                        except Exception as e:
                            logger.error(f"{self.name}: run record failed: {e}")
                        else:
                            # Defer the broadcast until the lock is released (same
                            # reason as `_run`).
                            if self.run_events is not None:
                                pending_run = summary_row(record, run_id)
        finally:
            # Outside the lock: best-effort push of the finalized run to live
            # panel subscribers; only fires when the insert succeeded.
            if pending_run is not None:
                try:
                    await self.run_events.broadcast({"type": "run", "run": pending_run})
                except Exception as e:
                    logger.error(f"{self.name}: run broadcast failed: {e}")
        return {
            "reply": reply,
            "result": record["result"],
            "error_stage": record["error_stage"],
            "error_text": record["error_text"],
        }

    @staticmethod
    async def _synthesize_stream(backend, text: str, lang: str = "ru"):
        """Resolve `backend`'s streaming synthesis entry point.

        Backends on the TtsBackend contract always have synthesize_stream (the
        ABC ships a buffered single-chunk default); a duck-typed backend that
        only exposes synthesize() is routed through that same default adapter,
        so the streaming path never hard-requires the method. Takes the backend
        explicitly so callers keep their read-the-property-once pattern."""
        fn = getattr(backend, "synthesize_stream", None)
        if fn is None:
            return await TtsBackend.synthesize_stream(backend, text, lang)
        return await fn(text, lang)

    @staticmethod
    async def _close_stream(chunks) -> None:
        """Best-effort aclose() of a chunk iterator that will never be (further)
        iterated. Guarded: the default buffered TtsBackend adapter and the
        streaming providers return async generators (which have aclose), but a
        plain duck-typed iterable may not. A close failure is logged, never
        raised, so it can't mask the original error on the failure paths."""
        aclose = getattr(chunks, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception as e:  # noqa: BLE001 - closing is best-effort
            logger.debug(f"TTS chunk iterator close failed: {e!r}")

    @staticmethod
    async def _tee_chunks(chunks, sink):
        """Pass-through wrapper over a TTS chunk async-iterator that also appends each
        chunk to `sink`, so _run keeps a copy of the full streamed clip to store as the
        run's TTS audio. On early close (aclose), the underlying iterator is closed
        too, so the backend's open response is always released."""
        try:
            async for chunk in chunks:
                sink.append(chunk)
                yield chunk
        finally:
            await Pipeline._close_stream(chunks)

    async def _start_tts(self, text: str, attempts: list | None = None):
        """Start synthesizing `text` on the first enabled TTS variant that starts.

        The enabled variants (tts.variants) are tried in config order. Each one's
        stream is opened and its FIRST chunk pulled, so a provider that fails before
        yielding any audio is passed over before anything reaches the device. Once a
        variant has started nothing switches: a later failure (mid-stream, playback)
        is the caller's. Appends one {"kind": "tts", "provider", "outcome": "spoke" |
        "error", "error", "t"} entry (t in ms) per variant tried to `attempts`.

        Returns (backend, mime, audio): `audio` is a chunk iterator yielding the
        pre-fetched chunk and then the rest, which the caller hands to play() (and
        closes). Raises the last variant's error when none started.
        """
        error = None
        for pid in self.rt.svc.tts_variants():
            backend = self.rt.tts_backends.get(pid)
            t = time.perf_counter()
            try:
                if backend is None:
                    raise StageError("tts", f"tts backend {pid!r} is not loaded")
                mime, chunks = await self._synthesize_stream(backend, text, "ru")
                audio = await _PrefetchedChunks.start(chunks)
            except Exception as e:
                error = e
                if attempts is not None:
                    attempts.append({"kind": "tts", "provider": pid, "outcome": "error",
                                     "error": str(e), "t": int((time.perf_counter() - t) * 1000)})
                desc = self._backend_desc(backend) if backend is not None else pid
                logger.error(f"{self.name}: TTS [{desc}] variant {pid!r} failed to start: {e}")
                continue
            if attempts is not None:
                attempts.append({"kind": "tts", "provider": pid, "outcome": "spoke",
                                 "error": None, "t": int((time.perf_counter() - t) * 1000)})
            return backend, mime, audio
        raise error

    async def _close_followup(self) -> None:
        """End a follow-up window that has heard no phrase yet the way silence ends it:
        the device gets `state idle` and turns its microphone off. Proactive speech
        calls it first, or the speaker would hear itself in the window as a phrase."""
        if self._followup is None or self._finalized:
            return
        # Synchronously, before any await: on_audio drops whatever still arrives.
        self._followup = None
        self._finalized = True
        streams, self._stt_streams = self._stt_streams, {}
        logger.info(f"{self.name}: follow-up window closed for proactive speech")
        self._emit(StageEvent.STT_END, {"text": ""})
        self._emit(StageEvent.RUN_END, {})
        await self._close_stt_streams(streams)

    async def speak(self, text: str, attempts: list | None = None) -> dict | None:
        """Public entry for proactive speech (reminders, external callers) and the
        filler: synthesize via the first TTS variant that starts (see _start_tts) and
        play it on the device, returning once it has played. A device without a
        speaker gets no synthesis at all. `attempts` collects the variants tried (a
        text run records them). Raises on failure (callers decide isolation).

        Returns what the play hook does ({"outcome", "dry", "gap_ms"}); None for a
        device without a speaker.
        """
        if not self.has_speaker:
            return None
        # Counted until it has played: a speech started while an answer still plays is
        # queued behind it on the device, so that answer opens no follow-up window.
        self._speaking += 1
        try:
            await self._close_followup()
            # The backend of the variant that started both synthesizes and labels the
            # log line, so they can never diverge.
            backend, mime, audio = await self._start_tts(text, attempts)
            logger.info(f"{self.name}: 🔔 speak [tts={self._backend_desc(backend)}]: {text!r}")
            return await self.play(mime, audio)
        finally:
            self._speaking -= 1

    async def _deliver_filler(self, text: str) -> None:
        """Speak an early 'filler' line via speak().

        It is queued on the device ahead of the reply, so the user hears a short
        'I'll go check it' line while the slow tool + final LLM round run, instead of
        waiting in silence. Fully isolated: a synthesis/play failure must NEVER affect
        the main run.
        """
        try:
            logger.info(f"{self.name}: 🗣️ filler: {text!r}")
            await self.speak(text)
        except Exception as e:
            logger.error(f"{self.name}: filler failed: {e}")

    def _ack_clip_bytes(self) -> tuple[str, bytes]:
        """Return (mime, audio) for the end-of-phrase ack clip, building/caching once.

        Source resolution (read live off core.ack so it hot-applies): a configured
        sound_path that exists on disk is loaded; otherwise the two-tone chime is
        synthesized. The result is cached keyed by the resolved source, so the file is
        read / transcoded / the chime built only once and re-resolved only when
        sound_path changes. See build_ack_clip for the per-source build details.
        """
        path = (self.core.ack.sound_path or "").strip()
        use_file = bool(path) and os.path.isfile(path)
        source_key = path if use_file else "<generated>"
        cached = self._ack_clip
        if cached is not None and cached[0] == source_key:
            return cached[1], cached[2]
        mime, audio = build_ack_clip(path, name=self.name)
        self._ack_clip = (source_key, mime, audio)
        return mime, audio

    def _schedule_ack(self) -> None:
        """Schedule the fire-and-forget end-of-phrase ack chime, if enabled.

        Synchronous gate (no await before the decision): enabled toggle + a device
        speaker. Building the clip and the play call run inside the background task
        so STT is never delayed. Tracked in _ack_tasks so the still-playing chime
        isn't garbage-collected; each task removes itself.
        """
        if not self.core.ack.enabled or not self.has_speaker:
            return
        task = asyncio.create_task(self._play_ack())
        self._ack_tasks.add(task)
        task.add_done_callback(self._ack_tasks.discard)

    async def _play_ack(self) -> None:
        """Play the ack clip on the device. Fully isolated — a build/play failure must
        NEVER affect or delay the run (which keeps its own final TTS for the answer).
        """
        try:
            mime, audio = self._ack_clip_bytes()
            logger.info(f"{self.name}: 🔔 end-of-phrase ack")
            await self.play(mime, audio)
        except Exception as e:
            logger.error(f"{self.name}: end-of-phrase ack failed: {e}")

    async def on_stop(self, abort: bool = False) -> None:
        """Explicit stop: finalize the run (exactly once)."""
        if self._wakeword_pending and not self._wakeword_confirmed:
            # A stop before the verdict must NOT bypass the gate into STT/LLM/TTS.
            # Cancel the in-flight verify and abort this run as a reject.
            task = self._wakeword_task
            if task is not None and not task.done():
                task.cancel()
            await self._wakeword_reject(error=False)
            return
        claimed = self._claim()
        if claimed is not None:
            await self._run("device_stop", claimed[0], self._conversation_id, pcm2=claimed[1])
