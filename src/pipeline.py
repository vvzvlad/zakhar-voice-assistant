"""Per-speaker voice pipeline: mic PCM -> STT -> LLM -> TTS -> audio URL.

One Pipeline instance per speaker. It owns its own audio buffer and lock; the
brain, TTS backend and audio server are stateless shared dependencies injected
by the DeviceClient.

Server-side VAD end-pointing
----------------------------
The HA Voice PE speaker fires its wake word on-device and then streams mic PCM
to us continuously (silence included). It NEVER signals end-of-speech, so the
device's handle_stop is effectively never called. To know when the user has
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

from src import config_store, context, llm
from src.audio_prep import (
    highpass,
    normalize_peak,
    pcm_to_wav_bytes,
    trim_start_pcm,
    vad_boost,
    write_wav,
)
from src.audio_codec import PLAYABLE_MIMES, to_playable
from src.audio_server import tts_url
from src.tts import TtsBackend
from src.pipeline_events import StageEvent
from src.llm_text import clean_llm_output
from src.prompt import build_system_prompt
from src.runs_store import live_row, summary_row
from src.chime import build_ack_clip
from src.stage_errors import StageError
# SAMPLE_RATE now lives in the VAD stage contract module; re-exported here so the
# many existing `from src.pipeline import SAMPLE_RATE` users keep working.
from src.vad import SAMPLE_RATE, EndpointPolicy  # noqa: F401  (SAMPLE_RATE re-export)

# Generous hard memory cap (~60s of 16 kHz / 16-bit mono PCM). VAD should end the
# utterance long before this; the cap only guards against unbounded growth if VAD
# misbehaves, and also forces a finalize when reached. This applies to NORMAL
# (wake-word) runs only; a manual capture sizes its own cap to the requested
# duration (see _capture_cap_bytes) so a long sample isn't truncated here.
HARD_CAP_BYTES = SAMPLE_RATE * 2 * 60

# Maximum duration (seconds) of a manual "record X seconds" capture. Single source
# of truth shared by the panel API / ESPHome client validation; the device-side
# template number caps at the same value in esphome/zakhar-voice-preroll.yaml.
CAPTURE_MAX_SECONDS = 300

# How long an armed manual capture stays valid waiting for its voice_assistant.start
# to arrive (the press -> start round-trip). If the button press is lost or the
# device never starts, the flag expires after this so it can't silently hijack a
# later real wake-word run. This is the arm-arrival window only — it is independent
# of (and much shorter than) the requested capture audio duration.
ARM_TTL = 5.0

class CaptureBusyError(Exception):
    """A manual capture is already armed/in-flight on this pipeline.

    Raised by arm_capture() when a second concurrent capture is requested for the
    same device while the first is still pending. The panel API maps it to HTTP 409
    ("capture already in progress") — the device can't record two samples at once.
    """


class CaptureEmptyError(Exception):
    """A manual capture finished but produced no audio (empty PCM buffer).

    Distinct from offline/missing-entity/busy errors (HTTP 409): this is a
    server-side capture failure, so the panel API maps it to HTTP 500. Raised
    (via the capture Future) by _finish_capture when the buffered PCM is empty.
    """


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

        # Per-run VAD session, opened in on_start (normal runs only) from the
        # swappable vad stage backend (self.rt.vad_backend). The session owns ALL
        # end-pointing state (framing, counters); the pipeline only feeds it
        # chunks and acts on the returned reason.
        self._vad_session = None
        self._stt_stream = None  # live streaming STT session (only for streaming-capable backends)
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

        # Manual "record X seconds" capture (see arm_capture). _capture_armed is set
        # BEFORE the device's voice_assistant.start fires; the next on_start consumes
        # it into the per-run _capture_run flag so a normal wake-word run is never
        # affected. In a capture run we skip VAD/STT/LLM/TTS entirely and, when the
        # device stops or the server-side deadline passes, return the PCM as WAV
        # bytes to the API caller (ephemeral — nothing is written to the server).
        # _capture_future is that channel: _finish_capture resolves it with the
        # in-memory WAV bytes (or fails it when the armed run never produces audio),
        # and DeviceClient.capture() awaits it.
        self._capture_armed = False
        self._capture_seconds = 0
        self._capture_run = False
        self._capture_deadline = 0.0
        self._capture_future: asyncio.Future | None = None
        # Deadline (time.monotonic()) by which the armed capture's start must arrive;
        # set in arm_capture, checked in on_start so a lost press can't hijack a later run.
        self._capture_arm_deadline = 0.0

        # Injected by DeviceClient on connect (bound to the live API client).
        self.send_event = None
        self.send_audio = None
        self.send_announcement = None

        # Background "filler" announcement tasks (early "I'll go check it" lines). Kept
        # in a set so a still-playing announcement isn't garbage-collected mid-flight;
        # each task removes itself on completion. Fire-and-forget — never awaited by the run.
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

    # Read-through convenience properties: external callers (DeviceClient.announce,
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
    def llm_cfg(self):
        return self.rt.llm_cfg

    @property
    def stt_backend(self):
        return self.rt.stt_backend

    @property
    def llm_backend(self):
        return self.rt.llm_backend

    @property
    def stress_backend(self):
        return self.rt.stress_backend

    @property
    def tts_backend(self):
        return self.rt.tts_backend

    @property
    def hub(self):
        return self.rt.hub

    @property
    def audio_server(self):
        return self.rt.audio_server

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
    def public_base_url(self):
        return self.rt.core.audio.public_base_url

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

    def arm_capture(self, seconds: int) -> "asyncio.Future[bytes]":
        """Arm a manual capture-only run of `seconds` for the NEXT voice_assistant start.

        Called by DeviceClient.capture() BEFORE it presses the device button, so the
        flag is already set when the resulting on_start arrives. The server-side
        capture deadline (seconds + small margin) is a safety net in case the device
        never signals stop; the device normally stops itself after `seconds`. A
        separate arm-arrival deadline (ARM_TTL) bounds how long the flag waits for its
        start so a lost press can't silently capture a later real wake-word run.

        Returns an asyncio.Future the caller awaits for the recorded WAV bytes (the
        manual capture is ephemeral — nothing is written to disk). The Future
        resolves with the WAV bytes in _finish_capture, or fails with an exception
        if the armed run aborts / expires without producing audio.
        """
        # Busy-guard (FIX A): refuse a second concurrent capture on the same
        # pipeline. If a previous capture Future is still pending (armed and waiting,
        # or a capture run is in flight), the device is already recording for another
        # caller — racing a new arm here would overwrite that Future and leave the
        # first awaiter hung until timeout. Reject with a distinct CaptureBusyError so
        # the API can map it to HTTP 409.
        if self._capture_future is not None and not self._capture_future.done():
            raise CaptureBusyError(f"{self.name} capture already in progress")
        # Safety net: never overwrite a still-pending Future. The busy-guard above
        # already rejects that case, so this is unreachable in normal flow — but if it
        # were ever bypassed, fail the orphan instead of silently dropping it so its
        # awaiter can't hang forever. (_fail_capture_future is a no-op on a done one.)
        self._fail_capture_future("superseded")
        self._capture_armed = True
        self._capture_seconds = seconds
        self._capture_future = asyncio.get_running_loop().create_future()
        # Small margin so the device's own stop wins under normal timing; the deadline
        # only ends the capture if the device-stop is lost.
        self._capture_deadline = time.monotonic() + seconds + 2.0
        # Arm-arrival deadline: the press -> voice_assistant.start round-trip must land
        # within ARM_TTL. If it doesn't (lost press / device never starts), on_start
        # clears the stale flag instead of hijacking a later real wake-word run.
        self._capture_arm_deadline = time.monotonic() + ARM_TTL
        return self._capture_future

    def _capture_cap_bytes(self) -> int:
        """Byte cap for the in-flight capture run, sized to the requested duration.

        A capture run must NOT be truncated by the 60 s normal-run HARD_CAP_BYTES:
        the user may have asked for up to CAPTURE_MAX_SECONDS. This returns a cap of
        (_capture_seconds + 2) s of PCM (a couple seconds of margin), so the byte cap
        is only a safety net sized to the request — the capture still ends primarily
        on the device stop (on_stop) or the time-based deadline.
        """
        return (self._capture_seconds + 2) * SAMPLE_RATE * 2

    def _fail_capture_future(self, message: str) -> None:
        """Reject the pending capture Future so a waiting caller never hangs.

        Safe to call when there is no Future or it's already resolved (no-op).
        """
        fut = self._capture_future
        self._capture_future = None
        if fut is not None and not fut.done():
            fut.set_exception(RuntimeError(message))

    def disarm_capture(self) -> None:
        """Cancel a pending manual capture (e.g. the caller timed out waiting).

        Clears the armed flag and fails the still-pending Future so the armed state
        can't hijack a later run and the timed-out caller gets a clean error. If a
        capture was already streaming (_capture_run), a later _finish_capture finds
        the Future already done/None and is a safe no-op — no double-set.

        Because arm_capture's busy-guard (FIX A) refuses a second concurrent capture,
        there is only ever one armed Future per pipeline at a time, so this can only
        ever fail the Future the matching capture() created — never a different
        request's Future.
        """
        self._capture_armed = False
        self._fail_capture_future("capture cancelled")

    async def on_start(
        self, conversation_id, flags, audio_settings, wake_word_phrase
    ) -> int:
        """Handle voice_assistant start: reset ALL per-run state, announce run."""
        self._conversation_id = conversation_id or ""
        self._buffer.clear()
        self._buffer2.clear()
        # Best-effort close a lingering streaming STT session from a prior aborted
        # run before starting this one (await is fine here — on_start is not the hot
        # path). Then null it so this run opens its own.
        if self._stt_stream is not None:
            try:
                await self._stt_stream.aclose()
            except Exception:  # noqa: BLE001 - defensive cleanup, never break a new run
                pass
            self._stt_stream = None
        self._vad_session = None
        self._vad_peak = 0
        self._finalized = False
        self._audio_logged = False
        self._mic_fallback_logged = False
        # Best-effort cancel any still-pending ack/filler announce tasks from a prior
        # run before starting this one. Both are fire-and-forget on the announce
        # channel and can linger up to the 30 s announce timeout; the ack fires on
        # every non-empty phrase, so without this they could accumulate across runs
        # and even outlive a reconnect. Cancelling here bounds their lifetime to ~one
        # run. We do NOT await them — each task self-removes via its done-callback,
        # and cancellation of an already-finished task is a no-op. Snapshot to a list
        # first since the done-callbacks mutate the sets. Fully isolated/defensive.
        for task in [*self._ack_tasks, *self._filler_tasks]:
            task.cancel()
        # Consume any armed capture into a per-run flag so a normal wake-word run
        # (armed == False) is completely unaffected.
        #
        # Two guards before consuming the flag:
        #  - Expiry (FIX 2): if the arm-arrival deadline has passed, the button press
        #    was likely lost / the device never started. Clear the stale flag and run
        #    this as a NORMAL run, so a later genuine wake word isn't silently captured.
        #  - Wake-word discrimination (FIX 3): a manual start from the button carries
        #    NO wake_word_phrase, whereas a real wake-word run has a non-empty phrase.
        #    Only consume the flag for a phraseless (manual) start; if a phrased run
        #    arrives while armed, leave the flag set so the genuine manual start still
        #    gets captured.
        if self._capture_armed and time.monotonic() > self._capture_arm_deadline:
            logger.warning(
                f"{self.name}: armed capture expired before start arrived; "
                f"running normally"
            )
            self._capture_armed = False
            # Fail the pending Future so the awaiting caller doesn't hang for the
            # full wait_for timeout — the capture will never produce audio now.
            self._fail_capture_future("capture armed but no start arrived")
        # Only a phraseless (manual button) start consumes the flag; a phrased real
        # wake word leaves it armed for the genuine manual start still to come.
        self._capture_run = self._capture_armed and not wake_word_phrase
        if self._capture_run:
            self._capture_armed = False
            self._capture_deadline = time.monotonic() + self._capture_seconds + 2.0
            logger.info(
                f"{self.name}: ⏺️ capture run started "
                f"({self._capture_seconds}s, cid={conversation_id})"
            )
            self._emit(StageEvent.RUN_START, {})
            return 0  # Capture-only: no STT_START, no VAD/STT/LLM/TTS.
        # Open the per-run VAD session. The end-pointing POLICY (generic thresholds)
        # is read live off core.vad here, while the speech classifier itself is the
        # swappable vad stage backend. One session per run, so a hot-swapped backend
        # or a changed policy applies on the next run. Capture runs returned above
        # and never open a session (no VAD in capture mode).
        vad_cfg = self.core.vad
        # Capture the identity from the SAME backend ref used to open the session,
        # so a hot-swap mid-run can't mislabel the finalize log line.
        vad_backend = self.rt.vad_backend
        self._vad_desc = self._backend_desc(vad_backend)
        self._vad_session = vad_backend.open(EndpointPolicy(
            silence_ms=vad_cfg.silence_ms,
            min_speech_ms=vad_cfg.min_speech_ms,
            max_utterance_ms=vad_cfg.max_utterance_ms,
            no_speech_timeout_ms=vad_cfg.no_speech_timeout_ms,
        ))
        logger.info(f"{self.name}: ▶️ run started (cid={conversation_id})")
        self._emit(StageEvent.RUN_START, {})
        self._emit(StageEvent.STT_START, {})
        # Speech is already in-stream (pre-roll captured the command), so the device
        # should show the active "listening for command" LED right away; the matching
        # STT_VAD_END (thinking) is emitted in _run once VAD finalizes the utterance.
        self._emit(StageEvent.STT_VAD_START, {})
        # Streaming STT (optional): if the selected backend supports a live session,
        # open it now and feed it chunks in on_audio so recognition runs DURING
        # speech. VAD still owns end-pointing; on finalize _run calls finish().
        # A backend without open_stream() (or one returning None) is simply
        # non-streaming — NOT an error — so it falls back to batch transcribe()
        # SILENTLY; only a genuine open failure warns. Both paths leave
        # _stt_stream None for the batch route.
        self._stt_stream = None
        opener = getattr(self.stt_backend, "open_stream", None)
        if opener is not None:
            try:
                self._stt_stream = opener()
            except Exception as e:  # noqa: BLE001 - never let STT setup break the run; batch-fallback
                logger.warning(f"{self.name}: streaming STT open failed, using batch: {e}")
                self._stt_stream = None
        return 0  # 0 = audio comes in-band over the API connection.

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

        The speaker streams continuously and never signals end-of-speech, so we
        detect it here: VAD over the PCM finalizes the utterance once speech is
        followed by enough trailing silence (or a max length / no-speech timeout).

        All buffer and VAD state updates happen SYNCHRONOUSLY (no await) before we
        decide to finalize, so concurrent eager on_audio tasks can't corrupt the
        buffer mid-update; the claim/snapshot in _claim() is likewise synchronous,
        so only _run() awaits.
        """
        if self._finalized:
            return  # Ignore late audio for an already-finalized run.

        # Select the mic channel for the WHOLE pipeline (capture + VAD + STT) from
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

        # Manual capture-only mode: accumulate PCM, run NO VAD/STT/LLM/TTS. End on the
        # device stop (on_stop) or when the server-side deadline (seconds + margin) is
        # exceeded — whichever comes first; the buffered PCM is then returned as WAV
        # bytes to the caller (never persisted). The byte cap here is sized to the
        # requested duration (NOT the 60 s normal-run HARD_CAP_BYTES) so a long capture
        # isn't silently truncated at 60 s; it's just a memory safety net.
        if self._capture_run:
            if not self._audio_logged:
                self._audio_logged = True
                logger.info(f"{self.name}: ⏺️ capturing sample...")
            cap_bytes = self._capture_cap_bytes()
            self._buffer.extend(data)
            if len(self._buffer) >= cap_bytes:
                del self._buffer[cap_bytes:]
                await self._finish_capture("maxlen")
            elif time.monotonic() >= self._capture_deadline:
                await self._finish_capture("deadline")
            return

        # Log once per run on the first chunk; the speaker streams many chunks.
        if not self._audio_logged:
            self._audio_logged = True
            logger.info(f"{self.name}: 🎤 receiving audio...")

        # The full utterance audio (everything streamed, silence included) is what
        # we send to STT.
        self._buffer.extend(data)
        # Feed the live streaming STT session (if any) the same selected-channel bytes
        # that go into the buffer. feed() is synchronous and never raises.
        if self._stt_stream is not None:
            self._stt_stream.feed(data)
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

        # All synchronous state is updated above; claim+run last (the only await).
        # _claim() is synchronous and happens-before any await, so it's atomic
        # relative to other eager on_audio tasks — no audio loss, no double finalize.
        if reason is not None:
            claimed = self._claim()
            if claimed is not None:
                # Snapshot the conversation id synchronously with the claim so a
                # re-triggered wake word can't relabel this run's events.
                await self._run(reason, claimed[0], self._conversation_id, pcm2=claimed[1])

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

    def _spoken_llm_fallback(self, e: StageError) -> str:
        """Map an LLM StageError to the configured phrase spoken instead of the
        raw error text (policy lives here, phrases live in LlmConfig)."""
        if e.kind == "rate_limit":
            return self.llm_cfg.reply_rate_limit
        return self.llm_cfg.reply_error

    async def _close_stt_stream(self) -> None:
        """Idempotently close and drop the live streaming STT session, if any."""
        if self._stt_stream is None:
            return
        stream = self._stt_stream
        self._stt_stream = None
        try:
            await stream.aclose()
        except Exception as e:  # noqa: BLE001 - cleanup must never break the run
            logger.debug(f"{self.name}: streaming STT close failed: {e}")

    async def on_disconnect(self) -> None:
        """Device dropped its connection. If a streaming STT session is open for a
        run that was never finalized (the speaker vanished mid-utterance), close it
        so the gRPC stream/channel don't linger until the next run. A FINALIZED run
        owns the session and closes it in _run's finally, so skip it here to avoid
        racing finish() against aclose()."""
        if not self._finalized:
            await self._close_stt_stream()

    async def _run(self, reason, pcm, conversation_id, pcm2: bytes = b"") -> None:
        """Run STT -> LLM -> TTS -> events on the already-claimed audio, once.

        The caller claims the run via _claim() (which sets _finalized, snapshots
        and clears the buffer) and passes the snapshotted pcm here. The lock only
        provides defensive serialization; _claim() already guarantees single entry.
        RUN_END is always sent.
        """
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
                    await self._close_stt_stream()
                    self._emit(StageEvent.RUN_END, {})
                    return

                # End-of-phrase ack ("блям"): we've just finalized the utterance, so play
                # a short confirmation chime to the speaker NOW as immediate "got it"
                # feedback, before STT/LLM/TTS run. Fire-and-forget (scheduled, never
                # awaited) so it never delays STT; tracked in a set so it isn't GC'd
                # mid-playback. The manual capture-only path bypasses _run entirely, so a
                # capture never beeps.
                #
                # Announce-channel overlap (ack <-> filler): both the ack here and the
                # early slow-tool "filler" (_deliver_filler) play over the SAME
                # send_announcement (await_response) path. In the narrow window where a
                # slow-tool filler fires while the ~300 ms ack announce is still playing,
                # the two announces may duck/queue against each other on the device. This
                # is non-fatal — both are isolated, best-effort, and never awaited by the
                # run — and the FINAL TTS reply uses a SEPARATE VA-event channel
                # (TTS_START/TTS_END), so only ack<->filler can transiently overlap, never
                # the real answer.
                self._schedule_ack()

                # Trim the configured lead-in (wake-word tail / button-press click) off the
                # start of the captured sample. The trim is applied ONCE here, so every
                # downstream consumer in this run uses the trimmed sample: STT, the t_vad
                # metric, the capture-session WAV and the stored diagnostic audio. Read live
                # off core.vad so it hot-applies. trim_start_pcm never returns empty audio,
                # so the non-empty guard above still holds afterwards. The manual
                # "record N seconds" capture-only path bypasses _run and is unaffected.
                trim_ms = self.core.vad.trim_start_ms
                if trim_ms > 0:
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
                # break the run, so it is fully wrapped.
                if self.core.capture.enabled:
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
                # audio_ms is intentionally left None for now: we don't decode the
                # synthesized mp3 to measure its playback duration yet.
                # Native TTS reply audio captured during synthesis (see the TTS section) and
                # stored after the run is inserted, mirroring the utterance-audio store. Stays
                # None unless synthesis produced bytes, so a failed/empty TTS stores nothing.
                tts_audio = None
                tts_mime = None
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
                }
                try:
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
                        # Discard any streaming session WITHOUT reading its result —
                        # we never want a transcript for silence.
                        await self._close_stt_stream()
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit(StageEvent.RUN_END, {})
                        return
                    # Utterance ended and there IS speech to process: tell the device to
                    # enter the "thinking" LED phase now. It stays in thinking through STT
                    # and the LLM until TTS_START switches it to "replying" — this is the
                    # indicator shown to the user between their question and the answer.
                    self._emit(StageEvent.STT_VAD_END, {})
                    # Read the backend property ONCE: the same ref both executes the
                    # call and labels the log lines, so they can never diverge.
                    stt = self.stt_backend
                    stt_t = time.perf_counter()
                    try:
                        # Streaming-capable backends already recognized DURING speech
                        # (fed in on_audio): finish() forces the final transcript.
                        # Otherwise fall back to a one-shot batch transcribe().
                        if self._stt_stream is not None:
                            text = await self._stt_stream.finish()
                        else:
                            text = await stt.transcribe(pcm)
                    except StageError as e:
                        # STT failed: record the run honestly as an error (not "empty"),
                        # balance the STT_START emitted in on_start with an empty
                        # STT_END, and end the run. Nothing is spoken — same audible
                        # behavior (silence) as before, but observability tells the truth.
                        record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
                        record["result"] = "error"
                        record["error_stage"] = "STT"
                        record["error_text"] = str(e)
                        logger.error(
                            f"{self.name}: STT [{self._backend_desc(stt)}] failed: {e}"
                        )
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit(StageEvent.RUN_END, {})
                        return
                    record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
                    logger.info(
                        f"{self.name}: 📝 STT [{self._backend_desc(stt)}] "
                        f"({time.perf_counter() - stt_t:.2f}s): {text!r}"
                    )
                    record["stt_text"] = text
                    self._emit(StageEvent.STT_END, {"text": text})
                    # Live STT partial: surface the recognized text in the panel
                    # immediately, before the empty-transcription check (fires for
                    # both empty and non-empty text).
                    self._emit_live(record)
                    if not text.strip():
                        # Empty transcription: result stays "empty"; record in finally.
                        logger.info(f"{self.name}: empty transcription, ending run")
                        self._emit(StageEvent.RUN_END, {})
                        return

                    self._emit(StageEvent.INTENT_START, {})
                    # Read the backend property ONCE: the same ref both runs the
                    # stage and labels the log line, so they can never diverge.
                    llm_backend = self.llm_backend
                    logger.info(
                        f"{self.name}: 🤖 → LLM "
                        f"[{self._backend_desc(llm_backend)}]: {text!r}"
                    )
                    llm_t = time.perf_counter()
                    history = context.load_context(
                        self._context_path,
                        max_turns=self.core.context.max_turns,
                        ttl_seconds=self.core.context.ttl_seconds,
                    )
                    filler_fired = False  # at most one early filler per run

                    async def _speak_filler(text: str, tool_names: list[str]) -> None:
                        # Policy lives here (llm.py is policy-free): speak at most once per
                        # run, and only for a SLOW tool (so instant smart-home actions don't
                        # double-talk). The slow KNOWLEDGE comes from the tool source via
                        # hub.is_slow (each source declares whether its tools are slow).
                        # Schedule synthesis+announce as a fire-and-forget task
                        # so the slow tool is NOT delayed by filler TTS. Synchronous
                        # gate/dedup first (no await before we decide), then schedule.
                        nonlocal filler_fired
                        if filler_fired or self.send_announcement is None:
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

                    llm_failed = False
                    try:
                        # The orchestrator prepares the stage input: the assembled
                        # system prompt (a SQLite read of the active profile; same
                        # blocking profile as the old file IO) plus history/user
                        # text/device. The stage is constructed per run so a
                        # hot-swapped backend/config applies naturally.
                        system_prompt = build_system_prompt(self.core, self.prompt_store)
                        stage = llm.LlmStage(llm_backend, self.hub, self.llm_cfg)
                        result = await stage.respond(
                            llm.LlmRequest(
                                system_prompt=system_prompt,
                                history=history,
                                user_text=text,
                                device=self.name,
                            ),
                            on_filler=_speak_filler,
                        )
                    except StageError as e:
                        # LLM failed: record the raw error, then continue into the
                        # normal TTS path speaking a configured fallback phrase
                        # (rate limits get their own line). The raw API error text
                        # never reaches the user's ears.
                        llm_failed = True
                        record["result"] = "error"
                        record["error_stage"] = "LLM"
                        record["error_text"] = str(e)
                        logger.error(f"{self.name}: LLM failed: {e}")
                        reply = self._spoken_llm_fallback(e)
                        # Preserve observability on failure: the stage attaches the
                        # partial data accumulated before the error (the same fields
                        # the old partial trace dict carried).
                        partial = getattr(e, "partial", None) or {}
                        record["model"] = partial.get("model")
                        record["tokens"] = partial.get("tokens")
                        record["rounds"] = partial.get("rounds") or []
                        record["request"] = partial.get("request")
                    else:
                        reply = result.reply
                        record["model"] = result.model
                        record["tokens"] = result.tokens
                        record["rounds"] = result.rounds
                        record["request"] = result.request_debug
                        # Did the model actually run any tool this run?
                        record["result"] = "tool" if result.tool_used else "ok"
                    record["t_llm"] = int((time.perf_counter() - llm_t) * 1000)
                    record["llm_text"] = reply
                    logger.info(
                        f"{self.name}: 💬 LLM reply ({time.perf_counter() - llm_t:.2f}s): "
                        f"{reply!r}"
                    )
                    # Live LLM partial: record now carries llm_text, model, tokens,
                    # rounds, t_llm and the updated result.
                    self._emit_live(record)

                    # A failed LLM turn is not a conversation turn: don't pollute the
                    # context history with the spoken fallback phrase.
                    if not llm_failed:
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

                    self._emit(
                        StageEvent.INTENT_END,
                        {
                            "conversation_id": conversation_id,
                            "continue_conversation": "0",
                        },
                    )

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
                    # Read the backend property ONCE: the same ref both synthesizes
                    # and labels the log lines, so they can never diverge.
                    tts = self.tts_backend
                    try:
                        tts_t = time.perf_counter()
                        if self.core.audio.stream_tts:
                            # Streaming TTS: register a stream URL and emit
                            # TTS_END as soon as synthesis STARTS — the speaker
                            # fetches the URL itself and plays while the backend
                            # is still producing audio. Errors raised before the
                            # URL exists (connect/auth, put_stream) fall through
                            # to the existing except below, exactly like the
                            # buffered path.
                            mime, chunks = await self._synthesize_stream(tts, reply, "ru")
                            tts_chunks = []
                            ext, url, feed_task = await self.serve_audio_stream(
                                mime, self._tee_chunks(chunks, tts_chunks)
                            )
                            try:
                                # t_tts = time-to-stream-start: the user-perceived TTS
                                # latency (total synth time is logged after the feed).
                                record["t_tts"] = int((time.perf_counter() - tts_t) * 1000)
                                logger.info(
                                    f"{self.name}: 🔊 TTS [{self._backend_desc(tts)}] "
                                    f"stream start ({time.perf_counter() - tts_t:.2f}s) "
                                    f"▶ serving {url}"
                                )
                                self._emit(StageEvent.TTS_END, {"url": url})  # speaker starts fetching NOW
                                try:
                                    # Synthesis is still running; wait for the whole
                                    # feed before closing the run record.
                                    served_bytes = await feed_task
                                except Exception as e:
                                    # Mid-stream TTS failure. TTS_END was already
                                    # emitted, so the speaker may play a truncated
                                    # clip — acceptable; record the failure honestly.
                                    # Same precedence as the buffered except below:
                                    # never clobber an earlier (LLM) root cause.
                                    record["result"] = "error"
                                    if record["error_stage"] is None:
                                        record["error_stage"] = "TTS"
                                        record["error_text"] = str(e)
                                    logger.error(
                                        f"{self.name}: TTS [{self._backend_desc(tts)}] "
                                        f"failed mid-stream: {e}"
                                    )
                                else:
                                    record["audio_bytes"] = served_bytes
                                    record["audio_fmt"] = ext
                                    # Keep the full native clip for the run's stored TTS audio (see finally).
                                    tts_audio = b"".join(tts_chunks)
                                    tts_mime = mime
                                    logger.info(
                                        f"{self.name}: 🔊 TTS stream complete "
                                        f"({time.perf_counter() - tts_t:.2f}s total, "
                                        f"{served_bytes} bytes)"
                                    )
                            finally:
                                # Orphan guard: if anything between obtaining the
                                # feed task and consuming its result raises (e.g.
                                # a custom send_event failing on TTS_END) or the
                                # run is cancelled, the task would otherwise never
                                # be awaited ("Task exception was never retrieved").
                                # On the happy/handled paths the task is already
                                # done: cancel() is then a no-op and the re-await
                                # only re-surfaces an already-consumed result/
                                # exception, logged at debug.
                                if not feed_task.done():
                                    feed_task.cancel()
                                try:
                                    await feed_task
                                except BaseException as fe:  # noqa: BLE001 - reaping must not mask the original error
                                    logger.debug(
                                        f"{self.name}: TTS feed task reaped: {fe!r}"
                                    )
                        else:
                            mime, audio = await tts.synthesize(reply, "ru")
                            record["t_tts"] = int((time.perf_counter() - tts_t) * 1000)
                            logger.info(
                                f"{self.name}: 🔊 TTS [{self._backend_desc(tts)}] "
                                f"({time.perf_counter() - tts_t:.2f}s, "
                                f"{len(audio)} bytes)"
                            )
                            ext, url, served_bytes = await self.serve_audio(mime, audio)
                            # Record what the speaker actually downloads (post-transcode),
                            # not the backend's native-format size.
                            record["audio_bytes"] = served_bytes
                            record["audio_fmt"] = ext
                            # Keep the full native clip for the run's stored TTS audio (see
                            # finally). Set only AFTER a successful serve_audio so a serving/
                            # transcode failure stores nothing — symmetric with the streaming
                            # branch, which captures only in its post-feed success path.
                            tts_audio = audio
                            tts_mime = mime
                            logger.info(f"{self.name}: ▶ serving {url}")
                            self._emit(StageEvent.TTS_END, {"url": url})
                    except Exception as e:
                        # No TTS_END on failure; the run still ends cleanly.
                        # result="error" unconditionally, but only claim the stage/
                        # text if nothing earlier set them: when the LLM already
                        # reported an error (error_stage="LLM") we continued into TTS
                        # anyway, so a TTS failure here must not overwrite the LLM
                        # root cause.
                        record["result"] = "error"
                        if record["error_stage"] is None:
                            record["error_stage"] = "TTS"
                            record["error_text"] = str(e)
                        logger.error(
                            f"{self.name}: TTS [{self._backend_desc(tts)}] failed: {e}"
                        )

                    logger.info(
                        f"{self.name}: ✅ run complete in {time.perf_counter() - t0:.2f}s"
                    )
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
                    # Single cleanup point for the streaming STT session covering every
                    # path out of the inner try (no_speech, STT error, empty/non-empty
                    # success, later-stage exceptions). finish() already drains/closes
                    # the gRPC call, so this is an idempotent ref-nulling no-op there;
                    # for the error/early-exit paths it actually closes the session.
                    await self._close_stt_stream()
                    # Record the run on every non-empty-pcm path (empty-STT return,
                    # success, TTS-fail, exception). A recording failure must never
                    # break the run or swallow RUN_END, so it is fully wrapped.
                    if self.runs_store is not None:
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
        """Public entry for TEXT requests (agent MCP `ask`): a full LLM turn with
        tools and conversation context, optionally spoken on the speaker.

        A simplified mirror of the LLM/TTS portion of `_run`: no audio, no VAD/STT
        and NO voice-assistant events (`_emit` is never called — there is no live
        ESPHome voice session for a text run, so VA events would hit a live voice
        connection out of band). Serialized with voice runs via the same per-device
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
                }
                try:
                    # Read the backend property ONCE: the same ref both runs the
                    # stage and labels the log line, so they can never diverge.
                    llm_backend = self.llm_backend
                    logger.info(
                        f"{self.name}: 🤖 → LLM (text) "
                        f"[{self._backend_desc(llm_backend)}]: {text!r}"
                    )
                    llm_t = time.perf_counter()
                    history = context.load_context(
                        self._context_path,
                        max_turns=self.core.context.max_turns,
                        ttl_seconds=self.core.context.ttl_seconds,
                    )
                    llm_failed = False
                    try:
                        system_prompt = build_system_prompt(self.core, self.prompt_store)
                        stage = llm.LlmStage(llm_backend, self.hub, self.llm_cfg)
                        # No on_filler: a text run has no listener waiting in
                        # silence, so no early filler line is spoken.
                        result = await stage.respond(
                            llm.LlmRequest(
                                system_prompt=system_prompt,
                                history=history,
                                user_text=text,
                                device=self.name,
                            ),
                        )
                    except StageError as e:
                        # Same handling as `_run`: record the raw error, reply with
                        # the configured fallback phrase, keep partial observability.
                        llm_failed = True
                        record["result"] = "error"
                        record["error_stage"] = "LLM"
                        record["error_text"] = str(e)
                        logger.error(f"{self.name}: LLM failed: {e}")
                        reply = self._spoken_llm_fallback(e)
                        partial = getattr(e, "partial", None) or {}
                        record["model"] = partial.get("model")
                        record["tokens"] = partial.get("tokens")
                        record["rounds"] = partial.get("rounds") or []
                        record["request"] = partial.get("request")
                    else:
                        reply = result.reply
                        record["model"] = result.model
                        record["tokens"] = result.tokens
                        record["rounds"] = result.rounds
                        record["request"] = result.request_debug
                        record["result"] = "tool" if result.tool_used else "ok"
                    record["t_llm"] = int((time.perf_counter() - llm_t) * 1000)
                    record["llm_text"] = reply
                    logger.info(
                        f"{self.name}: 💬 LLM reply ({time.perf_counter() - llm_t:.2f}s): "
                        f"{reply!r}"
                    )

                    # A failed LLM turn is not a conversation turn: don't pollute
                    # the context history with the spoken fallback phrase.
                    if not llm_failed:
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
                            tts_t = time.perf_counter()
                            await self.speak(spoken)
                            record["t_tts"] = int((time.perf_counter() - tts_t) * 1000)
                        except Exception as e:
                            # Same precedence rule as `_run`: a TTS failure must
                            # not overwrite an earlier (LLM) root cause. The reply
                            # is still returned to the caller.
                            record["result"] = "error"
                            if record["error_stage"] is None:
                                record["error_stage"] = "TTS"
                                record["error_text"] = str(e)
                            logger.error(
                                f"{self.name}: text-run TTS "
                                f"[{self._backend_desc(self.tts_backend)}] failed: {e}"
                            )
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

    async def serve_audio(self, mime: str, audio: bytes) -> tuple[str, str, int]:
        """Adapt the clip to a playable format and cache it; return (ext, url, nbytes).

        THE delivery boundary: TTS backends return their engine's NATIVE format
        (e.g. Piper -> audio/wav) and this single put-point adapts it to what
        the speaker firmware can decode (to_playable, run off-loop because the
        WAV->MP3 transcode blocks). The single adapt+put+tts_url path behind
        every audio-serving caller in the pipeline; also the public helper for
        device-layer callers that already hold ready audio bytes (e.g. the
        panel chime preview via play_media). `nbytes` is the size of the SERVED
        clip — what the speaker actually downloads (post-transcode).
        """
        mime, audio = await asyncio.to_thread(to_playable, mime, audio)
        audio_id = self.audio_server.put(audio, mime)
        ext, url = tts_url(self.public_base_url, audio_id, mime)
        return ext, url, len(audio)

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
        run's TTS audio. Preserves serve_audio_stream's ownership invariant: on early
        close (aclose), the underlying iterator is closed too."""
        try:
            async for chunk in chunks:
                sink.append(chunk)
                yield chunk
        finally:
            await Pipeline._close_stream(chunks)

    async def serve_audio_stream(self, mime, chunks):
        """Streaming delivery boundary: serve `chunks` (an async iterator of
        audio bytes) under a URL available IMMEDIATELY, while synthesis runs.

        Returns (ext, url, feed_task) where feed_task resolves to the total
        served bytes once the whole clip has been fed (or raises on a mid-stream
        synthesis failure — the URL was already handed out by then).

        OWNERSHIP INVARIANT: once called, this method owns the `chunks`
        iterator and guarantees it is always either fully consumed or
        explicitly closed. Streaming TTS backends open their HTTP response
        BEFORE returning the iterator and release it from the generator's
        finally, which only runs if the generator is iterated or closed —
        dropping the iterator un-iterated would leak the open response.

        A mime the speaker can't decode (e.g. Piper's WAV, which is unplayable
        from mid-stream anyway) falls back to the buffered path: drain the
        iterator fully and go through serve_audio (the transcode boundary); the
        returned awaitable is then already resolved to nbytes.
        """
        if mime not in PLAYABLE_MIMES:
            # Drain branch: an exception raised BY the iterator self-terminates
            # the generator, but one raised downstream (serve_audio after a
            # successful drain) would not touch it — close on ANY failure for a
            # uniform guarantee (aclose on a finished generator is a no-op).
            try:
                audio = b"".join([c async for c in chunks])
                ext, url, nbytes = await self.serve_audio(mime, audio)
            except BaseException:
                await self._close_stream(chunks)
                raise
            done = asyncio.get_running_loop().create_future()
            done.set_result(nbytes)
            return ext, url, done
        try:
            audio_id, feed = self.audio_server.put_stream(mime)
            ext, url = tts_url(self.public_base_url, audio_id, mime)

            async def _feed():
                total = 0
                try:
                    async for chunk in chunks:
                        total += len(chunk)
                        feed.write(chunk)
                except BaseException:
                    # Mid-stream failure (or cancellation): readers that got nothing
                    # see a 404, readers mid-send get a truncated clip; the caller
                    # awaits this task and records the error. The iterator either
                    # self-terminated (the error came from it) or is closed here
                    # (no-op when already finished) — the invariant holds.
                    feed.abort()
                    await Pipeline._close_stream(chunks)
                    raise
                feed.close()
                return total

            return ext, url, asyncio.create_task(_feed())
        except BaseException:
            # Anything failing between receiving the iterator and the feed task
            # taking ownership (put_stream, URL building, task creation) would
            # drop the iterator un-iterated; close it before propagating.
            await self._close_stream(chunks)
            raise

    async def speak(self, text: str) -> None:
        """Public entry for proactive speech (reminders, external callers).

        Synthesizes via the CURRENT tts stage, serves the clip, and plays it
        on the announcement channel (ducks current audio, plays while idle).
        The single text->speaker path shared by the filler and the device
        layer's announce. Raises on failure (callers decide isolation).
        """
        # Read the backend property ONCE: the same ref both synthesizes and
        # labels the log line, so they can never diverge.
        backend = self.tts_backend
        if self.core.audio.stream_tts:
            # Streaming: announce the stream URL as soon as synthesis starts;
            # the clip plays while the feed completes.
            mime, chunks = await self._synthesize_stream(backend, text, "ru")
            _ext, url, feed_task = await self.serve_audio_stream(mime, chunks)
            logger.info(
                f"{self.name}: 🔔 announce [tts={self._backend_desc(backend)}]: "
                f"{text!r} -> {url}"
            )
            try:
                if self.send_announcement is not None:
                    await self.send_announcement(media_id=url, timeout=30.0, text=text)
            except BaseException:
                # The announce itself failed (or was cancelled) BEFORE the feed
                # task was consumed: cancel a still-running feed (no point
                # finishing a synthesis nobody will play — and an instant
                # announce failure must not block on a long stream), then
                # await the task so it is never orphaned, suppressing ITS
                # failure (logged) so a feed error can't mask the original
                # announce error. Mirrors the orphan guard in _run.
                if not feed_task.done():
                    feed_task.cancel()
                try:
                    await feed_task
                except BaseException as fe:  # noqa: BLE001 - must not mask the announce error
                    logger.debug(f"{self.name}: announce TTS feed failed: {fe!r}")
                raise
            # Await the feed AFTER the announce returns (the announcement plays
            # while the feed completes) so a mid-stream synthesis failure still
            # surfaces to the caller and the task is never orphaned.
            await feed_task
            return
        mime, audio = await backend.synthesize(text, "ru")
        _ext, url, _nbytes = await self.serve_audio(mime, audio)
        logger.info(
            f"{self.name}: 🔔 announce [tts={self._backend_desc(backend)}]: "
            f"{text!r} -> {url}"
        )
        if self.send_announcement is not None:
            await self.send_announcement(media_id=url, timeout=30.0, text=text)

    async def _deliver_filler(self, text: str) -> None:
        """Speak an early 'filler' line on the announcement channel via speak().

        The announcement ducks any current audio and plays immediately, so the user
        hears a short 'I'll go check it' line while the slow tool + final LLM round
        run, instead of waiting in silence. Fully isolated: a synthesis/announce
        failure or its playback time must NEVER affect or delay the main run (which
        keeps its own final TTS_END for the real answer).
        """
        try:
            logger.info(f"{self.name}: 🗣️ filler: {text!r}")
            await self.speak(text)
        except Exception as e:
            logger.error(f"{self.name}: filler announce failed: {e}")

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

        Synchronous gate (no await before the decision): enabled toggle + a bound
        announcement channel. Building/serving the clip and the announce call all run
        inside the background task so STT is never delayed. Tracked in _ack_tasks so the
        still-playing announcement isn't garbage-collected; each task removes itself.
        """
        if not self.core.ack.enabled or self.send_announcement is None:
            return
        task = asyncio.create_task(self._play_ack())
        self._ack_tasks.add(task)
        task.add_done_callback(self._ack_tasks.discard)

    async def _play_ack(self) -> None:
        """Serve the ack clip through the audio cache and play it on the announce path.

        Mirrors _deliver_filler: ducks current audio and plays immediately. Fully
        isolated — a build/serve/announce failure must NEVER affect or delay the run
        (which keeps its own final TTS for the real answer).
        """
        try:
            mime, audio = self._ack_clip_bytes()
            _ext, url, _nbytes = await self.serve_audio(mime, audio)
            logger.info(f"{self.name}: 🔔 end-of-phrase ack -> {url}")
            if self.send_announcement is not None:
                await self.send_announcement(media_id=url, timeout=30.0, text="")
        except Exception as e:
            logger.error(f"{self.name}: end-of-phrase ack failed: {e}")

    async def _finish_capture(self, reason: str) -> None:
        """End a manual capture run: return the buffered PCM as WAV bytes, end the run.

        EPHEMERAL: NO STT/LLM/TTS and NOTHING written to disk. Claims the run
        synchronously (so a concurrent on_audio/on_stop can't double-finalize),
        builds the WAV bytes in memory and resolves the capture Future with them so
        DeviceClient.capture() can hand them back to the API caller. Empty audio
        fails the Future instead (so the caller surfaces an error rather than an
        empty WAV). Always emits RUN_END so the device returns to idle, and clears
        the per-run capture flag.
        """
        claimed = self._claim()
        if claimed is None:
            return  # Already finalized by a concurrent path.
        # Manual capture stays MONO: the second channel is discarded.
        pcm = claimed[0]
        self._capture_run = False
        logger.info(
            f"{self.name}: ⏺️ capture ended ({len(pcm)} bytes, "
            f"~{len(pcm) / (SAMPLE_RATE * 2):.1f}s, reason={reason})"
        )
        fut = self._capture_future
        self._capture_future = None
        if pcm:
            # Build the WAV container in memory (cheap, no disk) and hand it to the
            # awaiting caller. Resolve on the loop thread — this coroutine runs there.
            wav = pcm_to_wav_bytes(pcm)
            if fut is not None and not fut.done():
                fut.set_result(wav)
            logger.info(
                f"{self.name}: 🎧 captured sample -> {len(wav)} WAV bytes "
                f"(in-memory, {len(pcm)} PCM bytes)"
            )
        else:
            # No audio is a server-side capture failure, NOT an offline/missing-entity
            # condition — use a distinct type so the API maps it to HTTP 500, not 409.
            logger.info(f"{self.name}: capture produced no audio")
            if fut is not None and not fut.done():
                fut.set_exception(CaptureEmptyError("capture produced no audio"))
        # Return the device to idle. No STT/LLM/TTS events are sent for a capture run.
        self._emit(StageEvent.RUN_END, {})

    async def on_stop(self, abort: bool = False) -> None:
        """Explicit device stop: finalize the run (exactly once)."""
        if self._capture_run:
            await self._finish_capture("device_stop")
            return
        claimed = self._claim()
        if claimed is not None:
            await self._run("device_stop", claimed[0], self._conversation_id, pcm2=claimed[1])
