"""Per-speaker voice pipeline: mic PCM -> STT -> LLM -> TTS -> audio URL.

One Pipeline instance per speaker. It owns its own audio buffer and lock; the
brain, TTS backend and audio server are stateless shared dependencies injected
by the DeviceClient.

Server-side VAD end-pointing
----------------------------
The HA Voice PE speaker fires its wake word on-device and then streams mic PCM
to us continuously (silence included). It NEVER signals end-of-speech, so the
device's handle_stop is effectively never called. To know when the user has
finished speaking we run WebRTC VAD over the incoming PCM ourselves and finalize
the utterance once we've seen speech followed by enough trailing silence (or a
hard max length, or a no-speech timeout). The explicit device-stop path still
finalizes too — both routes funnel through a synchronous _claim() (which atomically
marks the run finalized and snapshots the audio) followed by the async _run(), so a
run is finalized exactly once even under concurrent eager on_audio tasks.
"""

import asyncio
import io
import os
import time
import wave

import numpy as np
import webrtcvad
from aioesphomeapi import VoiceAssistantEventType as VAET
from loguru import logger

from src import config_store, context, llm
from src.audio_server import tts_url
from src.runs_store import live_row, summary_row
from src.text import processing_response
from src.tts import make_ack_chime_mp3, wav_to_mp3

# WebRTC VAD requires mono 16-bit PCM frames of exactly 10/20/30 ms at 16 kHz.
# We use 20 ms frames = 16000 * 2 * 20/1000 = 640 bytes.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = 640  # 16-bit mono, 20 ms @ 16 kHz

# Generous hard memory cap (~60s of 16 kHz / 16-bit mono PCM). VAD should end the
# utterance long before this; the cap only guards against unbounded growth if VAD
# misbehaves, and also forces a finalize when reached. This applies to NORMAL
# (wake-word) runs only; a manual capture sizes its own cap to the requested
# duration (see _capture_cap_bytes) so a long sample isn't truncated here.
HARD_CAP_BYTES = SAMPLE_RATE * 2 * 60

# Maximum duration (seconds) of a manual "record X seconds" capture. Single source
# of truth shared by the panel API / ESPHome client validation; the device-side
# template number caps at the same value in esphome/zakhar-voice.yaml.
CAPTURE_MAX_SECONDS = 300

# How long an armed manual capture stays valid waiting for its voice_assistant.start
# to arrive (the press -> start round-trip). If the button press is lost or the
# device never starts, the flag expires after this so it can't silently hijack a
# later real wake-word run. This is the arm-arrival window only — it is independent
# of (and much shorter than) the requested capture audio duration.
ARM_TTL = 5.0

# Known Whisper STT hallucination markers (lowercase). Whisper tends to emit
# leftover subtitle-credit / stock phrases (training-data artifacts) on
# silence/noise: "DimaTorzok" is one such credit string and "Продолжение
# следует..." ("to be continued") is a recurring stock caption. When a
# transcription contains one of these (substring, case-insensitive), we treat
# the run as if nothing was said and drop it.
STT_HALLUCINATION_MARKERS = ("dimatorzok", "продолжение следует")

# Tool-name substrings (case-insensitive) that mark a "slow" tool — one that hits
# the network / does a second LLM round and makes the user wait several seconds
# (web/google search, calendar, weather, wiki). Only these trigger the early
# spoken "filler"; instant smart-home actions (set_light, set_lock, set_scene,
# set_reminder, …) are intentionally excluded so the assistant doesn't talk twice
# for a trivial action. Extend this list to cover new slow tools.
SLOW_TOOL_MARKERS = (
    "google", "search", "web", "browse",
    "weather", "погод",
    "event", "calendar", "календар",
    "wiki",
)


def contains_stt_hallucination(text: str) -> bool:
    """Return True if the STT text contains a known Whisper hallucination marker.

    These are known STT (Whisper) hallucinations — subtitle-credit artifacts that
    surface on silence/noise — and are dropped as if nothing was said. The check is
    case-insensitive (Whisper varies the casing of the artifact).
    """
    folded = text.casefold()
    return any(marker in folded for marker in STT_HALLUCINATION_MARKERS)


def is_slow_tool(name: str) -> bool:
    """Return True if a tool name looks like a slow (network/think) tool — i.e.
    contains a SLOW_TOOL_MARKERS substring (case-insensitive). Used to decide
    whether to speak an early filler before the tool runs."""
    folded = (name or "").casefold()
    return any(marker in folded for marker in SLOW_TOOL_MARKERS)


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


def _write_wav(path: str, pcm: bytes) -> None:
    """Write 16 kHz / mono / 16-bit PCM to a WAV file at `path`."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Build a 16 kHz / mono / 16-bit WAV container from PCM, fully in memory.

    Used by the manual (ephemeral) capture path: the bytes are handed straight
    back to the API caller, so nothing is ever written to disk.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _highpass(pcm: bytes, cutoff_hz: float = 80.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """High-pass-filter 16-bit mono PCM via a numpy rFFT (no SciPy).

    Removes DC offset and low-frequency rumble (table thumps, HVAC) below ~cutoff_hz
    that carry no speech and skew normalization / hurt STT. The whole utterance is
    filtered at once with a smooth raised-cosine transition band (0 below 0.5*cutoff,
    1 above cutoff), so there is no per-chunk state and no SciPy dependency. Empty /
    too-short input is returned unchanged; a trailing odd byte is preserved.
    """
    if not pcm:
        return pcm
    n = len(pcm) - (len(pcm) % 2)  # whole int16 samples only
    if n < 4:
        return pcm
    x = np.frombuffer(pcm[:n], dtype="<i2").astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    lo = cutoff_hz * 0.5
    ramp = np.clip((freqs - lo) / (cutoff_hz - lo), 0.0, 1.0)
    mask = 0.5 - 0.5 * np.cos(np.pi * ramp)  # raised-cosine 0->1 across [lo, cutoff_hz]
    y = np.fft.irfft(spec * mask, n=x.size)
    filtered = np.clip(y, -32768, 32767).astype("<i2").tobytes()
    return filtered + pcm[n:]  # keep any trailing odd byte unchanged


def _normalize_peak(pcm: bytes, target_dbfs: float = -3.0, max_gain: float = 30.0) -> bytes:
    """Peak-normalize 16-bit mono PCM so its loudest sample hits ``target_dbfs``.

    A per-utterance adaptive replacement for a fixed gain: the quiet less-processed
    mic channel is brought to a consistent level without clipping, while already-loud
    samples scale down. ``max_gain`` caps the boost so a near-silent clip does not blow
    up the noise floor. Empty / near-silent input is returned unchanged; a trailing odd
    byte is preserved.
    """
    if not pcm:
        return pcm
    n = len(pcm) - (len(pcm) % 2)
    if n == 0:
        return pcm
    x = np.frombuffer(pcm[:n], dtype="<i2").astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1.0:
        return pcm  # silence — leave untouched
    target = 32767.0 * (10.0 ** (target_dbfs / 20.0))
    gain = min(target / peak, max_gain)
    boosted = np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()
    return boosted + pcm[n:]


# Target peak (~-3 dBFS) the VAD makeup gain lifts quiet audio toward, and the floor
# below which the running utterance peak is treated as "no real signal yet" (so leading
# pre-roll silence is NOT amplified into false speech).
# Target int16 peak the boost lifts the quiet channel's SPEECH toward. Deliberately
# MODERATE (~-15 dBFS, not full-scale): a higher target needs a larger gain, which also
# amplifies the channel's (clean, quiet) trailing silence enough that WebRTC VAD reads
# it as speech and never end-points (the utterance runs to max-length). At -15 dBFS the
# quiet channel's speech (peak ~50) reaches ~5800 (clearly speech) while its silence
# (peak ~5) stays ~600 (clearly non-speech), so the pause is still detected → fast end-point.
_VAD_BOOST_TARGET = 5824.0    # 32767 * 10**(-15/20)
# int16 peak below which we treat the (very clean) less-processed channel as pre-roll
# silence and don't boost — keeps leading noise from being amplified into false speech.
# Must sit BELOW the real speech level of the quiet channel (measured ~50-80) and ABOVE
# its silence floor (~1-5), so the boost engages once the wake word is heard.
_VAD_BOOST_FLOOR = 30


def _vad_boost(frame: bytes, peak: int, max_gain: float = 128.0) -> bytes:
    """Lift a 16-bit mono PCM frame toward _VAD_BOOST_TARGET for the VAD decision only.

    `peak` is the running peak of the WHOLE utterance so far (not this frame), so the
    gain (target/peak) is the same for every frame once the loud wake word has set the
    peak — this preserves the speech-vs-silence energy ratio (silence stays detectable)
    while bringing the quiet less-processed channel into WebRTC VAD's range. Returns the
    frame unchanged until a real signal has been seen (peak < floor) or when no boost is
    needed (gain <= 1). Used ONLY for the is_speech() decision — never stored.
    """
    if peak < _VAD_BOOST_FLOOR:
        return frame
    gain = min(_VAD_BOOST_TARGET / peak, max_gain)
    if gain <= 1.0:
        return frame
    n = len(frame) - (len(frame) % 2)
    if n == 0:
        return frame
    s = np.frombuffer(frame[:n], dtype="<i2").astype(np.float32) * gain
    return np.clip(s, -32768, 32767).astype("<i2").tobytes()


def _trim_start_pcm(pcm: bytes, trim_ms: int) -> bytes:
    """Drop the first ``trim_ms`` of 16 kHz / mono / 16-bit PCM from an utterance.

    Used to cut the wake-word tail / button-press lead-in off the captured sample
    before STT, so it does not pollute the transcription. The cut is sample-aligned
    (SAMPLE_RATE * 2 / 1000 = 32 bytes/ms is always even). If the trim would consume
    the whole sample (or more), the PCM is returned unchanged so we never hand empty
    audio to STT.
    """
    if trim_ms <= 0:
        return pcm
    trim_bytes = int(trim_ms * SAMPLE_RATE * 2 / 1000)
    if trim_bytes <= 0 or trim_bytes >= len(pcm):
        return pcm
    return pcm[trim_bytes:]


def build_ack_clip(sound_path: str, *, name: str = "") -> tuple[str, bytes]:
    """Build the end-of-phrase ack clip (mime, audio) from a sound_path.

    If sound_path points to an existing file, load it — transcoding a WAV to MP3 (the
    speaker firmware can't decode WAV; a bad/unreadable WAV falls back to the
    synthesized chime so playback never goes silent). An empty or missing path yields
    the synthesized two-tone chime. `name` is only used for log context. Does file IO
    and (for WAV) a blocking transcode, so callers on the event loop should run it via
    asyncio.to_thread. No caching here — callers cache as needed.
    """
    path = (sound_path or "").strip()
    use_file = bool(path) and os.path.isfile(path)
    if use_file:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext == "wav":
            try:
                return "audio/mpeg", wav_to_mp3(raw)
            except Exception as e:
                logger.warning(
                    f"{name}: ack chime transcode failed for {path}: {e}; "
                    f"using synthesized chime"
                )
                return "audio/mpeg", make_ack_chime_mp3()
        if ext == "flac":
            return "audio/flac", raw
        # mp3 (and any other) served verbatim as audio/mpeg
        return "audio/mpeg", raw
    return "audio/mpeg", make_ack_chime_mp3()


class Pipeline:
    """Drives one voice run for a single speaker."""

    def __init__(self, name, runtime):
        self.name = name
        # The mutable Runtime holder: all config and backends are read THROUGH it
        # per request (no frozen copies), so reconfiguration takes effect live.
        self.rt = runtime
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._conversation_id = ""

        # VAD aggressiveness is baked into the webrtcvad.Vad object, so the object
        # must be rebuilt when it changes (handled in on_start). Keep _vad a plain
        # instance attr so tests can monkeypatch it with a fake exposing
        # is_speech(frame, rate) -> bool. The other VAD thresholds are read live in
        # on_audio straight off self.rt.core.vad, so they are NOT copied here.
        self._vad_aggressiveness = self.rt.core.vad.aggressiveness
        self._vad = webrtcvad.Vad(self._vad_aggressiveness)

        # Per-run VAD state (reset in on_start).
        self._frame_rem = bytearray()  # leftover bytes between non-640-aligned chunks
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_detected = False
        self._elapsed_ms = 0
        self._vad_peak = 0              # running peak of the utterance (for VAD frame boost)
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
        """Emit a voice_assistant event with a flat dict[str, str] payload."""
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
        # webrtcvad.Vad bakes the aggressiveness in at construction, so rebuild the
        # object when the live config value has changed since we last built it.
        aggr = self.rt.core.vad.aggressiveness
        if aggr != self._vad_aggressiveness:
            self._vad = webrtcvad.Vad(aggr)
            self._vad_aggressiveness = aggr
        self._conversation_id = conversation_id or ""
        self._buffer.clear()
        self._frame_rem.clear()
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_detected = False
        self._elapsed_ms = 0
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
            self._emit(VAET.VOICE_ASSISTANT_RUN_START, {})
            return 0  # Capture-only: no STT_START, no VAD/STT/LLM/TTS.
        logger.info(f"{self.name}: ▶️ run started (cid={conversation_id})")
        self._emit(VAET.VOICE_ASSISTANT_RUN_START, {})
        self._emit(VAET.VOICE_ASSISTANT_STT_START, {})
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
        if self.core.vad.mic_channel == 1:
            if data2:
                data = data2
            elif not self._mic_fallback_logged:
                self._mic_fallback_logged = True
                logger.debug(
                    f"{self.name}: mic_channel=1 but device sent no second channel; "
                    f"using channel 0"
                )

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
        reason = None
        if len(self._buffer) >= HARD_CAP_BYTES:
            del self._buffer[HARD_CAP_BYTES:]
            reason = "maxlen"

        # Feed the VAD frame by frame. Device chunks aren't 640-aligned, so we keep
        # leftover bytes in _frame_rem and consume whole 640-byte frames only.
        # Skip the loop when HARD_CAP already set reason="maxlen": we're about to
        # finalize this chunk anyway, and the buffer was truncated, so there's no
        # point feeding (possibly truncated) bytes into the VAD counters.
        self._frame_rem.extend(data)
        # When mic_normalize is on, track the running peak of the (selected-channel)
        # utterance so the VAD frame boost below can lift this quiet channel into
        # WebRTC VAD's range. The buffer/STT path is unaffected (it normalizes per
        # utterance in _run).
        if self.core.vad.mic_normalize and data:
            m = len(data) - (len(data) % 2)
            if m:
                self._vad_peak = max(
                    self._vad_peak,
                    int(np.abs(np.frombuffer(data[:m], dtype="<i2")).max()),
                )
        if reason is None:
            while len(self._frame_rem) >= FRAME_BYTES:
                frame = bytes(self._frame_rem[:FRAME_BYTES])
                del self._frame_rem[:FRAME_BYTES]
                vad = self.rt.core.vad
                # Boost the frame for the speech/silence decision only (not the stored
                # buffer) so WebRTC VAD can detect the quiet less-processed channel.
                vframe = _vad_boost(frame, self._vad_peak) if vad.mic_normalize else frame
                speech = self._vad.is_speech(vframe, SAMPLE_RATE)
                self._elapsed_ms += FRAME_MS
                if speech:
                    self._speech_ms += FRAME_MS
                    self._silence_ms = 0
                    if self._speech_ms >= vad.min_speech_ms:
                        self._speech_detected = True
                else:
                    # Trailing silence only counts once real speech has been observed.
                    if self._speech_detected:
                        self._silence_ms += FRAME_MS

        # Decide end-of-utterance (reason or None). The HARD_CAP "maxlen" set above
        # takes precedence; otherwise check VAD endpoint, max length, no-speech.
        if reason is None:
            vad = self.rt.core.vad
            if self._speech_detected and self._silence_ms >= vad.silence_ms:
                reason = "endpoint"
            elif self._elapsed_ms >= vad.max_utterance_ms:
                reason = "maxlen"
            elif (
                not self._speech_detected
                and self._elapsed_ms >= vad.no_speech_timeout_ms
            ):
                reason = "no_speech"

        if reason is not None:
            logger.debug(
                f"{self.name}: VAD finalize reason={reason} speech_ms={self._speech_ms} "
                f"silence_ms={self._silence_ms} elapsed_ms={self._elapsed_ms} "
                f"speech_detected={self._speech_detected} vad_peak={self._vad_peak}"
            )

        # All synchronous state is updated above; claim+run last (the only await).
        # _claim() is synchronous and happens-before any await, so it's atomic
        # relative to other eager on_audio tasks — no audio loss, no double finalize.
        if reason is not None:
            pcm = self._claim()
            if pcm is not None:
                # Snapshot the conversation id synchronously with the claim so a
                # re-triggered wake word can't relabel this run's events.
                await self._run(reason, pcm, self._conversation_id)

    def _claim(self) -> bytes | None:
        """Atomically claim this run for finalization and snapshot the audio.

        Runs with no await, so concurrent eager on_audio tasks can't double-claim
        or lose audio. Returns the PCM to process, or None if already finalized.
        """
        if self._finalized:
            return None
        self._finalized = True
        pcm = bytes(self._buffer)
        self._buffer.clear()
        return pcm

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

    async def _run(self, reason, pcm, conversation_id) -> None:
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
                    f"(~{len(pcm) / (SAMPLE_RATE * 2):.1f}s), reason={reason}"
                )
                if not pcm:
                    # Truly-empty audio: nothing to transcribe and nothing to record.
                    logger.info(f"{self.name}: empty audio, ending run")
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
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
                # off core.vad so it hot-applies. _trim_start_pcm never returns empty audio,
                # so the non-empty guard above still holds afterwards. The manual
                # "record N seconds" capture-only path bypasses _run and is unaffected.
                trim_ms = self.core.vad.trim_start_ms
                if trim_ms > 0:
                    trimmed = _trim_start_pcm(pcm, trim_ms)
                    if len(trimmed) != len(pcm):
                        logger.info(
                            f"{self.name}: ✂️ trimmed {len(pcm) - len(trimmed)} bytes "
                            f"(~{trim_ms} ms) off sample start"
                        )
                    pcm = trimmed

                # Optional pre-STT conditioning of the FULL utterance (read live off
                # core.vad, so the toggles hot-apply). High-pass first to strip
                # DC/rumble, then peak-normalize so the quieter less-processed channel
                # reaches a consistent level without clipping. Applied ONCE to the whole
                # sample (never per chunk), so the capture WAV, the stored diagnostic
                # audio and STT all get the exact audio Whisper sees.
                if self.core.vad.mic_highpass:
                    pcm = _highpass(pcm)
                if self.core.vad.mic_normalize:
                    pcm = _normalize_peak(pcm)

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
                            _write_wav(p, data)

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
                record = {
                    "ts": time.time(),
                    "device": self.name,
                    "reason": reason,
                    "result": "empty",
                    "t_vad": int(len(pcm) / (SAMPLE_RATE * 2) * 1000),
                    "t_stt": 0, "t_llm": 0, "t_ruaccent": 0, "t_tts": 0,
                    "stt_text": "", "llm_text": "",
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
                        self._emit(VAET.VOICE_ASSISTANT_STT_END, {"text": ""})
                        self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                        return
                    stt_t = time.perf_counter()
                    text = await self.stt_backend.transcribe(pcm)
                    record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
                    logger.info(
                        f"{self.name}: 📝 STT ({time.perf_counter() - stt_t:.2f}s): "
                        f"{text!r}"
                    )
                    # Whisper emits leftover subtitle-credit artifacts (e.g.
                    # "DimaTorzok") on silence/noise. Blank such hallucinated text so it
                    # falls through into the empty-transcription branch below — the run
                    # ends exactly like an empty result (no LLM/TTS, recorded "empty").
                    if contains_stt_hallucination(text):
                        logger.info(
                            f"{self.name}: 🗑️ discarding STT hallucination: {text!r}"
                        )
                        text = ""
                    record["stt_text"] = text
                    self._emit(VAET.VOICE_ASSISTANT_STT_END, {"text": text})
                    # Live STT partial: surface the recognized text in the panel
                    # immediately, before the empty-transcription check (fires for
                    # both empty and non-empty text).
                    self._emit_live(record)
                    if not text.strip():
                        # Empty transcription: result stays "empty"; record in finally.
                        logger.info(f"{self.name}: empty transcription, ending run")
                        self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                        return

                    self._emit(VAET.VOICE_ASSISTANT_INTENT_START, {})
                    logger.info(f"{self.name}: 🤖 → LLM: {text!r}")
                    llm_t = time.perf_counter()
                    history = context.load_context(
                        self._context_path,
                        max_turns=self.core.context.max_turns,
                        ttl_seconds=self.core.context.ttl_seconds,
                    )
                    trace: dict = {}

                    filler_fired = False  # at most one early filler per run

                    async def _speak_filler(text: str, tool_names: list[str]) -> None:
                        # Policy lives here (llm.py is policy-free): speak at most once per
                        # run, and only for a SLOW tool (so instant smart-home actions don't
                        # double-talk). Schedule synthesis+announce as a fire-and-forget task
                        # so the slow tool is NOT delayed by filler TTS. Synchronous
                        # gate/dedup first (no await before we decide), then schedule.
                        nonlocal filler_fired
                        if filler_fired or self.send_announcement is None:
                            return
                        if not any(is_slow_tool(n) for n in tool_names):
                            return  # fast action: the final reply alone is enough
                        spoken = processing_response(text)  # same +stress/ё/што post-processing as the final reply
                        if not spoken:
                            return
                        filler_fired = True
                        record["filler_text"] = spoken
                        record["t_filler"] = int((time.perf_counter() - t0) * 1000)
                        task = asyncio.create_task(self._deliver_filler(spoken))
                        self._filler_tasks.add(task)
                        task.add_done_callback(self._filler_tasks.discard)

                    reply = await llm.call_llm_api(
                        self.llm_backend,
                        self.hub,
                        text,
                        core=self.core,
                        llm_cfg=self.llm_cfg,
                        history=history,
                        trace=trace,
                        device=self.name,
                        on_filler=_speak_filler,
                    )
                    record["t_llm"] = int((time.perf_counter() - llm_t) * 1000)
                    record["llm_text"] = reply
                    record["model"] = trace.get("model")
                    record["tokens"] = trace.get("tokens")
                    record["rounds"] = trace.get("rounds") or []
                    record["request"] = trace.get("request")
                    # Did the model actually run any tool this round?
                    tool_used = any(r.get("calls") for r in record["rounds"])
                    if reply.startswith("Ошибка:"):
                        # LLM-layer error (already a human-readable string). Classify it
                        # but keep current behavior: continue on to the TTS attempt.
                        record["result"] = "error"
                        record["error_stage"] = "LLM"
                        record["error_text"] = reply
                    else:
                        record["result"] = "tool" if tool_used else "ok"
                    logger.info(
                        f"{self.name}: 💬 LLM reply ({time.perf_counter() - llm_t:.2f}s): "
                        f"{reply!r}"
                    )
                    # Live LLM partial: record now carries llm_text, model, tokens,
                    # rounds, t_llm and the updated result.
                    self._emit_live(record)

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
                        VAET.VOICE_ASSISTANT_INTENT_END,
                        {
                            "conversation_id": conversation_id,
                            "continue_conversation": "0",
                        },
                    )

                    self._emit(VAET.VOICE_ASSISTANT_TTS_START, {"text": reply})
                    try:
                        tts_t = time.perf_counter()
                        mime, audio = await self.tts_backend.synthesize(reply, "ru")
                        record["t_tts"] = int((time.perf_counter() - tts_t) * 1000)
                        logger.info(
                            f"{self.name}: 🔊 TTS ({time.perf_counter() - tts_t:.2f}s, "
                            f"{len(audio)} bytes)"
                        )
                        audio_id = self.audio_server.put(audio, mime)
                        ext, url = tts_url(self.public_base_url, audio_id, mime)
                        record["audio_bytes"] = len(audio)
                        record["audio_fmt"] = ext
                        logger.info(f"{self.name}: ▶ serving {url}")
                        self._emit(VAET.VOICE_ASSISTANT_TTS_END, {"url": url})
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
                        logger.error(f"TTS failed: {e}")

                    logger.info(
                        f"{self.name}: ✅ run complete in {time.perf_counter() - t0:.2f}s"
                    )
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                except Exception as e:
                    record["result"] = "error"
                    record["error_stage"] = "pipeline"
                    record["error_text"] = str(e)
                    logger.exception(f"{self.name}: pipeline run failed: {e}")
                    self._emit(
                        VAET.VOICE_ASSISTANT_ERROR,
                        {"code": "server_error", "message": str(e)},
                    )
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                finally:
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
                            # Store the finalized utterance audio (the captured sample after
                            # the core.vad.trim_start_ms lead-in trim — exactly what STT
                            # received) in a rolling window of the last runs.audio_keep, so
                            # it can be downloaded/played from the log to diagnose
                            # mis-triggers (e.g. a wake-word tail reaching STT). Best-effort:
                            # a storage failure must never break the run or swallow the
                            # broadcast.
                            stored_audio = False
                            if self.core.runs.store_audio and pcm:
                                try:
                                    wav = _pcm_to_wav_bytes(pcm)
                                    await asyncio.to_thread(
                                        self.runs_store.put_audio,
                                        run_id, wav, self.core.runs.audio_keep,
                                    )
                                    stored_audio = True
                                except Exception as e:
                                    logger.error(
                                        f"{self.name}: utterance audio store failed: {e}"
                                    )
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

    async def _deliver_filler(self, text: str) -> None:
        """Synthesize an early 'filler' line and play it on the announcement channel.

        The announcement ducks any current audio and plays immediately, so the user
        hears a short 'I'll go check it' line while the slow tool + final LLM round
        run, instead of waiting in silence. Fully isolated: a synthesis/announce
        failure or its playback time must NEVER affect or delay the main run (which
        keeps its own final TTS_END for the real answer).
        """
        try:
            mime, audio = await self.tts_backend.synthesize(text, "ru")
            audio_id = self.audio_server.put(audio, mime)
            _ext, url = tts_url(self.public_base_url, audio_id, mime)
            logger.info(f"{self.name}: 🗣️ filler announce: {text!r} -> {url}")
            if self.send_announcement is not None:
                await self.send_announcement(media_id=url, timeout=30.0, text=text)
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
            audio_id = self.audio_server.put(audio, mime)
            _ext, url = tts_url(self.public_base_url, audio_id, mime)
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
        pcm = self._claim()
        if pcm is None:
            return  # Already finalized by a concurrent path.
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
            wav = _pcm_to_wav_bytes(pcm)
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
        self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})

    async def on_stop(self, abort: bool = False) -> None:
        """Explicit device stop: finalize the run (exactly once)."""
        if self._capture_run:
            await self._finish_capture("device_stop")
            return
        pcm = self._claim()
        if pcm is not None:
            await self._run("device_stop", pcm, self._conversation_id)
