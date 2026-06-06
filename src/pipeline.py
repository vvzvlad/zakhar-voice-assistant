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
import os
import time

import webrtcvad
from aioesphomeapi import VoiceAssistantEventType as VAET
from loguru import logger

from src import context, llm, stt
from src.settings import settings

# WebRTC VAD requires mono 16-bit PCM frames of exactly 10/20/30 ms at 16 kHz.
# We use 20 ms frames = 16000 * 2 * 20/1000 = 640 bytes.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = 640  # 16-bit mono, 20 ms @ 16 kHz

# Generous hard memory cap (~60s of 16 kHz / 16-bit mono PCM). VAD should end the
# utterance long before this; the cap only guards against unbounded growth if VAD
# misbehaves, and also forces a finalize when reached.
HARD_CAP_BYTES = SAMPLE_RATE * 2 * 60


class Pipeline:
    """Drives one voice run for a single speaker."""

    def __init__(
        self,
        name,
        client_ext,
        client_local,
        tts_backend,
        audio_server,
        public_base_url,
        context_dir,
    ):
        self.name = name
        self.client_ext = client_ext
        self.client_local = client_local
        self.tts_backend = tts_backend
        self.audio_server = audio_server
        self.public_base_url = public_base_url
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._conversation_id = ""
        self._context_path = os.path.join(context_dir, f"context_{name}.txt")

        # VAD end-pointing tunables (non-secret, defaulted in settings). Stored as
        # plain ints so tests can shrink the thresholds on the instance directly.
        self.vad_silence_ms = settings.vad_silence_ms
        self.vad_min_speech_ms = settings.vad_min_speech_ms
        self.vad_max_utterance_ms = settings.vad_max_utterance_ms
        self.vad_no_speech_timeout_ms = settings.vad_no_speech_timeout_ms
        # Instance attribute so tests can monkeypatch it with a fake exposing
        # is_speech(frame, rate) -> bool.
        self._vad = webrtcvad.Vad(settings.vad_aggressiveness)

        # Per-run VAD state (reset in on_start).
        self._frame_rem = bytearray()  # leftover bytes between non-640-aligned chunks
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_detected = False
        self._elapsed_ms = 0
        self._finalized = False
        # Logging-only flag: log "receiving audio" once per run, not per chunk.
        self._audio_logged = False

        # Injected by DeviceClient on connect (bound to the live API client).
        self.send_event = None
        self.send_audio = None

    def _emit(self, event_type, data=None):
        """Emit a voice_assistant event with a flat dict[str, str] payload."""
        if self.send_event is not None:
            self.send_event(
                event_type, {str(k): str(v) for k, v in (data or {}).items()}
            )

    async def on_start(
        self, conversation_id, flags, audio_settings, wake_word_phrase
    ) -> int:
        """Handle voice_assistant start: reset ALL per-run state, announce run."""
        self._conversation_id = conversation_id or ""
        self._buffer.clear()
        self._frame_rem.clear()
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_detected = False
        self._elapsed_ms = 0
        self._finalized = False
        self._audio_logged = False
        logger.info(f"{self.name}: ▶️ run started (cid={conversation_id})")
        self._emit(VAET.VOICE_ASSISTANT_RUN_START, {})
        self._emit(VAET.VOICE_ASSISTANT_STT_START, {})
        return 0  # 0 = audio comes in-band over the API connection.

    async def on_audio(self, data: bytes, data2=None) -> None:
        """Accumulate mic PCM and run VAD end-pointing. data2 is ignored.

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
        if reason is None:
            while len(self._frame_rem) >= FRAME_BYTES:
                frame = bytes(self._frame_rem[:FRAME_BYTES])
                del self._frame_rem[:FRAME_BYTES]
                speech = self._vad.is_speech(frame, SAMPLE_RATE)
                self._elapsed_ms += FRAME_MS
                if speech:
                    self._speech_ms += FRAME_MS
                    self._silence_ms = 0
                    if self._speech_ms >= self.vad_min_speech_ms:
                        self._speech_detected = True
                else:
                    # Trailing silence only counts once real speech has been observed.
                    if self._speech_detected:
                        self._silence_ms += FRAME_MS

        # Decide end-of-utterance (reason or None). The HARD_CAP "maxlen" set above
        # takes precedence; otherwise check VAD endpoint, max length, no-speech.
        if reason is None:
            if self._speech_detected and self._silence_ms >= self.vad_silence_ms:
                reason = "endpoint"
            elif self._elapsed_ms >= self.vad_max_utterance_ms:
                reason = "maxlen"
            elif (
                not self._speech_detected
                and self._elapsed_ms >= self.vad_no_speech_timeout_ms
            ):
                reason = "no_speech"

        # All synchronous state is updated above; claim+run last (the only await).
        # _claim() is synchronous and happens-before any await, so it's atomic
        # relative to other eager on_audio tasks — no audio loss, no double finalize.
        if reason is not None:
            pcm = self._claim()
            if pcm is not None:
                await self._run(reason, pcm)

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

    async def _run(self, reason, pcm) -> None:
        """Run STT -> LLM -> TTS -> events on the already-claimed audio, once.

        The caller claims the run via _claim() (which sets _finalized, snapshots
        and clears the buffer) and passes the snapshotted pcm here. The lock only
        provides defensive serialization; _claim() already guarantees single entry.
        RUN_END is always sent.
        """
        async with self._lock:
            # Total-run timer (logging only); started after the claim/lock.
            t0 = time.perf_counter()
            logger.info(
                f"{self.name}: 🎙️ captured {len(pcm)} bytes "
                f"(~{len(pcm) / (SAMPLE_RATE * 2):.1f}s), reason={reason}"
            )
            try:
                if not pcm:
                    logger.info(f"{self.name}: empty audio, ending run")
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                    return

                stt_t = time.perf_counter()
                text = await stt.transcribe(self.client_ext, pcm)
                logger.info(
                    f"{self.name}: 📝 STT ({time.perf_counter() - stt_t:.2f}s): "
                    f"{text!r}"
                )
                self._emit(VAET.VOICE_ASSISTANT_STT_END, {"text": text})
                if not text.strip():
                    logger.info(f"{self.name}: empty transcription, ending run")
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                    return

                self._emit(VAET.VOICE_ASSISTANT_INTENT_START, {})
                logger.info(f"{self.name}: 🤖 → LLM: {text!r}")
                llm_t = time.perf_counter()
                reply = await llm.call_groq_api(
                    self.client_ext, self.client_local, text
                )
                logger.info(
                    f"{self.name}: 💬 LLM reply ({time.perf_counter() - llm_t:.2f}s): "
                    f"{reply!r}"
                )

                try:
                    context.append_context(self._context_path, text, reply)
                except Exception as e:
                    # Context persistence failure must not break the run.
                    logger.error(f"{self.name}: context append failed: {e}")

                self._emit(
                    VAET.VOICE_ASSISTANT_INTENT_END,
                    {
                        "conversation_id": self._conversation_id,
                        "continue_conversation": "0",
                    },
                )

                self._emit(VAET.VOICE_ASSISTANT_TTS_START, {"text": reply})
                try:
                    tts_t = time.perf_counter()
                    mime, audio = await self.tts_backend.synthesize(reply, "ru")
                    logger.info(
                        f"{self.name}: 🔊 TTS ({time.perf_counter() - tts_t:.2f}s, "
                        f"{len(audio)} bytes)"
                    )
                    audio_id = self.audio_server.put(audio)
                    url = f"{self.public_base_url.rstrip('/')}/tts/{audio_id}.mp3"
                    logger.info(f"{self.name}: ▶ serving {url}")
                    self._emit(VAET.VOICE_ASSISTANT_TTS_END, {"url": url})
                except Exception as e:
                    # No TTS_END on failure; the run still ends cleanly.
                    logger.error(f"TTS failed: {e}")

                logger.info(
                    f"{self.name}: ✅ run complete in {time.perf_counter() - t0:.2f}s"
                )
                self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
            except Exception as e:
                logger.exception(f"{self.name}: pipeline run failed: {e}")
                self._emit(
                    VAET.VOICE_ASSISTANT_ERROR,
                    {"code": "server_error", "message": str(e)},
                )
                self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})

    async def on_stop(self, abort: bool = False) -> None:
        """Explicit device stop: finalize the run (exactly once)."""
        pcm = self._claim()
        if pcm is not None:
            await self._run("device_stop", pcm)
