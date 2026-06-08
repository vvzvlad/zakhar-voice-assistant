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

from src import context, llm
from src.runs_store import summary_row

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
        hub,
        stt_backend,
        llm_backend,
        tts_backend,
        audio_server,
        core,
        llm_cfg,
        runs_store=None,
        run_events=None,
    ):
        self.name = name
        # Multi-source tool hub: advertises smart-home + built-in tools (e.g. OpenWeatherMap)
        # to the model and executes them.
        self.hub = hub
        self.stt_backend = stt_backend
        self.llm_backend = llm_backend
        self.tts_backend = tts_backend
        self.audio_server = audio_server
        self.core = core
        self.llm_cfg = llm_cfg
        # Optional observability sink: a RunsStore (or compatible) whose insert()
        # is offloaded to a thread. None disables run recording.
        self.runs_store = runs_store
        # Optional live-broadcast hub: a RunEventsHub fanning out finalized runs to
        # connected panel WebSocket clients. None disables live broadcasting.
        self.run_events = run_events
        self.public_base_url = core.audio.public_base_url
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._conversation_id = ""
        self._context_path = os.path.join(core.context.dir, f"context_{name}.txt")

        # VAD end-pointing tunables (non-secret, defaulted in core.vad). Stored as
        # plain ints so tests can shrink the thresholds on the instance directly.
        self.vad_silence_ms = core.vad.silence_ms
        self.vad_min_speech_ms = core.vad.min_speech_ms
        self.vad_max_utterance_ms = core.vad.max_utterance_ms
        self.vad_no_speech_timeout_ms = core.vad.no_speech_timeout_ms
        # Instance attribute so tests can monkeypatch it with a fake exposing
        # is_speech(frame, rate) -> bool.
        self._vad = webrtcvad.Vad(core.vad.aggressiveness)

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
            if not pcm:
                # Truly-empty audio: nothing to transcribe and nothing to record.
                logger.info(f"{self.name}: empty audio, ending run")
                self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                return

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
                "model": None, "tokens": None,
                "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
                "error_stage": None, "error_text": None,
                "rounds": [],
            }
            try:
                stt_t = time.perf_counter()
                text = await self.stt_backend.transcribe(pcm)
                record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
                record["stt_text"] = text
                logger.info(
                    f"{self.name}: 📝 STT ({time.perf_counter() - stt_t:.2f}s): "
                    f"{text!r}"
                )
                self._emit(VAET.VOICE_ASSISTANT_STT_END, {"text": text})
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
                reply = await llm.call_llm_api(
                    self.llm_backend,
                    self.hub,
                    text,
                    core=self.core,
                    llm_cfg=self.llm_cfg,
                    history=history,
                    trace=trace,
                    device=self.name,
                )
                record["t_llm"] = int((time.perf_counter() - llm_t) * 1000)
                record["llm_text"] = reply
                record["model"] = trace.get("model")
                record["tokens"] = trace.get("tokens")
                record["rounds"] = trace.get("rounds") or []
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
                        "conversation_id": self._conversation_id,
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
                    ext = {
                        "audio/wav": "wav",
                        "audio/mpeg": "mp3",
                        "audio/flac": "flac",
                    }.get(mime, "mp3")
                    record["audio_bytes"] = len(audio)
                    record["audio_fmt"] = ext
                    audio_id = self.audio_server.put(audio, mime)
                    url = f"{self.public_base_url.rstrip('/')}/tts/{audio_id}.{ext}"
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
                        # Push the freshly recorded run to live panel subscribers.
                        # Fully isolated: a broadcast failure must never affect the run.
                        if self.run_events is not None:
                            try:
                                await self.run_events.broadcast(
                                    {"type": "run", "run": summary_row(record, run_id)}
                                )
                            except Exception as e:
                                logger.error(f"{self.name}: run broadcast failed: {e}")

    async def on_stop(self, abort: bool = False) -> None:
        """Explicit device stop: finalize the run (exactly once)."""
        pcm = self._claim()
        if pcm is not None:
            await self._run("device_stop", pcm)
