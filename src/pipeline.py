"""Per-speaker voice pipeline: mic PCM -> STT -> LLM -> TTS -> audio URL.

One Pipeline instance per speaker. It owns its own audio buffer and lock; the
brain, TTS backend and audio server are stateless shared dependencies injected
by the DeviceClient.
"""

import asyncio
import os

from aioesphomeapi import VoiceAssistantEventType as VAET
from loguru import logger

from src import context, llm, stt

# Safety cap on phrase length: ~30s of 16 kHz / 16-bit mono PCM.
MAX_UTTERANCE_BYTES = 16000 * 2 * 30


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
        self._over_cap_logged = False
        self._context_path = os.path.join(context_dir, f"context_{name}.txt")
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
        """Handle voice_assistant start: reset state and announce the run."""
        self._conversation_id = conversation_id or ""
        self._buffer.clear()
        self._over_cap_logged = False
        self._emit(VAET.VOICE_ASSISTANT_RUN_START, {})
        self._emit(VAET.VOICE_ASSISTANT_STT_START, {})
        return 0  # 0 = audio comes in-band over the API connection.

    async def on_audio(self, data: bytes, data2=None) -> None:
        """Accumulate incoming mic PCM up to the safety cap. data2 is ignored.

        Must stay synchronous (no await): the buffer is mutated lock-free here,
        which is safe only because this body runs atomically relative to the
        snapshot-and-clear in on_stop. Adding an await would introduce a race.
        """
        if len(self._buffer) < MAX_UTTERANCE_BYTES:
            remaining = MAX_UTTERANCE_BYTES - len(self._buffer)
            self._buffer.extend(data[:remaining])
        elif not self._over_cap_logged:
            self._over_cap_logged = True
            logger.warning(f"{self.name}: utterance exceeded cap, dropping audio")

    async def on_stop(self, abort: bool = False) -> None:
        """Run the pipeline on the buffered audio. RUN_END is always sent."""
        async with self._lock:
            pcm = bytes(self._buffer)
            self._buffer.clear()
            try:
                if not pcm:
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                    return

                text = await stt.transcribe(self.client_ext, pcm)
                self._emit(VAET.VOICE_ASSISTANT_STT_END, {"text": text})
                if not text.strip():
                    self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
                    return

                self._emit(VAET.VOICE_ASSISTANT_INTENT_START, {})
                reply = await llm.call_groq_api(
                    self.client_ext, self.client_local, text
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
                    mime, audio = await self.tts_backend.synthesize(reply, "ru")
                    audio_id = self.audio_server.put(audio)
                    url = f"{self.public_base_url.rstrip('/')}/tts/{audio_id}.mp3"
                    self._emit(VAET.VOICE_ASSISTANT_TTS_END, {"url": url})
                except Exception as e:
                    # No TTS_END on failure; the run still ends cleanly.
                    logger.error(f"TTS failed: {e}")

                self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
            except Exception as e:
                logger.exception(f"{self.name}: pipeline run failed: {e}")
                self._emit(
                    VAET.VOICE_ASSISTANT_ERROR,
                    {"code": "server_error", "message": str(e)},
                )
                self._emit(VAET.VOICE_ASSISTANT_RUN_END, {})
