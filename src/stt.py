"""Speech-to-text backends (pluggable; cloud Whisper via the STT provider is the default)."""

import asyncio
import io
import json
import wave
from abc import ABC, abstractmethod

import httpx
from loguru import logger

from src.settings import settings

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def pcm_to_wav(
    pcm: bytes, sample_rate: int = 16000, sample_width: int = 2, channels: int = 1
) -> bytes:
    """Wrap raw PCM samples in a WAV container (pure, testable)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


class SttBackend(ABC):
    """Abstract STT backend: raw 16 kHz/16-bit/mono PCM -> transcript.

    Returns the transcript on success, "" on empty input/failure.
    """

    @abstractmethod
    async def transcribe(self, pcm: bytes) -> str:
        ...


class GroqSttBackend(SttBackend):
    """Groq Whisper HTTP backend. Posts a WAV-wrapped PCM and returns the text."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw 16 kHz/16-bit mono PCM via Groq Whisper.

        Returns the recognized text on success. On empty input, non-200 response
        or any httpx error returns "" (graceful degradation).
        """
        if not pcm:
            return ""

        wav_bytes = pcm_to_wav(pcm)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": settings.stt_model,
            "language": "ru",
            "response_format": "json",
            "temperature": "0",
        }
        headers = {"Authorization": f"Bearer {settings.stt_api_key}"}

        try:
            resp = await self.client.post(
                GROQ_STT_URL, headers=headers, data=data, files=files, timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("text", "").strip()
            logger.error(f"Groq STT error: {resp.status_code} - {resp.text}")
            return ""
        except httpx.HTTPError as e:
            logger.error(f"Groq STT request failed: {str(e)}")
            return ""


class VoskSttBackend(SttBackend):
    """In-process Vosk backend (small offline model, 16 kHz mono CPU decode).

    The model is loaded once and shared; KaldiRecognizer instances are spawned
    per call (the model is thread-safe for spawning recognizers). Decoding is
    blocking C code, so it runs in a worker thread.
    """

    def __init__(self, model_path: str):
        # Imported lazily so the heavy dependency/model are only required when the
        # Vosk backend is actually selected at runtime (never in tests/CI).
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self._model = Model(model_path)  # fail fast if the dir is missing

    def _decode(self, pcm: bytes) -> str:
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, 16000)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        # Speaker PCM is already 16 kHz mono 16-bit — no resampling needed.
        return await asyncio.to_thread(self._decode, pcm)


def make_stt_backend(name: str, client: httpx.AsyncClient, settings_obj) -> SttBackend:
    """Construct an STT backend by name."""
    if name == "groq":
        return GroqSttBackend(client)
    if name == "vosk":
        return VoskSttBackend(settings_obj.vosk_model_path)
    raise ValueError(f"Unknown STT backend: {name}")
