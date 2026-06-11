"""Speech-to-text backends (pluggable; cloud Whisper via the STT provider is the default)."""

import asyncio
import io
import json
import wave
from abc import ABC, abstractmethod

import httpx
from loguru import logger

from src.stage_errors import StageError

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Groq rejects transcription prompts longer than this many characters (HTTP 400).
# Truncate locally so an over-long vocabulary hint degrades gracefully instead of
# failing the whole utterance.
GROQ_PROMPT_MAX_CHARS = 896


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

    Returns the transcript on success, "" when there is nothing to transcribe
    or no speech was recognized. Raises StageError("stt", ...) on failure.
    """

    @abstractmethod
    async def transcribe(self, pcm: bytes) -> str:
        ...


class GroqSttBackend(SttBackend):
    """Groq Whisper HTTP backend. Posts a WAV-wrapped PCM and returns the text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        language: str = "ru",
        temperature: float = 0.0,
        prompt: str = "",
        timeout: int = 60,
    ):
        self.client = client
        self.api_key = api_key
        self.model = model
        self.language = language
        self.temperature = temperature
        self.prompt = prompt
        self.timeout = timeout

    async def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw 16 kHz/16-bit mono PCM via Groq Whisper.

        Returns the recognized text on success; "" on empty input (nothing to
        transcribe) or a 200 with empty text (no speech recognized). Raises
        StageError("stt", ...) on a non-200 response or any httpx error.
        """
        if not pcm:
            return ""

        wav_bytes = pcm_to_wav(pcm)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.model,
            "language": self.language,
            "response_format": "json",
            "temperature": str(self.temperature),
        }
        # Bias recognition toward specific words only when a hint is provided;
        # never send an empty prompt to the API. Groq caps the prompt at
        # GROQ_PROMPT_MAX_CHARS characters, so truncate over-long hints (and warn)
        # rather than letting the API reject the whole request.
        prompt = self.prompt.strip()
        if len(prompt) > GROQ_PROMPT_MAX_CHARS:
            logger.warning(
                f"Groq STT prompt is {len(prompt)} chars; truncating to {GROQ_PROMPT_MAX_CHARS}"
            )
            prompt = prompt[:GROQ_PROMPT_MAX_CHARS]
        if prompt:
            data["prompt"] = prompt
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = await self.client.post(
                GROQ_STT_URL, headers=headers, data=data, files=files, timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json().get("text", "").strip()
            logger.error(f"Groq STT error: {resp.status_code} - {resp.text}")
            raise StageError("stt", f"Groq STT error: {resp.status_code} - {resp.text}")
        except httpx.HTTPError as e:
            logger.error(f"Groq STT request failed: {str(e)}")
            raise StageError("stt", f"Groq STT request failed: {e}") from e


class VoskSttBackend(SttBackend):
    """In-process Vosk backend (small offline model, 16 kHz mono CPU decode).

    The model is loaded once and shared; KaldiRecognizer instances are spawned
    per call (the model is thread-safe for spawning recognizers). Decoding is
    blocking C code, so it runs in a worker thread.
    """

    def __init__(self, model_path: str, *, model=None):
        # The model is injectable for testing. When None (production path), it is
        # loaded lazily so the heavy dependency/model are only required when the
        # Vosk backend is actually selected at runtime (never in tests/CI).
        if model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(model_path)  # fail fast if the dir is missing
        self._model = model

    def _make_recognizer(self):
        """Build a KaldiRecognizer for the shared model (overridable in tests)."""
        from vosk import KaldiRecognizer

        return KaldiRecognizer(self._model, 16000)

    def _decode(self, pcm: bytes) -> str:
        rec = self._make_recognizer()
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        # Speaker PCM is already 16 kHz mono 16-bit — no resampling needed.
        try:
            return await asyncio.to_thread(self._decode, pcm)
        except StageError:
            raise
        except Exception as e:
            # Honor the SttBackend contract: any decode failure (native Vosk error,
            # malformed recognizer JSON, ...) surfaces as StageError("stt", ...) so
            # the pipeline records the run as an STT error instead of dying raw.
            logger.error(f"Vosk STT decode failed: {str(e)}")
            raise StageError("stt", f"Vosk STT decode failed: {e}") from e


def make_stt_backend(
    name: str,
    client: httpx.AsyncClient,
    *,
    api_key: str = "",
    model: str = "whisper-large-v3-turbo",
    prompt: str = "",
    vosk_model_path: str = "models/vosk-model-small-ru-0.22",
) -> SttBackend:
    """Construct an STT backend by name (provider plugins are the primary path now)."""
    if name == "groq":
        return GroqSttBackend(client, api_key=api_key, model=model, prompt=prompt)
    if name == "vosk":
        return VoskSttBackend(vosk_model_path)
    raise ValueError(f"Unknown STT backend: {name}")
