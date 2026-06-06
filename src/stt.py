"""Groq Whisper speech-to-text client."""

import io
import wave

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


async def transcribe(client_ext: httpx.AsyncClient, pcm: bytes) -> str:
    """Transcribe raw 16 kHz/16-bit mono PCM via Groq Whisper.

    Returns the recognized text on success. On empty input, non-200 response or
    any httpx error returns "" (graceful degradation, like the original proxy).
    """
    if not pcm:
        return ""

    wav_bytes = pcm_to_wav(pcm)
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": settings.groq_stt_model,
        "language": "ru",
        "response_format": "json",
        "temperature": "0",
    }
    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}

    try:
        resp = await client_ext.post(
            GROQ_STT_URL, headers=headers, data=data, files=files, timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        logger.error(f"Groq STT error: {resp.status_code} - {resp.text}")
        return ""
    except httpx.HTTPError as e:
        logger.error(f"Groq STT request failed: {str(e)}")
        return ""
