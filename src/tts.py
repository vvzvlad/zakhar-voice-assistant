"""Text-to-speech backends (pluggable; TeraTTS HTTP is the default)."""

import asyncio
import io
import wave
from abc import ABC, abstractmethod
from urllib.parse import quote

import httpx

from src.settings import settings


class TtsBackend(ABC):
    """Abstract TTS backend: text -> (mime, audio_bytes)."""

    @abstractmethod
    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        ...


class TeraTtsHttpBackend(TtsBackend):
    """TeraTTS HTTP service backend. Returns MP3 (audio/mpeg)."""

    def __init__(self, base_url: str, client: httpx.AsyncClient, timeout: int):
        self.base_url = base_url
        self.client = client
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        url = f"{self.base_url.rstrip('/')}/synthesize/{quote(text, safe='')}"
        resp = await self.client.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)


class PiperTtsBackend(TtsBackend):
    """In-process Piper backend (neural VITS via onnxruntime). Returns WAV.

    The voice is loaded once and shared (espeak-ng-data is bundled in the Piper
    package, so no system espeak is needed). Synthesis is blocking, so it runs in
    a worker thread. Output is WAV 22050 Hz mono 16-bit (audio/wav).
    """

    def __init__(self, voice_path: str):
        # Imported lazily so the heavy dependency/model are only required when the
        # Piper backend is actually selected at runtime (never in tests/CI).
        from piper import PiperVoice

        # The config json sits next to the onnx at <path>.json.
        self._voice = PiperVoice.load(voice_path, voice_path + ".json")

    def _synth(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)
        return buf.getvalue()

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        wav_bytes = await asyncio.to_thread(self._synth, text)
        return ("audio/wav", wav_bytes)


def make_tts_backend(
    name: str, base_url: str, client: httpx.AsyncClient, timeout: int
) -> TtsBackend:
    """Construct a TTS backend by name."""
    if name == "teratts":
        return TeraTtsHttpBackend(base_url, client, timeout)
    if name == "piper":
        return PiperTtsBackend(settings.piper_voice_path)
    raise ValueError(f"Unknown TTS backend: {name}")
