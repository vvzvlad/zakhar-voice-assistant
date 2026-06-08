"""Text-to-speech backends (pluggable; TeraTTS HTTP is the default)."""

import asyncio
import io
import wave
from abc import ABC, abstractmethod
from urllib.parse import quote

import httpx
from loguru import logger

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


def wav_to_mp3(wav_bytes: bytes, bit_rate: int = 64, quality: int = 2) -> bytes:
    """Transcode a 16-bit PCM WAV (mono/stereo) to MP3 via lameenc.

    The speaker firmware can't decode WAV, so Piper output is served as MP3.
    """
    import lameenc  # local import: only needed when Piper is used

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_bit_rate(bit_rate)
    enc.set_quality(quality)
    return bytes(enc.encode(pcm) + enc.flush())


class PiperTtsBackend(TtsBackend):
    """In-process Piper backend (neural VITS via onnxruntime). Returns MP3.

    The voice is loaded once and shared (espeak-ng-data is bundled in the Piper
    package, so no system espeak is needed). Synthesis is blocking, so it runs in
    a worker thread. Piper produces WAV 22050 Hz mono 16-bit, which is then
    transcoded to MP3 (audio/mpeg) because the speaker firmware can't decode WAV.
    """

    def __init__(self, voice_path: str):
        # Imported lazily so the heavy dependency/model are only required when the
        # Piper backend is actually selected at runtime (never in tests/CI).
        from piper import PiperVoice

        # The config json sits next to the onnx at <path>.json.
        self._voice = PiperVoice.load(voice_path, voice_path + ".json")
        logger.info(f"Piper TTS voice loaded: {voice_path}")

    def _synth(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._voice.synthesize_wav(text, wf)
        # Transcode to MP3 here so the blocking lameenc call runs in the worker
        # thread (via asyncio.to_thread in synthesize), not on the event loop.
        return wav_to_mp3(buf.getvalue())

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        mp3_bytes = await asyncio.to_thread(self._synth, text)
        return ("audio/mpeg", mp3_bytes)


def make_tts_backend(
    name: str, base_url: str, client: httpx.AsyncClient, timeout: int
) -> TtsBackend:
    """Construct a TTS backend by name."""
    if name == "teratts":
        return TeraTtsHttpBackend(base_url, client, timeout)
    if name == "piper":
        return PiperTtsBackend(settings.piper_voice_path)
    raise ValueError(f"Unknown TTS backend: {name}")
