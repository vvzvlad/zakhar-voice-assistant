"""Text-to-speech backends (pluggable; TeraTTS HTTP is the default)."""

from abc import ABC, abstractmethod
from urllib.parse import quote

import httpx


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


def make_tts_backend(
    name: str, base_url: str, client: httpx.AsyncClient, timeout: int
) -> TtsBackend:
    """Construct a TTS backend by name."""
    if name == "teratts":
        return TeraTtsHttpBackend(base_url, client, timeout)
    raise ValueError(f"Unknown TTS backend: {name}")
