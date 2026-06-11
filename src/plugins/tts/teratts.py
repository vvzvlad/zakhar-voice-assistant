"""TeraTTS HTTP brick (local service): config schema and backend."""

from urllib.parse import quote

import httpx
from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register
# The canonical LLM->TTS text is the model's own notation: plain text with "+"
# before the stressed vowel (e.g. "прив+ет"). The backend adapts that canon to
# its engine via the shared opt-in helpers.
from src.plugins.tts._ru_text import expand_units, phonetic_ru, stress_to_acute
from src.tts import TtsBackend


class TeraTtsHttpBackend(TtsBackend):
    """TeraTTS HTTP service backend. Returns MP3 (audio/mpeg)."""

    def __init__(self, base_url: str, client: httpx.AsyncClient, timeout: int):
        self.base_url = base_url
        self.client = client
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        # Same adaptation chain as Piper: TeraTTS historically received
        # Piper-style processed text, so this preserves its behavior.
        text = phonetic_ru(expand_units(stress_to_acute(text)))
        url = f"{self.base_url.rstrip('/')}/synthesize/{quote(text, safe='')}"
        resp = await self.client.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)


class TeraTtsConfig(BaseModel):
    base_url: str = ""


@register
class TeraTtsProvider(Provider):
    category = "tts"
    id = "teratts"
    label = "TeraTTS (HTTP)"
    ConfigModel = TeraTtsConfig

    def create(self, cfg: TeraTtsConfig, deps: Deps):
        return TeraTtsHttpBackend(cfg.base_url, deps.http_local, deps.tts_timeout)
