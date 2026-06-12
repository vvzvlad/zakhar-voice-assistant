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

    def _request_url(self, text: str) -> str:
        """Adapt the canonical text and build the GET URL. Same adaptation chain
        as Piper: TeraTTS historically received Piper-style processed text, so
        this preserves its behavior. Shared by synthesize and synthesize_stream."""
        text = phonetic_ru(expand_units(stress_to_acute(text)))
        return f"{self.base_url.rstrip('/')}/synthesize/{quote(text, safe='')}"

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        resp = await self.client.get(self._request_url(text), timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)

    async def synthesize_stream(self, text: str, lang: str = "ru"):
        """Native chunked synthesis: stream the GET body as it arrives. The mime
        comes from the response Content-Type header (fallback audio/mpeg); HTTP
        errors raise HERE, before the iterator is returned."""
        req = self.client.build_request(
            "GET", self._request_url(text), timeout=self.timeout
        )
        resp = await self.client.send(req, stream=True)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            await resp.aclose()  # always release the streamed response on error
            raise

        # The open streamed response is released by the generator's finally,
        # which only runs if the generator is iterated or aclose()d. The caller
        # (Pipeline.serve_audio_stream) owns the returned iterator and
        # guarantees exactly that — fully consumed or explicitly closed.
        async def _gen():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return (resp.headers.get("Content-Type", "audio/mpeg"), _gen())


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
