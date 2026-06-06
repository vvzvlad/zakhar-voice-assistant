"""Shared aiohttp audio server: serves TTS MP3s to speakers from a TTL cache."""

import asyncio
import uuid

from aiohttp import web
from loguru import logger


class AudioServer:
    """One shared HTTP server for all speakers with an in-memory TTL cache.

    Speakers fetch generated MP3s from /tts/<id>.mp3. Entries expire after `ttl`
    seconds (measured on the monotonic event-loop clock) and are pruned on put.
    """

    def __init__(self, host: str, port: int, ttl: int):
        self.host = host
        self.port = port
        self.ttl = ttl
        self._cache: dict[str, tuple[bytes, float]] = {}
        self._runner: web.AppRunner | None = None

    def put(self, data: bytes) -> str:
        """Store audio bytes, prune expired entries, return its id."""
        now = asyncio.get_running_loop().time()
        # Prune expired entries.
        expired = [k for k, (_, expiry) in self._cache.items() if expiry <= now]
        for k in expired:
            del self._cache[k]
        audio_id = uuid.uuid4().hex
        self._cache[audio_id] = (data, now + self.ttl)
        return audio_id

    async def _handle_tts(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if name.endswith(".mp3"):
            name = name[: -len(".mp3")]
        entry = self._cache.get(name)
        if entry is not None:
            data, expiry = entry
            if expiry > asyncio.get_running_loop().time():
                return web.Response(body=data, content_type="audio/mpeg")
        return web.Response(status=404)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def start(self) -> None:
        app = web.Application()
        app.add_routes(
            [
                web.get("/tts/{name}", self._handle_tts),
                web.get("/health", self._handle_health),
            ]
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"audio server on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
