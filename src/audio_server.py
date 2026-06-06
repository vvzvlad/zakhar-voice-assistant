"""Shared aiohttp audio server: serves TTS MP3s to speakers from a TTL cache."""

import asyncio
import uuid

from aiohttp import web
from loguru import logger


class AudioServer:
    """One shared HTTP server for all speakers with an in-memory TTL cache.

    Speakers fetch generated audio from /tts/<id>.<ext> (e.g. .mp3 for TeraTTS,
    .wav for Piper). Each entry stores its own content type. Entries expire after
    `ttl` seconds (measured on the monotonic event-loop clock) and are pruned on put.
    """

    def __init__(self, host: str, port: int, ttl: int):
        self.host = host
        self.port = port
        self.ttl = ttl
        self._cache: dict[str, tuple[bytes, float, str]] = {}
        self._runner: web.AppRunner | None = None

    def put(self, data: bytes, content_type: str = "audio/mpeg") -> str:
        """Store audio bytes + content type, prune expired entries, return its id."""
        now = asyncio.get_running_loop().time()
        # Prune expired entries.
        expired = [k for k, (_, expiry, _) in self._cache.items() if expiry <= now]
        for k in expired:
            del self._cache[k]
        audio_id = uuid.uuid4().hex
        self._cache[audio_id] = (data, now + self.ttl, content_type)
        return audio_id

    async def _handle_tts(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        # Strip any trailing extension (.mp3/.wav/.flac) the speaker appends.
        if "." in name:
            name = name.rsplit(".", 1)[0]
        entry = self._cache.get(name)
        if entry is not None:
            data, expiry, content_type = entry
            if expiry > asyncio.get_running_loop().time():
                return web.Response(body=data, content_type=content_type)
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
