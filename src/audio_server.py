"""Shared aiohttp audio server: serves TTS MP3s to speakers from a TTL cache."""

import asyncio
import uuid

from aiohttp import web
from loguru import logger


# mime -> file extension for the /tts/<id>.<ext> URL the speaker fetches.
_EXT_FOR_MIME = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac"}


def tts_url(public_base_url: str, audio_id: str, mime: str) -> tuple[str, str]:
    """Return (ext, url) for a cached TTS audio id of the given mime type.

    Centralizes the mime->extension table and the /tts/<id>.<ext> URL shape so the
    pipeline and the announce path stay in sync.
    """
    ext = _EXT_FOR_MIME.get(mime, "mp3")
    url = f"{public_base_url.rstrip('/')}/tts/{audio_id}.{ext}"
    return ext, url


class AudioServer:
    """One shared HTTP server for all speakers with an in-memory TTL cache.

    Speakers fetch generated audio from /tts/<id>.<ext> (.mp3 for both TeraTTS and
    Piper). Each entry stores its own content type. Entries expire after
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
        self._runner = web.AppRunner(app, access_log=None)  # disable per-request access logs
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        # When bound to an OS-assigned port (0), record the actual port so rebind()'s
        # no-op check and any port readers see the real value.
        if self.port == 0 and self._runner is not None:
            addrs = self._runner.addresses
            if addrs:
                self.port = addrs[0][1]
        logger.info(f"audio server on {self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def rebind(self, host: str, port: int) -> None:
        """Re-bind the listening socket to a new host/port. No-op if unchanged.
        The in-memory TTL cache (self._cache) is preserved across the rebind."""
        if host == self.host and port == self.port:
            return
        await self.stop()
        self.host, self.port = host, port
        await self.start()
