"""Shared aiohttp audio server: serves TTS audio to speakers.

Two kinds of entries live side by side:
- byte-cache entries (`put`): a complete clip stored with a TTL, served as a
  plain response — the classic buffer-then-serve path;
- streaming entries (`put_stream`): a `StreamFeed` the producer writes chunks
  into while synthesis is still running; readers get a chunked response that
  follows the feed. When the feed closes, the complete clip is moved into the
  byte cache (normal TTL), so late or repeated fetches (speaker retries)
  replay it through the existing path.
"""

import asyncio
import uuid

from aiohttp import web
from loguru import logger


# mime -> file extension for the /tts/<id>.<ext> URL the speaker fetches.
_EXT_FOR_MIME = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac"}


def ext_for_mime(mime: str) -> str:
    """File extension for a TTS audio mime; defaults to mp3 for unknown types."""
    return _EXT_FOR_MIME.get(mime, "mp3")


def tts_url(public_base_url: str, audio_id: str, mime: str) -> tuple[str, str]:
    """Return (ext, url) for a cached TTS audio id of the given mime type.

    Centralizes the mime->extension table and the /tts/<id>.<ext> URL shape so the
    pipeline and the announce path stay in sync.
    """
    ext = ext_for_mime(mime)
    url = f"{public_base_url.rstrip('/')}/tts/{audio_id}.{ext}"
    return ext, url


class StreamFeed:
    """Producer-side handle of a streaming audio entry.

    The producer write()s chunks as the TTS backend yields them, then close()s
    (or abort()s on failure); readers replay `chunks` from index 0 and wait for
    more via wait_change(). All methods are called from the event loop — no
    thread safety needed.
    """

    def __init__(self):
        self.chunks: list[bytes] = []
        self.done = False
        self.failed = False
        # Wake-up event, REPLACED on every change: a waiter snapshots the
        # current event and waits on it; any later change sets that snapshot,
        # so no wakeup is ever missed.
        self._event = asyncio.Event()
        # Set by AudioServer.put_stream: invoked once on close()/abort() so the
        # server can move the finished clip into the byte cache / drop the entry.
        self._on_done = None
        # Deadline on the event-loop clock past which an un-closed stream is
        # considered dead (set by AudioServer.put_stream; None = unbounded).
        # Enforced from the reader's wait loop (wait_change) as a safety net:
        # without it, a producer that dies silently — no close()/abort() and no
        # later put()/put_stream() to trigger a prune — would park readers
        # forever inside `await ev.wait()`.
        self.deadline: float | None = None

    def _wake(self) -> None:
        ev, self._event = self._event, asyncio.Event()
        ev.set()

    def write(self, chunk: bytes) -> None:
        """Append one audio chunk and wake readers. Empty chunks are ignored;
        writes after close()/abort() are ignored (defensive)."""
        if self.done or not chunk:
            return
        self.chunks.append(chunk)
        self._wake()

    def close(self) -> None:
        """Mark the stream complete: readers drain the remaining chunks and end."""
        if self.done:
            return
        self.done = True
        self._wake()
        if self._on_done is not None:
            self._on_done()

    def abort(self) -> None:
        """Mark the stream failed: readers that sent nothing yet 404, readers
        mid-send just end (truncated audio; the producer logged the cause)."""
        if self.done:
            return
        self.done = True
        self.failed = True
        self._wake()
        if self._on_done is not None:
            self._on_done()

    async def wait_change(self, seen: int) -> None:
        """Block until the feed has chunks beyond `seen` or is done. The event
        is snapshotted with no await between the check and the wait, so a write
        landing in between can never be missed (single-threaded event loop).

        Safety net: when a deadline is set, the wait is BOUNDED so the reader
        itself re-checks the deadline even when the producer never writes/
        closes again. On expiry the feed is treated as aborted — exactly the
        mid-send failure path: every waiter wakes, before-first-byte readers
        get a 404, mid-send readers end with a truncated body. The existing
        prune/entry-time checks stay; this covers readers already parked in
        the wait loop when the producer dies."""
        while not self.done and len(self.chunks) <= seen:
            ev = self._event
            if self.deadline is None:
                await ev.wait()
                continue
            remaining = self.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                # Overdue producer that never closed: abort sets done/failed,
                # wakes all readers and retires the entry via the on-done hook.
                self.abort()
                return
            try:
                # Clamp to >= 1s granularity: deadline detection may be up to
                # ~1s late, which is fine for a safety net and avoids hot loops.
                await asyncio.wait_for(ev.wait(), timeout=max(remaining, 1.0))
            except asyncio.TimeoutError:
                continue  # no change; loop re-checks done/deadline


class AudioServer:
    """One shared HTTP server for all speakers with an in-memory TTL cache.

    Speakers fetch generated audio from /tts/<id>.<ext>; the extension/content-type
    depends on the TTS backend. Each entry stores its own content type. Byte-cache entries expire after
    `ttl` seconds (measured on the monotonic event-loop clock) and are pruned on
    put/put_stream. Streaming entries (`put_stream`) are served as chunked
    responses that follow the producer's StreamFeed; on close the complete clip
    becomes a normal byte-cache entry (so retries replay it), on abort it is
    dropped. A stream never closed by its producer is treated as aborted once its
    deadline (registration time + ttl) passes.
    """

    def __init__(self, host: str, port: int, ttl: int):
        self.host = host
        self.port = port
        self.ttl = ttl
        self._cache: dict[str, tuple[bytes, float, str]] = {}
        # audio_id -> (feed, deadline, content_type) for in-flight streams.
        self._streams: dict[str, tuple[StreamFeed, float, str]] = {}
        self._runner: web.AppRunner | None = None

    def _prune(self, now: float) -> None:
        """Drop expired byte-cache entries and abort overdue un-closed streams
        (a producer that never closed must not keep the entry alive forever)."""
        expired = [k for k, (_, expiry, _) in self._cache.items() if expiry <= now]
        for k in expired:
            del self._cache[k]
        overdue = [
            feed
            for feed, deadline, _ in self._streams.values()
            if deadline <= now and not feed.done
        ]
        for feed in overdue:
            feed.abort()  # the on-done hook removes the entry from _streams

    def put(self, data: bytes, content_type: str = "audio/mpeg") -> str:
        """Store audio bytes + content type, prune expired entries, return its id."""
        now = asyncio.get_running_loop().time()
        self._prune(now)
        audio_id = uuid.uuid4().hex
        self._cache[audio_id] = (data, now + self.ttl, content_type)
        return audio_id

    def put_stream(self, content_type: str = "audio/mpeg") -> tuple[str, StreamFeed]:
        """Register a streaming entry and return (audio_id, feed) immediately —
        before any audio exists — so the URL can be handed to the speaker while
        synthesis runs. On feed.close() the complete clip moves into the byte
        cache with the normal TTL; on feed.abort() the entry is dropped."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        self._prune(now)
        audio_id = uuid.uuid4().hex
        feed = StreamFeed()

        def _finish():
            # Runs once, from feed.close()/abort(): hand the clip over to the
            # byte cache (or drop it on failure) and retire the stream entry.
            if self._streams.pop(audio_id, None) is None:
                return  # already pruned as overdue
            if not feed.failed:
                data = b"".join(feed.chunks)
                self._cache[audio_id] = (data, loop.time() + self.ttl, content_type)

        feed._on_done = _finish
        # The deadline lives both in _streams (for prune/entry checks) and on
        # the feed itself so readers parked in wait_change can enforce it.
        feed.deadline = now + self.ttl
        self._streams[audio_id] = (feed, now + self.ttl, content_type)
        return audio_id, feed

    async def _stream_entry_response(
        self, request: web.Request, feed: StreamFeed, content_type: str
    ) -> web.StreamResponse:
        """Serve one reader of an in-flight stream: replay from chunk 0, follow
        the feed (chunked transfer) until it is closed/aborted. Each concurrent
        reader gets its own cursor, so all of them see the full clip."""
        resp: web.StreamResponse | None = None
        sent = 0
        try:
            while True:
                while sent < len(feed.chunks):
                    chunk = feed.chunks[sent]
                    sent += 1
                    if resp is None:
                        resp = web.StreamResponse()
                        resp.content_type = content_type
                        await resp.prepare(request)  # no content-length -> chunked
                    await resp.write(chunk)
                if feed.done:
                    break
                await feed.wait_change(sent)
        except (ConnectionResetError, ConnectionError, OSError):
            # Client went away mid-stream: end quietly, the feed keeps running
            # for other readers / the cache handoff.
            return resp if resp is not None else web.Response(status=404)
        if feed.failed and resp is None:
            # Failed before anything was sent: a clean 404 (the speaker retries
            # or gives up); a mid-send failure just ends the (truncated) body.
            return web.Response(status=404)
        if resp is None:
            # Successfully closed but empty (unvoiceable text): empty 200 body.
            resp = web.StreamResponse()
            resp.content_type = content_type
            await resp.prepare(request)
        await resp.write_eof()
        return resp

    async def _handle_tts(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        # Strip any trailing extension (.mp3/.wav/.flac) the speaker appends.
        if "." in name:
            name = name.rsplit(".", 1)[0]
        now = asyncio.get_running_loop().time()
        entry = self._cache.get(name)
        if entry is not None:
            data, expiry, content_type = entry
            if expiry > now:
                return web.Response(body=data, content_type=content_type)
        stream = self._streams.get(name)
        if stream is not None:
            feed, deadline, content_type = stream
            if deadline <= now and not feed.done:
                # Overdue producer (never closed): treat the stream as aborted.
                feed.abort()
                return web.Response(status=404)
            return await self._stream_entry_response(request, feed, content_type)
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
        if self.port == 0:
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
