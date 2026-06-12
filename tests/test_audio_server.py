import asyncio

from aiohttp import web

from src.audio_server import AudioServer


class FakeRequest:
    """Minimal stand-in for web.Request: only match_info is read by the handler."""

    def __init__(self, name):
        self.match_info = {"name": name}


async def test_put_returns_stored_content_type_wav(tmp_path):
    server = AudioServer("127.0.0.1", 0, ttl=300)
    audio_id = server.put(b"RIFF....", "audio/wav")

    resp = await server._handle_tts(FakeRequest(f"{audio_id}.wav"))
    assert isinstance(resp, web.Response)
    assert resp.body == b"RIFF...."
    assert resp.content_type == "audio/wav"


async def test_put_default_mp3_mime_and_extension_strip(tmp_path):
    server = AudioServer("127.0.0.1", 0, ttl=300)
    # Default content type is audio/mpeg.
    audio_id = server.put(b"MP3")

    resp = await server._handle_tts(FakeRequest(f"{audio_id}.mp3"))
    assert resp.body == b"MP3"
    assert resp.content_type == "audio/mpeg"


async def test_unknown_id_returns_404():
    server = AudioServer("127.0.0.1", 0, ttl=300)
    resp = await server._handle_tts(FakeRequest("missing.mp3"))
    assert resp.status == 404


async def test_expired_entry_returns_404():
    server = AudioServer("127.0.0.1", 0, ttl=0)  # immediate expiry
    audio_id = server.put(b"MP3")
    # Let the monotonic clock advance past the (zero) TTL.
    await asyncio.sleep(0.01)
    resp = await server._handle_tts(FakeRequest(f"{audio_id}.mp3"))
    assert resp.status == 404


def _free_port() -> int:
    """Grab an OS-assigned free TCP port on the loopback and release it immediately."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _bound_port(server: AudioServer) -> int:
    """Port the running server is actually listening on."""
    return server._runner.addresses[0][1]


async def _fetch(session, port: int, path: str):
    async with session.get(f"http://127.0.0.1:{port}{path}") as resp:
        return resp.status, (await resp.read())


async def test_rebind_moves_port_and_preserves_cache():
    # Bind to one free port, cache an entry, then rebind to a SECOND free port. The
    # new port must serve and the previously put entry must still be fetchable. Two
    # distinct ports are reserved up front so the assertion is deterministic.
    import aiohttp

    old_port, new_port = _free_port(), _free_port()
    while new_port == old_port:           # exceedingly rare, but keep it deterministic
        new_port = _free_port()

    server = AudioServer("127.0.0.1", old_port, ttl=300)
    await server.start()
    try:
        assert _bound_port(server) == old_port
        audio_id = server.put(b"MP3-DATA")

        async with aiohttp.ClientSession() as session:
            status, body = await _fetch(session, old_port, f"/tts/{audio_id}.mp3")
            assert status == 200 and body == b"MP3-DATA"

            await server.rebind("127.0.0.1", new_port)
            assert _bound_port(server) == new_port

            # New socket serves, and the cache survived the rebind.
            status, body = await _fetch(session, new_port, f"/tts/{audio_id}.mp3")
            assert status == 200 and body == b"MP3-DATA"
    finally:
        await server.stop()


async def test_rebind_same_host_port_is_noop():
    # Rebinding to the identical host/port is a no-op: it must not raise and the
    # server keeps serving on the same port (runner is not torn down). Use a
    # CONCRETE reserved port so the no-op check is exercised on a real value, not
    # the port=0 coincidence.
    import aiohttp

    port = _free_port()
    server = AudioServer("127.0.0.1", port, ttl=300)
    await server.start()
    try:
        assert _bound_port(server) == port
        runner_before = server._runner
        audio_id = server.put(b"STILL-HERE")

        # Same host/port -> early return; the running socket is untouched.
        await server.rebind(server.host, server.port)
        assert server._runner is runner_before

        async with aiohttp.ClientSession() as session:
            status, body = await _fetch(session, port, f"/tts/{audio_id}.mp3")
            assert status == 200 and body == b"STILL-HERE"
    finally:
        await server.stop()


async def test_start_with_ephemeral_port_records_actual_port():
    # Constructed with port=0, start() must record the OS-assigned port on self.port
    # so rebind()'s no-op check and any port readers see the real value.
    server = AudioServer("127.0.0.1", 0, ttl=300)
    await server.start()
    try:
        assert server.port != 0
        assert server.port == _bound_port(server)
    finally:
        await server.stop()


# --- streaming entries (put_stream / StreamFeed) -----------------------------


async def test_stream_serves_chunks_written_before_and_after_connect():
    # A reader must get chunks written BEFORE it connected (replay from chunk 0)
    # and then follow chunks written AFTER, over a real chunked HTTP response.
    import aiohttp

    server = AudioServer("127.0.0.1", 0, ttl=300)
    await server.start()
    try:
        audio_id, feed = server.put_stream("audio/mpeg")
        feed.write(b"AAA")  # written before any reader connects

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{server.port}/tts/{audio_id}.mp3"
            ) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "audio/mpeg"
                # The pre-connect chunk arrives while the stream is still open.
                first = await resp.content.readexactly(3)
                # Feed more and close: the open response follows the feed.
                feed.write(b"BBBB")
                feed.close()
                rest = await resp.content.read()
        assert first + rest == b"AAABBBB"
    finally:
        await server.stop()


async def test_stream_fetch_after_close_serves_full_clip_from_cache():
    # After close() the complete clip becomes a regular byte-cache entry, so a
    # late (or repeated) fetch — e.g. a speaker retry — replays the full clip.
    server = AudioServer("127.0.0.1", 0, ttl=300)
    audio_id, feed = server.put_stream("audio/mpeg")
    feed.write(b"X")
    feed.write(b"YZ")
    feed.close()

    assert server._streams == {}  # the stream entry was retired into the cache
    resp = await server._handle_tts(FakeRequest(f"{audio_id}.mp3"))
    assert isinstance(resp, web.Response)
    assert resp.body == b"XYZ"
    assert resp.content_type == "audio/mpeg"


async def test_stream_two_concurrent_readers_both_get_full_clip():
    import aiohttp

    server = AudioServer("127.0.0.1", 0, ttl=300)
    await server.start()
    try:
        audio_id, feed = server.put_stream()
        url = f"http://127.0.0.1:{server.port}/tts/{audio_id}.mp3"
        feed.write(b"one-")

        async def read_all(session):
            async with session.get(url) as resp:
                assert resp.status == 200
                return await resp.read()

        async with aiohttp.ClientSession() as session:
            readers = [
                asyncio.create_task(read_all(session)),
                asyncio.create_task(read_all(session)),
            ]
            await asyncio.sleep(0.05)  # let both readers attach mid-stream
            feed.write(b"two")
            feed.close()
            bodies = await asyncio.gather(*readers)
        # Each reader has its own cursor and replays from chunk 0.
        assert bodies == [b"one-two", b"one-two"]
    finally:
        await server.stop()


async def test_stream_abort_before_any_read_returns_404():
    server = AudioServer("127.0.0.1", 0, ttl=300)
    audio_id, feed = server.put_stream()
    feed.write(b"partial")
    feed.abort()

    # The aborted entry is dropped — no cache handoff, nothing to serve.
    assert server._streams == {}
    resp = await server._handle_tts(FakeRequest(f"{audio_id}.mp3"))
    assert resp.status == 404


async def test_stream_reader_released_at_deadline_before_first_byte():
    # A reader already parked inside the wait loop must be released by its OWN
    # bounded wait once the deadline passes, even though the producer dies
    # silently (no close()/abort()) and nothing else triggers a prune. Before
    # any byte was sent the aborted stream surfaces as a clean 404.
    import aiohttp

    server = AudioServer("127.0.0.1", 0, ttl=1)  # short deadline: now + 1s
    await server.start()
    try:
        audio_id, feed = server.put_stream()
        # No chunks ever written, producer never closes: the reader connects
        # and waits. Generous 5s bound — the deadline fires after ~1s.

        async def fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{server.port}/tts/{audio_id}.mp3"
                ) as resp:
                    return resp.status, await resp.read()

        status, body = await asyncio.wait_for(fetch(), timeout=5)
        assert status == 404
        assert body == b""
        assert feed.failed is True   # treated as aborted at the deadline
        assert server._streams == {} # entry retired, not handed to the cache
    finally:
        await server.stop()


async def test_stream_reader_released_at_deadline_mid_send():
    # Same silent-producer scenario, but the reader already received a chunk:
    # at the deadline the response must END (truncated body), not hang — the
    # same quiet termination as a mid-send abort().
    import aiohttp

    server = AudioServer("127.0.0.1", 0, ttl=1)
    await server.start()
    try:
        audio_id, feed = server.put_stream()
        feed.write(b"head")  # the producer dies after the first chunk

        async def fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{server.port}/tts/{audio_id}.mp3"
                ) as resp:
                    assert resp.status == 200
                    return await resp.read()

        body = await asyncio.wait_for(fetch(), timeout=5)
        assert body == b"head"       # truncated at the dead producer's deadline
        assert feed.failed is True
        assert server._streams == {}
    finally:
        await server.stop()


async def test_stream_unclosed_past_ttl_treated_as_gone():
    # Safety net: a producer that never closes must not keep the entry alive
    # forever — past its deadline the stream is treated as aborted on access.
    server = AudioServer("127.0.0.1", 0, ttl=0)  # immediate deadline
    audio_id, feed = server.put_stream()
    feed.write(b"stuck")
    await asyncio.sleep(0.01)  # let the monotonic clock pass the deadline

    resp = await server._handle_tts(FakeRequest(f"{audio_id}.mp3"))
    assert resp.status == 404
    assert feed.failed is True   # retired as aborted, not handed to the cache
    assert server._streams == {}
