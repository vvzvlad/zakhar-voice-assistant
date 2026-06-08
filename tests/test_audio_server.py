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
