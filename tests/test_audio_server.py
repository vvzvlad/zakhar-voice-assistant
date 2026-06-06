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
