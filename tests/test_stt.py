import io
import wave

import httpx
import pytest
import respx

from src.stt import GROQ_STT_URL, GroqSttBackend, make_stt_backend, pcm_to_wav


def test_pcm_to_wav_roundtrip():
    pcm = b"\x01\x02" * 1600  # 3200 bytes -> 1600 frames at 16-bit mono
    wav_bytes = pcm_to_wav(pcm)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == len(pcm) // 2


@respx.mock
async def test_groq_backend_returns_text_on_200():
    respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "привет мир"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == "привет мир"


@respx.mock
async def test_groq_backend_sends_configured_language_and_temperature():
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(
            client,
            api_key="test-key",
            model="whisper-large-v3-turbo",
            language="en",
            temperature=0.5,
            timeout=42,
        )
        await backend.transcribe(b"\x01\x02" * 100)

    req = route.calls.last.request
    # The multipart body carries the configured language and temperature.
    body = req.content.decode("utf-8", "replace")
    # Match the actual multipart field, not a stray "en" inside "Content-Disposition".
    assert 'name="language"\r\n\r\nen' in body
    assert "0.5" in body
    # The configured per-request timeout is applied to the POST.
    assert req.extensions["timeout"]["read"] == 42


@respx.mock
async def test_groq_backend_default_language_and_temperature():
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="k", model="whisper-large-v3-turbo")
        await backend.transcribe(b"\x01\x02" * 100)

    body = route.calls.last.request.content.decode("utf-8", "replace")
    assert "ru" in body            # default language
    assert "0.0" in body           # default temperature (str(0.0))
    assert route.calls.last.request.extensions["timeout"]["read"] == 60


@respx.mock
async def test_groq_backend_returns_empty_on_error():
    respx.post(GROQ_STT_URL).mock(return_value=httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == ""


@respx.mock
async def test_groq_backend_empty_pcm_skips_http():
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        result = await backend.transcribe(b"")
    assert result == ""
    assert not route.called


async def test_make_stt_backend_groq():
    async with httpx.AsyncClient(verify=False) as client:
        backend = make_stt_backend("groq", client, api_key="k", model="whisper-large-v3-turbo")
    assert isinstance(backend, GroqSttBackend)


async def test_make_stt_backend_unknown_raises():
    async with httpx.AsyncClient(verify=False) as client:
        with pytest.raises(ValueError):
            make_stt_backend("nope", client)
