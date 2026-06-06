import io
import wave

import httpx
import respx

from src.stt import GROQ_STT_URL, pcm_to_wav, transcribe


def test_pcm_to_wav_roundtrip():
    pcm = b"\x01\x02" * 1600  # 3200 bytes -> 1600 frames at 16-bit mono
    wav_bytes = pcm_to_wav(pcm)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == len(pcm) // 2


@respx.mock
async def test_transcribe_returns_text_on_200():
    respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "привет мир"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        result = await transcribe(client, b"\x01\x02" * 100)
    assert result == "привет мир"


@respx.mock
async def test_transcribe_returns_empty_on_error():
    respx.post(GROQ_STT_URL).mock(return_value=httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(verify=False) as client:
        result = await transcribe(client, b"\x01\x02" * 100)
    assert result == ""


@respx.mock
async def test_transcribe_empty_pcm_skips_http():
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        result = await transcribe(client, b"")
    assert result == ""
    assert not route.called
