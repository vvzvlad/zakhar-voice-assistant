import io
import wave

import httpx
import pytest
import respx

from src.stage_errors import StageError
from src.stt import (
    GROQ_STT_URL,
    GroqSttBackend,
    VoskSttBackend,
    make_stt_backend,
    pcm_to_wav,
)


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
async def test_groq_backend_raises_stage_error_on_non_200():
    respx.post(GROQ_STT_URL).mock(return_value=httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"
    assert "500" in str(ei.value)


@respx.mock
async def test_groq_backend_raises_stage_error_on_transport_error():
    respx.post(GROQ_STT_URL).mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"


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


class _RecordingModel:
    """Stub Vosk model that records whether a recognizer was ever built from it."""

    def __init__(self):
        self.recognizer_requested = False


class _StubRecognizer:
    """Stub KaldiRecognizer returning a preset FinalResult() JSON string."""

    def __init__(self, final_result: str):
        self._final_result = final_result
        self.set_words_calls = []
        self.accepted = None

    def SetWords(self, value):  # noqa: N802 - mirror Vosk API
        self.set_words_calls.append(value)

    def AcceptWaveform(self, pcm):  # noqa: N802 - mirror Vosk API
        self.accepted = pcm

    def FinalResult(self):  # noqa: N802 - mirror Vosk API
        return self._final_result


async def test_vosk_empty_pcm_short_circuits_without_recognizer():
    # Inject a stub model and a recognizer factory that flags if it is ever called.
    model = _RecordingModel()
    backend = VoskSttBackend("unused/path", model=model)

    def _fail_factory():
        model.recognizer_requested = True
        raise AssertionError("recognizer must not be constructed for empty PCM")

    backend._make_recognizer = _fail_factory

    result = await backend.transcribe(b"")

    assert result == ""
    # The recognizer/model decode path was never entered for empty input.
    assert model.recognizer_requested is False


async def test_vosk_decode_extracts_and_strips_text():
    model = _RecordingModel()
    backend = VoskSttBackend("unused/path", model=model)

    rec = _StubRecognizer('{"text": "  привет  "}')
    backend._make_recognizer = lambda: rec

    result = await backend.transcribe(b"\x01\x02" * 100)

    # The "text" field is extracted and surrounding whitespace stripped.
    assert result == "привет"
    # The decode path actually drove the injected recognizer.
    assert rec.accepted == b"\x01\x02" * 100
    assert rec.set_words_calls == [False]


async def test_vosk_decode_missing_text_key_returns_empty():
    model = _RecordingModel()
    backend = VoskSttBackend("unused/path", model=model)

    backend._make_recognizer = lambda: _StubRecognizer('{"result": []}')

    result = await backend.transcribe(b"\x01\x02" * 100)

    # No "text" key in the recognizer result -> empty transcript.
    assert result == ""


class _RaisingRecognizer:
    """Stub recognizer whose AcceptWaveform raises (simulates a native decode crash)."""

    def SetWords(self, value):  # noqa: N802 - mirror Vosk API
        pass

    def AcceptWaveform(self, pcm):  # noqa: N802 - mirror Vosk API
        raise RuntimeError("native decode crashed")

    def FinalResult(self):  # noqa: N802 - mirror Vosk API
        return "{}"


async def test_vosk_decode_exception_raises_stage_error():
    # A recognizer crash must surface as StageError("stt", ...) per the SttBackend
    # contract — NOT escape as the raw RuntimeError (which the pipeline does not catch).
    backend = VoskSttBackend("unused/path", model=_RecordingModel())
    backend._make_recognizer = lambda: _RaisingRecognizer()

    with pytest.raises(StageError) as ei:
        await backend.transcribe(b"\x01\x02" * 100)

    assert ei.value.stage == "stt"
    assert "native decode crashed" in str(ei.value)
    # The original exception is chained (raise ... from e) for debuggability.
    assert isinstance(ei.value.__cause__, RuntimeError)


async def test_vosk_malformed_final_result_raises_stage_error():
    # FinalResult() returning non-JSON must surface as StageError("stt", ...),
    # not leak a raw json.JSONDecodeError out of transcribe().
    backend = VoskSttBackend("unused/path", model=_RecordingModel())
    backend._make_recognizer = lambda: _StubRecognizer("not-json{{{")

    with pytest.raises(StageError) as ei:
        await backend.transcribe(b"\x01\x02" * 100)

    assert ei.value.stage == "stt"
