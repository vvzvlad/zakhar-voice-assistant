import io
import wave

import httpx
import pytest
import respx

import src.plugins  # noqa: F401  register all providers
from src.plugins.base import Deps, get_provider
from src.plugins.stt.groq import (
    GROQ_STT_URL,
    GroqSttBackend,
    GroqSttConfig,
    contains_stt_hallucination,
)
from src.plugins.stt.vosk import VoskSttBackend
from src.stage_errors import StageError
from src.stt import pcm_to_wav


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


def test_contains_stt_hallucination():
    assert contains_stt_hallucination("Субтитры создавал DimaTorzok")
    assert contains_stt_hallucination("dimatorzok")
    assert contains_stt_hallucination("Продолжение следует...")
    assert not contains_stt_hallucination("включи свет")


@respx.mock
async def test_groq_backend_discards_hallucination_as_empty():
    # A 200 carrying a known Whisper hallucination marker ("DimaTorzok"
    # subtitle-credit artifact) is discarded: transcribe returns "" — the
    # "no speech recognized" contract — so the pipeline ends the run like an
    # empty transcription.
    respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "Субтитры создавал DimaTorzok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="test-key", model="whisper-large-v3-turbo")
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == ""


def _multipart_field(body: str, name: str) -> str | None:
    """Extract one multipart form field's value from a decoded request body.

    Returns None when the field is absent. Matches the exact Content-Disposition
    field-name boundary (name="<name>") rather than a naive substring, so e.g. a
    prompt VALUE containing the word "prompt" can't confuse the check.
    """
    marker = f'name="{name}"\r\n\r\n'
    idx = body.find(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = body.find("\r\n--", start)
    assert end != -1, "multipart field value must be terminated by a boundary"
    return body[start:end]


@respx.mock
@pytest.mark.parametrize("prompt", ["", "   "])
async def test_groq_backend_omits_empty_prompt_field(prompt):
    # Empty / whitespace-only prompt -> NO "prompt" multipart field at all
    # (never send an empty prompt to the API).
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(
            client, api_key="k", model="whisper-large-v3-turbo", prompt=prompt
        )
        await backend.transcribe(b"\x01\x02" * 100)

    body = route.calls.last.request.content.decode("utf-8", "replace")
    assert _multipart_field(body, "prompt") is None
    # Sanity: the parser does see real fields in the same body.
    assert _multipart_field(body, "language") == "ru"


@respx.mock
async def test_groq_backend_sends_configured_prompt_and_bearer_auth():
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(
            client, api_key="test-key", model="whisper-large-v3-turbo",
            prompt="Захар, Wirenboard",
        )
        await backend.transcribe(b"\x01\x02" * 100)

    req = route.calls.last.request
    body = req.content.decode("utf-8", "replace")
    # The configured vocabulary hint travels as the exact "prompt" field value.
    assert _multipart_field(body, "prompt") == "Захар, Wirenboard"
    # The API key travels as a Bearer Authorization header.
    assert req.headers["Authorization"] == "Bearer test-key"


@respx.mock
async def test_groq_backend_truncates_overlong_prompt_to_896():
    from src.plugins.stt.groq import GROQ_PROMPT_MAX_CHARS

    # Pin the documented cap so a silent constant change fails this test.
    assert GROQ_PROMPT_MAX_CHARS == 896

    prompt = "Захар" + "а" * 995  # 1000 chars, distinctive head
    route = respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(
            client, api_key="k", model="whisper-large-v3-turbo", prompt=prompt
        )
        await backend.transcribe(b"\x01\x02" * 100)

    body = route.calls.last.request.content.decode("utf-8", "replace")
    # Exactly the first 896 characters are sent — no more, no less.
    assert _multipart_field(body, "prompt") == prompt[:896]


@respx.mock
@pytest.mark.parametrize("payload", [{}, {"text": "   "}])
async def test_groq_backend_missing_or_blank_text_returns_empty(payload):
    # A 200 without "text" (or with whitespace-only text) is "no speech
    # recognized" -> transcribe returns "".
    respx.post(GROQ_STT_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="k", model="whisper-large-v3-turbo")
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == ""


@respx.mock
async def test_groq_backend_malformed_200_body_raises_stage_error():
    # Д1 regression: a 200 whose body is not JSON makes resp.json() raise
    # json.JSONDecodeError (a ValueError, NOT an httpx.HTTPError). It must
    # surface as StageError("stt", ...) per the SttBackend contract, not leak raw.
    respx.post(GROQ_STT_URL).mock(
        return_value=httpx.Response(200, text="not json at all")
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = GroqSttBackend(client, api_key="k", model="whisper-large-v3-turbo")
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"
    assert "malformed" in str(ei.value)
    # The original decode error is chained for debuggability.
    assert isinstance(ei.value.__cause__, ValueError)


def test_contains_stt_hallucination_uppercase_cyrillic():
    # Markers are stored lowercase; matching must survive UPPERCASE Cyrillic
    # input — pins the casefold() normalization in contains_stt_hallucination.
    assert contains_stt_hallucination("ПРОДОЛЖЕНИЕ СЛЕДУЕТ...")
    assert contains_stt_hallucination("DIMATORZOK")


async def test_registry_groq_provider_creates_groq_backend():
    # REGISTRY-based construction (the primary path now): the groq STT provider's
    # create() returns a GroqSttBackend wired to the cloud HTTP client.
    async with httpx.AsyncClient(verify=False) as cloud, httpx.AsyncClient(verify=False) as local:
        deps = Deps(http_cloud=cloud, http_local=local)
        backend = get_provider("stt", "groq").create(GroqSttConfig(api_key="k"), deps)
        assert isinstance(backend, GroqSttBackend)
        assert backend.client is cloud
        assert backend.api_key == "k"


def test_registry_unknown_stt_provider_raises():
    with pytest.raises(ValueError):
        get_provider("stt", "nope")


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
