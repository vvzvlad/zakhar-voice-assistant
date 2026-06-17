import base64
import io
import json
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
from src.plugins.stt.openrouter import (
    OPENROUTER_STT_URL,
    OpenRouterSttBackend,
    OpenRouterSttConfig,
)
from src.plugins.stt.vosk import (
    VoskStreamSession,
    VoskSttBackend,
    VoskSttConfig,
    VoskSttProvider,
    _list_vosk_models,
)
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


def test_vosk_make_recognizer_uses_grammar_when_vocab_present(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("vosk")

    class FakeKaldi:
        def __init__(self, model, rate, *grammar):
            calls.append((rate, grammar))

    fake.KaldiRecognizer = FakeKaldi
    monkeypatch.setitem(sys.modules, "vosk", fake)
    backend = VoskSttBackend("unused/path", model=object(), vocabulary=lambda: ["привет", "захар"])
    backend._make_recognizer()
    # grammar arg is the JSON list of words + "[unk]"
    assert calls == [(16000, ('["привет", "захар", "[unk]"]',))]


def test_vosk_make_recognizer_no_grammar_when_vocab_none(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("vosk")

    class FakeKaldi:
        def __init__(self, model, rate, *grammar):
            calls.append((rate, grammar))

    fake.KaldiRecognizer = FakeKaldi
    monkeypatch.setitem(sys.modules, "vosk", fake)
    # No accessor at all -> plain full-vocab recognizer (no grammar arg).
    backend = VoskSttBackend("unused/path", model=object(), vocabulary=None)
    backend._make_recognizer()
    assert calls == [(16000, ())]


def test_vosk_make_recognizer_no_grammar_when_vocab_empty(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.ModuleType("vosk")

    class FakeKaldi:
        def __init__(self, model, rate, *grammar):
            calls.append((rate, grammar))

    fake.KaldiRecognizer = FakeKaldi
    monkeypatch.setitem(sys.modules, "vosk", fake)
    # An accessor returning [] means "no closed vocabulary" -> full-vocab recognizer.
    backend = VoskSttBackend("unused/path", model=object(), vocabulary=lambda: [])
    backend._make_recognizer()
    assert calls == [(16000, ())]


def test_vosk_current_vocab_swallows_errors_and_falls_back():
    # A vocabulary accessor that raises must NEVER break STT: _current_vocab returns
    # [] (full vocabulary) instead of propagating the error.
    def boom():
        raise RuntimeError("vocab source down")

    backend = VoskSttBackend("unused/path", model=object(), vocabulary=boom)
    assert backend._current_vocab() == []

    # A callable returning a list is passed through verbatim.
    backend2 = VoskSttBackend("unused/path", model=object(), vocabulary=lambda: ["свет"])
    assert backend2._current_vocab() == ["свет"]

    # No accessor -> [].
    backend3 = VoskSttBackend("unused/path", model=object(), vocabulary=None)
    assert backend3._current_vocab() == []


async def test_vosk_decode_strips_unk_sentinel():
    # The closed-vocab "[unk]" sentinel must never leak into the transcript fed to
    # the NLU: it is stripped from the decoded text.
    model = _RecordingModel()
    backend = VoskSttBackend("unused/path", model=model)
    backend._make_recognizer = lambda: _StubRecognizer('{"text": "включи [unk] свет"}')

    result = await backend.transcribe(b"\x01\x02" * 100)

    assert result == "включи свет"


def test_vosk_provider_create_wires_vocabulary_per_flag(monkeypatch):
    # VoskSttProvider.create() hands deps.command_vocabulary to the backend only when
    # restrict_to_nlu is on; otherwise vocabulary=None. Use a recording stub backend
    # so no real Vosk model is loaded.
    import src.plugins.stt.vosk as vosk_mod

    seen = []

    class _RecordingBackend:
        def __init__(self, model_path, *, model=None, vocabulary=None):
            seen.append((model_path, vocabulary))

    monkeypatch.setattr(vosk_mod, "VoskSttBackend", _RecordingBackend)

    cb = lambda: ["свет"]  # noqa: E731 - terse sentinel accessor
    deps = Deps(http_cloud=None, http_local=None, command_vocabulary=cb)

    vosk_mod.VoskSttProvider().create(VoskSttConfig(restrict_to_nlu=True), deps)
    vosk_mod.VoskSttProvider().create(VoskSttConfig(restrict_to_nlu=False), deps)

    assert seen[0][1] is cb        # flag on -> the live accessor is wired
    assert seen[1][1] is None      # flag off -> no vocabulary restriction


def test_vosk_backend_stores_vocabulary_accessor():
    # Backend-level wiring: the vocabulary accessor passed to the constructor is the
    # exact object the backend reads from (no real model loaded — model is injected).
    cb = lambda: ["захар"]  # noqa: E731
    assert VoskSttBackend("p", model=object(), vocabulary=cb)._vocabulary is cb


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


# --- Vosk streaming session (live decode during speech) -----------------------


class _StubStreamRecognizer:
    """Stub KaldiRecognizer for the streaming session. `accepts` scripts what each
    AcceptWaveform() call returns (True = internal endpoint -> Result(), False ->
    PartialResult()); shorter than the chunk count -> defaults to False. `results`
    is the queue of Result() JSON strings (one per True). `partial` and `final`
    back PartialResult()/FinalResult()."""

    def __init__(self, *, accepts=(), results=(), partial='{"partial": ""}',
                 final='{"text": ""}'):
        self._accepts = list(accepts)
        self._results = list(results)
        self._partial = partial
        self._final = final
        self.set_words_calls = []
        self._call = 0

    def SetWords(self, value):  # noqa: N802 - mirror Vosk API
        self.set_words_calls.append(value)

    def AcceptWaveform(self, pcm):  # noqa: N802 - mirror Vosk API
        is_final = self._accepts[self._call] if self._call < len(self._accepts) else False
        self._call += 1
        return is_final

    def Result(self):  # noqa: N802 - mirror Vosk API
        return self._results.pop(0)

    def PartialResult(self):  # noqa: N802 - mirror Vosk API
        return self._partial

    def FinalResult(self):  # noqa: N802 - mirror Vosk API
        return self._final


async def test_vosk_stream_finish_returns_final_with_unk_stripped():
    # Common short-utterance path: AcceptWaveform always False, the whole transcript
    # comes from FinalResult() at the end, with the "[unk]" sentinel stripped.
    rec = _StubStreamRecognizer(final='{"text": "включи [unk] свет"}')
    session = VoskStreamSession(rec)
    session.feed(b"\x01\x02" * 100)
    session.feed(b"\x03\x04" * 100)

    result = await session.finish()

    assert result == "включи свет"


async def test_vosk_stream_joins_midutterance_segment_and_tail():
    # An internal endpoint mid-utterance (AcceptWaveform True once -> Result segment),
    # then more audio, then the FinalResult tail: both segments join in order.
    rec = _StubStreamRecognizer(
        accepts=[True],  # first chunk endpoints, rest default to False
        results=['{"text": "включи свет"}'],
        final='{"text": "на кухне"}',
    )
    session = VoskStreamSession(rec)
    session.feed(b"\x01" * 100)  # -> Result segment "включи свет"
    session.feed(b"\x02" * 100)  # -> partial (ignored here)

    result = await session.finish()

    assert result == "включи свет на кухне"


async def test_vosk_stream_feed_after_finish_is_noop_and_aclose_idempotent():
    # feed() after finish() must be a no-op (never raise); aclose() is idempotent.
    rec = _StubStreamRecognizer(final='{"text": "привет"}')
    session = VoskStreamSession(rec)
    session.feed(b"\x01" * 100)

    result = await session.finish()
    assert result == "привет"

    # Post-finish feed is silently dropped.
    session.feed(b"\x05" * 100)  # must not raise

    # aclose() twice -> idempotent, never raises.
    await session.aclose()
    await session.aclose()


async def test_vosk_stream_decode_error_surfaces_as_stage_error():
    # A decode crash (AcceptWaveform raises) must surface from finish() as
    # StageError("stt", ...), not escape as the raw exception.
    class _Boom(_StubStreamRecognizer):
        def AcceptWaveform(self, pcm):  # noqa: N802
            raise RuntimeError("native decode crashed")

    session = VoskStreamSession(_Boom())
    session.feed(b"\x01" * 100)

    with pytest.raises(StageError) as ei:
        await session.finish()

    assert ei.value.stage == "stt"
    assert "native decode crashed" in str(ei.value)


async def test_vosk_open_stream_builds_session_with_recognizer():
    # open_stream() builds a fresh recognizer, disables word timings, and returns a
    # live streaming session routed through feed()/finish().
    backend = VoskSttBackend("unused/path", model=_RecordingModel())
    rec = _StubStreamRecognizer(final='{"text": ""}')
    backend._make_recognizer = lambda: rec

    session = backend.open_stream()
    try:
        assert isinstance(session, VoskStreamSession)
        assert rec.set_words_calls == [False]
    finally:
        await session.aclose()


# --- OpenRouter STT backend ----------------------------------------------------


@respx.mock
async def test_openrouter_stt_backend_posts_base64_wav_json():
    route = respx.post(OPENROUTER_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "привет мир"})
    )
    pcm = b"\x01\x02" * 100
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client,
            api_key="or-key",
            model="openai/whisper-large-v3-turbo",
            language="ru",
            temperature=0.3,
            timeout=42,
        )
        result = await backend.transcribe(pcm)
    assert result == "привет мир"

    req = route.calls.last.request
    # The API key travels as a Bearer Authorization header, plus the app title.
    assert req.headers["Authorization"] == "Bearer or-key"
    assert req.headers["X-Title"] == "Zakhar Voice Assistant"
    # The configured per-request timeout is applied to the POST.
    assert req.extensions["timeout"]["read"] == 42
    body = json.loads(req.content)
    assert body["model"] == "openai/whisper-large-v3-turbo"
    assert body["input_audio"]["format"] == "wav"
    # "data" is plain base64 (not a data URI) of the WAV-wrapped PCM.
    wav_bytes = base64.b64decode(body["input_audio"]["data"], validate=True)
    assert wav_bytes == pcm_to_wav(pcm)
    assert body["language"] == "ru"
    assert body["temperature"] == 0.3


@respx.mock
async def test_openrouter_stt_backend_omits_empty_language():
    # `language` is optional on the endpoint; a falsy config value must not be sent.
    route = respx.post(OPENROUTER_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo", language=""
        )
        await backend.transcribe(b"\x01\x02" * 100)
    body = json.loads(route.calls.last.request.content)
    assert "language" not in body


@respx.mock
async def test_openrouter_stt_backend_empty_pcm_skips_http():
    route = respx.post(OPENROUTER_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        result = await backend.transcribe(b"")
    assert result == ""
    assert not route.called


@respx.mock
@pytest.mark.parametrize("payload", [{}, {"text": "   "}])
async def test_openrouter_stt_backend_missing_or_blank_text_returns_empty(payload):
    # A 200 without "text" (or with whitespace-only text) is "no speech
    # recognized" -> transcribe returns "".
    respx.post(OPENROUTER_STT_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == ""


@respx.mock
async def test_openrouter_stt_backend_discards_hallucination_as_empty():
    # The OpenRouter catalog is whisper-family heavy, so the same subtitle-credit
    # artifacts apply: a 200 carrying a known hallucination marker is discarded
    # and transcribe returns "" — the "no speech recognized" contract.
    respx.post(OPENROUTER_STT_URL).mock(
        return_value=httpx.Response(200, json={"text": "Субтитры создавал DimaTorzok"})
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        result = await backend.transcribe(b"\x01\x02" * 100)
    assert result == ""


@respx.mock
async def test_openrouter_stt_backend_raises_stage_error_on_non_200():
    respx.post(OPENROUTER_STT_URL).mock(return_value=httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"
    assert "500" in str(ei.value)


@respx.mock
async def test_openrouter_stt_backend_raises_stage_error_on_transport_error():
    respx.post(OPENROUTER_STT_URL).mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"


@respx.mock
async def test_openrouter_stt_backend_malformed_200_body_raises_stage_error():
    # A 200 whose body is not JSON makes resp.json() raise json.JSONDecodeError
    # (a ValueError, NOT an httpx.HTTPError). It must surface as
    # StageError("stt", ...) per the SttBackend contract, not leak raw.
    respx.post(OPENROUTER_STT_URL).mock(
        return_value=httpx.Response(200, text="not json at all")
    )
    async with httpx.AsyncClient(verify=False) as client:
        backend = OpenRouterSttBackend(
            client, api_key="k", model="openai/whisper-large-v3-turbo"
        )
        with pytest.raises(StageError) as ei:
            await backend.transcribe(b"\x01\x02" * 100)
    assert ei.value.stage == "stt"
    assert "malformed" in str(ei.value)
    # The original decode error is chained for debuggability.
    assert isinstance(ei.value.__cause__, ValueError)


async def test_registry_openrouter_stt_provider_creates_backend():
    # REGISTRY-based construction: the openrouter STT provider's create() returns
    # an OpenRouterSttBackend wired to the cloud HTTP client.
    async with httpx.AsyncClient(verify=False) as cloud, httpx.AsyncClient(verify=False) as local:
        deps = Deps(http_cloud=cloud, http_local=local)
        backend = get_provider("stt", "openrouter").create(
            OpenRouterSttConfig(api_key="k"), deps
        )
        assert isinstance(backend, OpenRouterSttBackend)
        assert backend.client is cloud
        assert backend.api_key == "k"
        assert backend.model == "openai/whisper-large-v3-turbo"


# --- _list_vosk_models / VoskSttProvider.options (local model enumeration) ----


def _make_vosk_model(dir_path, name: str):
    """Create a fake Vosk model dir <name> with the structural am/ and conf/
    subdirectories that mark a standard model. Returns the model dir path."""
    model_dir = dir_path / name
    (model_dir / "am").mkdir(parents=True)
    (model_dir / "conf").mkdir(parents=True)
    return model_dir


def test_list_vosk_models_includes_dir_with_am_and_conf(tmp_path):
    m1 = _make_vosk_model(tmp_path, "m1")
    out = _list_vosk_models(str(tmp_path))
    assert out == [{"value": str(m1), "label": "m1"}]


def test_list_vosk_models_excludes_non_model_dir_and_plain_file(tmp_path):
    _make_vosk_model(tmp_path, "m1")
    (tmp_path / "notmodel").mkdir()  # missing am/ and conf/
    (tmp_path / "plain.txt").write_text("x")  # a file, not a dir
    out = _list_vosk_models(str(tmp_path))
    assert [o["label"] for o in out] == ["m1"]


def test_list_vosk_models_missing_dir_returns_empty(tmp_path):
    assert _list_vosk_models(str(tmp_path / "nope")) == []


def test_vosk_provider_options_model_path_scans_configured_dir(tmp_path):
    m1 = _make_vosk_model(tmp_path, "m1")
    # options() ignores deps (no network); pass None.
    out = VoskSttProvider().options("model_path", VoskSttConfig(model_path=str(m1)), None)
    assert out == [{"value": str(m1), "label": "m1"}]


def test_vosk_provider_options_other_field_returns_none(tmp_path):
    m1 = _make_vosk_model(tmp_path, "m1")
    assert VoskSttProvider().options("nope", VoskSttConfig(model_path=str(m1)), None) is None
