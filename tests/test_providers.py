import httpx
import pytest
import respx
from pydantic import ValidationError

import src.plugins  # noqa: F401  register all providers
from src.plugins.base import Deps, get_provider
from src.plugins.llm import groq as groq_mod
from src.plugins.llm import openrouter as openrouter_mod
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import LlmConfig
from src.plugins.llm.groq import GROQ_API_URL, GROQ_MODELS_URL, GroqLlmConfig
from src.plugins.llm.openrouter import OPENROUTER_API_URL, OPENROUTER_MODELS_URL
from src.plugins.stt import groq as stt_groq_mod
from src.plugins.stt import openrouter as stt_openrouter_mod
from src.plugins.stt.groq import GroqSttConfig
from src.plugins.stt.openrouter import OPENROUTER_STT_MODELS_URL, OpenRouterSttConfig
from src.plugins.stt.vosk import VoskSttConfig
from src.plugins.tts import fishaudio as fishaudio_mod
from src.plugins.tts.fishaudio import FISH_MODELS_URL, FishAudioTtsConfig
from src.plugins.tts.piper import PiperConfig
from src.plugins.tts.teratts import TeraTtsConfig
from src.plugins.tts.yandex import YandexTtsConfig


def _deps():
    return Deps(
        http_cloud=httpx.AsyncClient(),
        http_local=httpx.AsyncClient(),
        tts_timeout=30,
    )


# --- ConfigModel defaults / validation ---------------------------------------

def test_config_models_build_with_defaults():
    assert TeraTtsConfig().base_url == ""
    assert PiperConfig().sentence_silence == 0.4
    assert YandexTtsConfig().voice == "zahar"
    assert GroqSttConfig().model == "whisper-large-v3-turbo"
    assert VoskSttConfig().model_path == "models/vosk-model-small-ru-0.22"
    assert LlmConfig().model == "anthropic/claude-haiku-4.5"


def test_groq_stt_config_new_fields_defaults_and_types():
    cfg = GroqSttConfig()
    assert cfg.language == "ru"
    assert cfg.temperature == 0.0
    assert cfg.timeout == 60
    assert isinstance(cfg.temperature, float)
    assert isinstance(cfg.timeout, int)
    # Types validate/coerce from the JSON document.
    parsed = GroqSttConfig(language="en", temperature=1, timeout="30")
    assert parsed.temperature == 1.0 and isinstance(parsed.temperature, float)
    assert parsed.timeout == 30 and isinstance(parsed.timeout, int)


def test_llm_config_new_fields_defaults():
    cfg = LlmConfig()
    assert cfg.timeout == 300
    assert isinstance(cfg.timeout, int)
    assert cfg.reply_empty_after_tools == "Готово."
    assert cfg.reply_empty == "Я тебя не расслышал, повтори."
    assert cfg.reply_rate_limit == "Лимит запросов исчерпан. Попробуй ещё раз чуть позже."


def test_fishaudio_reference_id_label_companion_field():
    # The hidden companion field defaults to "" and is marked hidden in the schema
    # so the panel never renders it as its own input (it only persists the chosen
    # voice's human label for flicker-free first render).
    assert FishAudioTtsConfig().reference_id_label == ""
    props = FishAudioTtsConfig.model_json_schema()["properties"]
    assert props["reference_id_label"]["hidden"] is True


def test_llm_model_label_companion_field():
    # The hidden companion field defaults to "" and is marked hidden in the schema;
    # GroqLlmConfig inherits it unchanged (its model options are plain id strings,
    # so its label naturally equals the id).
    assert LlmConfig().model_label == ""
    props = LlmConfig.model_json_schema()["properties"]
    assert props["model_label"]["hidden"] is True
    assert GroqLlmConfig().model_label == ""


def test_yandex_speed_range_enforced():
    YandexTtsConfig(speed=2.5)            # in range, OK
    with pytest.raises(ValidationError):
        YandexTtsConfig(speed=9.9)        # > 3.0
    with pytest.raises(ValidationError):
        YandexTtsConfig(speed=0.0)        # < 0.1


def test_llm_temperature_and_tokens_ranges():
    LlmConfig(temperature=0.0, max_tokens=1, max_tool_rounds=1)
    with pytest.raises(ValidationError):
        LlmConfig(temperature=2.5)
    with pytest.raises(ValidationError):
        LlmConfig(max_tokens=0)
    with pytest.raises(ValidationError):
        LlmConfig(max_tool_rounds=0)


# --- options() ----------------------------------------------------------------

def test_yandex_options_voice_returns_list():
    prov = get_provider("tts", "yandex")
    voices = prov.options("voice", YandexTtsConfig(), _deps())
    assert isinstance(voices, list) and voices
    assert "zahar" in voices


def test_yandex_options_unknown_field_returns_none():
    prov = get_provider("tts", "yandex")
    assert prov.options("nope", YandexTtsConfig(), _deps()) is None


def test_yandex_options_role_depends_on_voice():
    prov = get_provider("tts", "yandex")
    deps = _deps()
    zahar_roles = prov.options("role", YandexTtsConfig(voice="zahar"), deps)
    assert zahar_roles == ["neutral", "good"]   # zahar has no "evil"
    jane_roles = prov.options("role", YandexTtsConfig(voice="jane"), deps)
    assert "evil" in jane_roles                 # jane does support "evil"


# --- LLM model-list options() (network-backed, TTL-cached) -------------------

@pytest.fixture
def _reset_model_caches():
    """Model lists are TTL-cached at module level; reset before AND after each test
    so test order never matters."""
    def reset():
        openrouter_mod._models_cache.update({"at": 0.0, "data": None})
        groq_mod._models_cache.update({"at": 0.0, "api_key": None, "data": None})
    reset()
    yield
    reset()


def test_llm_model_field_schema_is_dynamic_freeform_select():
    # The annotation must survive on BOTH classes: pydantic does not inherit Field
    # metadata on overridden fields, so GroqLlmConfig re-attaches it explicitly.
    for model_cls in (LlmConfig, GroqLlmConfig):
        prop = model_cls.model_json_schema()["properties"]["model"]
        assert prop["widget"] == "select"
        assert prop["options"] == "dynamic"
        assert prop["freeform"] is True


def test_llm_options_unknown_field_returns_none():
    deps = _deps()
    assert get_provider("llm", "openrouter").options("nope", LlmConfig(), deps) is None
    assert get_provider("llm", "groq").options("nope", GroqLlmConfig(), deps) is None


@respx.mock
async def test_openrouter_options_model_returns_sorted_value_label_list(_reset_model_caches):
    respx.get(OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(200, json={"data": [
        {"id": "z/zeta", "name": "Zeta"},
        {"id": "a/alpha", "name": "alpha Model"},
        {"id": "m/mid"},  # no "name" -> id doubles as the label
    ]}))
    prov = get_provider("llm", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", LlmConfig(), deps)
    # Sorted case-insensitively by label.
    assert out == [
        {"value": "a/alpha", "label": "alpha Model"},
        {"value": "m/mid", "label": "m/mid"},
        {"value": "z/zeta", "label": "Zeta"},
    ]


@respx.mock
async def test_openrouter_options_model_cache_hit_skips_second_fetch(_reset_model_caches):
    route = respx.get(OPENROUTER_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a/x", "name": "X"}]})
    )
    prov = get_provider("llm", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        first = await prov.options("model", LlmConfig(), deps)
        second = await prov.options("model", LlmConfig(), deps)
    assert first == second == [{"value": "a/x", "label": "X"}]
    assert route.call_count == 1


@respx.mock
async def test_openrouter_options_model_failure_is_not_cached(_reset_model_caches):
    route = respx.get(OPENROUTER_MODELS_URL)
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"data": [{"id": "a/x", "name": "X"}]}),
    ]
    prov = get_provider("llm", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        with pytest.raises(httpx.HTTPStatusError):
            await prov.options("model", LlmConfig(), deps)
        # The failed attempt left no cache entry; the retry refetches and succeeds.
        out = await prov.options("model", LlmConfig(), deps)
    assert out == [{"value": "a/x", "label": "X"}]
    assert route.call_count == 2


@respx.mock
async def test_groq_options_model_empty_api_key_returns_empty_without_request(_reset_model_caches):
    route = respx.get(GROQ_MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    prov = get_provider("llm", "groq")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", GroqLlmConfig(), deps)
    assert out == []
    assert route.call_count == 0  # the endpoint requires auth; no request was made


@respx.mock
async def test_groq_options_model_fetches_sorted_ids_with_bearer_auth(_reset_model_caches):
    route = respx.get(GROQ_MODELS_URL).mock(return_value=httpx.Response(200, json={
        "object": "list",
        "data": [{"id": "zzz-model"}, {"id": "Abc-model"}],
    }))
    prov = get_provider("llm", "groq")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", GroqLlmConfig(api_key="gsk-1"), deps)
    assert out == ["Abc-model", "zzz-model"]  # plain ids, case-insensitive sort
    assert route.calls.last.request.headers["Authorization"] == "Bearer gsk-1"


@respx.mock
async def test_groq_options_model_cache_is_keyed_by_api_key(_reset_model_caches):
    route = respx.get(GROQ_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m1"}]})
    )
    prov = get_provider("llm", "groq")
    deps = _deps()
    async with deps.http_cloud:
        await prov.options("model", GroqLlmConfig(api_key="gsk-1"), deps)
        await prov.options("model", GroqLlmConfig(api_key="gsk-1"), deps)  # cache hit
        assert route.call_count == 1
        # A different key must NOT be served the list fetched with the old key.
        await prov.options("model", GroqLlmConfig(api_key="gsk-2"), deps)
        assert route.call_count == 2
        assert route.calls.last.request.headers["Authorization"] == "Bearer gsk-2"


# --- STT model-list options() (network-backed, TTL-cached) -------------------

@pytest.fixture
def _reset_stt_model_caches():
    """STT model lists are TTL-cached at module level (own caches, never shared
    with the LLM providers); reset before AND after each test so order never matters."""
    def reset():
        stt_openrouter_mod._models_cache.update({"at": 0.0, "data": None})
        stt_groq_mod._models_cache.update({"at": 0.0, "api_key": None, "data": None})
    reset()
    yield
    reset()


def test_stt_model_field_schema_is_dynamic_freeform_select():
    # Both STT configs carry the shared MODEL_FIELD_EXTRA annotation on `model`.
    for model_cls in (GroqSttConfig, OpenRouterSttConfig):
        prop = model_cls.model_json_schema()["properties"]["model"]
        assert prop["widget"] == "select"
        assert prop["options"] == "dynamic"
        assert prop["freeform"] is True


def test_stt_options_unknown_field_returns_none():
    deps = _deps()
    assert get_provider("stt", "openrouter").options("nope", OpenRouterSttConfig(), deps) is None
    assert get_provider("stt", "groq").options("nope", GroqSttConfig(), deps) is None


@respx.mock
async def test_openrouter_stt_options_model_returns_sorted_value_label_list(_reset_stt_model_caches):
    respx.get(OPENROUTER_STT_MODELS_URL).mock(return_value=httpx.Response(200, json={"data": [
        {"id": "z/whisper-z", "name": "Zeta Whisper"},
        {"id": "a/whisper-a", "name": "alpha Whisper"},
        {"id": "m/whisper-m"},  # no "name" -> id doubles as the label
    ]}))
    prov = get_provider("stt", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", OpenRouterSttConfig(), deps)
    # Sorted case-insensitively by label.
    assert out == [
        {"value": "a/whisper-a", "label": "alpha Whisper"},
        {"value": "m/whisper-m", "label": "m/whisper-m"},
        {"value": "z/whisper-z", "label": "Zeta Whisper"},
    ]


@respx.mock
async def test_openrouter_stt_options_model_cache_hit_skips_second_fetch(_reset_stt_model_caches):
    route = respx.get(OPENROUTER_STT_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a/x", "name": "X"}]})
    )
    prov = get_provider("stt", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        first = await prov.options("model", OpenRouterSttConfig(), deps)
        second = await prov.options("model", OpenRouterSttConfig(), deps)
    assert first == second == [{"value": "a/x", "label": "X"}]
    assert route.call_count == 1


@respx.mock
async def test_openrouter_stt_options_model_failure_is_not_cached(_reset_stt_model_caches):
    route = respx.get(OPENROUTER_STT_MODELS_URL)
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"data": [{"id": "a/x", "name": "X"}]}),
    ]
    prov = get_provider("stt", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        with pytest.raises(httpx.HTTPStatusError):
            await prov.options("model", OpenRouterSttConfig(), deps)
        # The failed attempt left no cache entry; the retry refetches and succeeds.
        out = await prov.options("model", OpenRouterSttConfig(), deps)
    assert out == [{"value": "a/x", "label": "X"}]
    assert route.call_count == 2


@respx.mock
async def test_groq_stt_options_model_empty_api_key_returns_empty_without_request(_reset_stt_model_caches):
    route = respx.get(stt_groq_mod.GROQ_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    prov = get_provider("stt", "groq")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", GroqSttConfig(), deps)
    assert out == []
    assert route.call_count == 0  # the endpoint requires auth; no request was made


@respx.mock
async def test_groq_stt_options_model_filters_to_whisper_sorted(_reset_stt_model_caches):
    # Groq serves LLMs and whisper models in one list; only whisper ids
    # (case-insensitive substring) are valid for the transcriptions endpoint.
    route = respx.get(stt_groq_mod.GROQ_MODELS_URL).mock(return_value=httpx.Response(200, json={
        "object": "list",
        "data": [
            {"id": "llama-3.3-70b-versatile"},
            {"id": "whisper-large-v3-turbo"},
            {"id": "Whisper-large-v3"},
            {"id": "gemma2-9b-it"},
        ],
    }))
    prov = get_provider("stt", "groq")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("model", GroqSttConfig(api_key="gsk-1"), deps)
    assert out == ["Whisper-large-v3", "whisper-large-v3-turbo"]  # whisper only, ci-sorted
    assert route.calls.last.request.headers["Authorization"] == "Bearer gsk-1"


@respx.mock
async def test_groq_stt_options_model_cache_is_keyed_by_api_key(_reset_stt_model_caches):
    route = respx.get(stt_groq_mod.GROQ_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "whisper-1"}]})
    )
    prov = get_provider("stt", "groq")
    deps = _deps()
    async with deps.http_cloud:
        await prov.options("model", GroqSttConfig(api_key="gsk-1"), deps)
        await prov.options("model", GroqSttConfig(api_key="gsk-1"), deps)  # cache hit
        assert route.call_count == 1
        # A different key must NOT be served the list fetched with the old key.
        await prov.options("model", GroqSttConfig(api_key="gsk-2"), deps)
        assert route.call_count == 2
        assert route.calls.last.request.headers["Authorization"] == "Bearer gsk-2"


# --- TTS yandex create() returns the backend (no network) --------------------

def test_yandex_create_returns_backend():
    prov = get_provider("tts", "yandex")
    backend = prov.create(YandexTtsConfig(api_key="k"), _deps())
    assert backend.__class__.__name__ == "YandexTtsBackend"


# --- LLM OpenAICompat backend via respx --------------------------------------

@respx.mock
async def test_openai_compat_posts_with_auth_and_no_tools():
    route = respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(200, json={"choices": [], "model": "m", "usage": {}})
    )
    async with httpx.AsyncClient() as client:
        backend = OpenAICompatLlmBackend(
            url=OPENROUTER_API_URL, api_key="sk-test", model="m",
            temperature=0.5, max_tokens=10, client=client,
        )
        out = await backend.complete([{"role": "user", "content": "hi"}], None)

    assert out == {"choices": [], "model": "m", "usage": {}}
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = req.content.decode()
    assert '"tools"' not in body          # no tools -> not sent
    assert '"tool_choice"' not in body
    import json as _json
    assert _json.loads(body)["stream"] is False


@respx.mock
async def test_openai_compat_includes_tools_when_given():
    route = respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    tools = [{"type": "function", "function": {"name": "t"}}]
    async with httpx.AsyncClient() as client:
        backend = OpenAICompatLlmBackend(
            url=OPENROUTER_API_URL, api_key="k", model="m",
            temperature=0.5, max_tokens=10, client=client,
        )
        await backend.complete([{"role": "user", "content": "hi"}], tools)

    body = route.calls.last.request.content.decode()
    assert '"tools"' in body
    assert '"tool_choice"' in body


@respx.mock
async def test_openrouter_sends_x_title_header():
    route = respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    prov = get_provider("llm", "openrouter")
    deps = _deps()
    async with deps.http_cloud:
        backend = prov.create(LlmConfig(api_key="k"), deps)
        await backend.complete([{"role": "user", "content": "hi"}], None)
    assert route.calls.last.request.headers["X-Title"] == "Zakhar Voice Assistant"


@respx.mock
async def test_groq_provider_posts_to_groq_url_without_x_title():
    route = respx.post(GROQ_API_URL).mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    prov = get_provider("llm", "groq")
    deps = _deps()
    async with deps.http_cloud:
        backend = prov.create(prov.ConfigModel(api_key="k"), deps)
        await backend.complete([{"role": "user", "content": "hi"}], None)
    assert "X-Title" not in route.calls.last.request.headers


@respx.mock
async def test_openai_compat_raises_on_non_2xx():
    respx.post(OPENROUTER_API_URL).mock(return_value=httpx.Response(500))
    async with httpx.AsyncClient() as client:
        backend = OpenAICompatLlmBackend(
            url=OPENROUTER_API_URL, api_key="k", model="m",
            temperature=0.5, max_tokens=10, client=client,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await backend.complete([{"role": "user", "content": "hi"}], None)


# --- TTS fishaudio: config, options() and the voice catalog ------------------

@pytest.fixture
def _reset_fishaudio_voices_cache():
    """The voice catalog is TTL-cached at module level, keyed by api_key; clear
    before AND after each test so test order never matters."""
    fishaudio_mod._voices_cache.clear()
    yield
    fishaudio_mod._voices_cache.clear()


def test_fishaudio_config_defaults():
    cfg = FishAudioTtsConfig()
    assert cfg.api_key == ""
    assert cfg.reference_id == ""
    assert cfg.model == "s2-pro"
    assert cfg.speed == 1.0


def test_fishaudio_speed_range_enforced():
    FishAudioTtsConfig(speed=2.0)             # in range, OK
    FishAudioTtsConfig(speed=0.5)             # in range, OK
    with pytest.raises(ValidationError):
        FishAudioTtsConfig(speed=2.1)         # > 2.0
    with pytest.raises(ValidationError):
        FishAudioTtsConfig(speed=0.4)         # < 0.5


def test_fishaudio_options_model_returns_static_list():
    prov = get_provider("tts", "fishaudio")
    assert prov.options("model", FishAudioTtsConfig(), _deps()) == ["s1", "s2-pro"]


def test_fishaudio_options_unknown_field_returns_none():
    prov = get_provider("tts", "fishaudio")
    assert prov.options("nope", FishAudioTtsConfig(), _deps()) is None


def test_fishaudio_options_reference_id_empty_api_key_returns_empty_sync():
    # The catalog requires auth: with no api_key the option list is [] and no
    # coroutine (hence no network call) is produced.
    prov = get_provider("tts", "fishaudio")
    out = prov.options("reference_id", FishAudioTtsConfig(), _deps())
    assert out == []


@respx.mock
async def test_fishaudio_options_reference_id_merges_own_and_popular_voices(
        _reset_fishaudio_voices_cache):
    import inspect

    # Two GETs against the same catalog URL: own voices (self=true) first, then
    # the popular list; the merge keeps own-first order and dedups by _id.
    route = respx.get(FISH_MODELS_URL)
    route.side_effect = [
        httpx.Response(200, json={"total": 2, "items": [
            {"_id": "own-1", "title": "My Voice", "languages": ["ru", "en"]},
            {"_id": "own-2", "title": "", "languages": []},  # no title -> _id is the label
        ]}),
        httpx.Response(200, json={"total": 2, "items": [
            {"_id": "own-1", "title": "My Voice", "languages": ["ru", "en"]},  # dup, dropped
            {"_id": "pop-1", "title": "Popular Voice", "languages": ["en"]},
        ]}),
    ]
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        coro = prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps)
        assert inspect.isawaitable(coro)  # options() stays sync, returns a coroutine
        out = await coro
    assert out == [
        {"value": "own-1", "label": "My Voice [ru,en]"},
        {"value": "own-2", "label": "own-2"},
        {"value": "pop-1", "label": "Popular Voice [en]"},
    ]
    assert route.call_count == 2
    first, second = route.calls[0].request, route.calls[1].request
    assert first.headers["Authorization"] == "Bearer fk-1"
    assert second.headers["Authorization"] == "Bearer fk-1"
    assert first.url.params["self"] == "true"
    assert second.url.params["sort_by"] == "task_count"


@respx.mock
async def test_fishaudio_options_reference_id_cache_is_keyed_by_api_key(
        _reset_fishaudio_voices_cache):
    route = respx.get(FISH_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"total": 1, "items": [
            {"_id": "v1", "title": "V1", "languages": []},
        ]}))
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps)
        await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps)  # cache hit
        assert route.call_count == 2  # two catalog GETs once, none on the cache hit
        # A different key must NOT be served the list fetched with the old key.
        await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-2"), deps)
        assert route.call_count == 4
        assert route.calls.last.request.headers["Authorization"] == "Bearer fk-2"


@respx.mock
async def test_fishaudio_options_reference_id_failure_is_not_cached(
        _reset_fishaudio_voices_cache):
    route = respx.get(FISH_MODELS_URL)
    route.side_effect = [
        httpx.Response(500),  # own-voices request fails -> nothing cached
        httpx.Response(200, json={"total": 1, "items": [{"_id": "v1", "title": "V1"}]}),
        httpx.Response(200, json={"total": 0, "items": []}),
    ]
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        with pytest.raises(httpx.HTTPStatusError):
            await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps)
        # The failed attempt left no cache entry; the retry refetches and succeeds.
        out = await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps)
    assert out == [{"value": "v1", "label": "V1"}]
    assert route.call_count == 3


def test_fishaudio_create_returns_backend():
    prov = get_provider("tts", "fishaudio")
    backend = prov.create(FishAudioTtsConfig(api_key="k"), _deps())
    assert backend.__class__.__name__ == "FishAudioTtsBackend"


# --- TTS fishaudio: server-side voice search (query=...) ----------------------

def test_fishaudio_reference_id_schema_carries_remote_search_flag():
    # The frontend keys server-side search off this json_schema_extra flag.
    props = FishAudioTtsConfig.model_json_schema()["properties"]
    assert props["reference_id"]["search"] == "remote"


@respx.mock
async def test_fishaudio_options_reference_id_search_queries_catalog_by_title(
        _reset_fishaudio_voices_cache):
    import inspect

    route = respx.get(FISH_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"total": 2, "items": [
            {"_id": "s-1", "title": "Anna RU", "languages": ["ru"]},
            {"_id": "s-2", "title": "", "languages": []},  # no title -> _id is the label
        ]}))
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        coro = prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps, query="anna")
        assert inspect.isawaitable(coro)  # options() stays sync, returns a coroutine
        out = await coro
    assert out == [
        {"value": "s-1", "label": "Anna RU [ru]"},
        {"value": "s-2", "label": "s-2"},
    ]
    # Exactly ONE catalog GET (no own/popular pair), filtered server-side.
    assert route.call_count == 1
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer fk-1"
    assert req.url.params["title"] == "anna"
    assert req.url.params["page_size"] == "30"


@respx.mock
async def test_fishaudio_search_results_are_not_cached(_reset_fishaudio_voices_cache):
    # User-triggered searches are never cached: the same query twice issues two
    # HTTP requests.
    route = respx.get(FISH_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"total": 1, "items": [
            {"_id": "s-1", "title": "Anna RU", "languages": ["ru"]},
        ]}))
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps, query="anna")
        await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps, query="anna")
    assert route.call_count == 2


@respx.mock
async def test_fishaudio_options_reference_id_empty_query_keeps_baseline_path(
        _reset_fishaudio_voices_cache):
    # An explicit empty query is the no-search baseline: the merged own+popular
    # pair of requests, exactly as without the `query` argument.
    route = respx.get(FISH_MODELS_URL)
    route.side_effect = [
        httpx.Response(200, json={"total": 1, "items": [
            {"_id": "own-1", "title": "My Voice", "languages": ["ru"]},
        ]}),
        httpx.Response(200, json={"total": 1, "items": [
            {"_id": "pop-1", "title": "Popular Voice", "languages": ["en"]},
        ]}),
    ]
    prov = get_provider("tts", "fishaudio")
    deps = _deps()
    async with deps.http_cloud:
        out = await prov.options("reference_id", FishAudioTtsConfig(api_key="fk-1"), deps, query="")
    assert out == [
        {"value": "own-1", "label": "My Voice [ru]"},
        {"value": "pop-1", "label": "Popular Voice [en]"},
    ]
    assert route.call_count == 2
    assert route.calls[0].request.url.params["self"] == "true"
    assert route.calls[1].request.url.params["sort_by"] == "task_count"
