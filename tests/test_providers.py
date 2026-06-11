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
from src.plugins.stt.groq import GroqSttConfig
from src.plugins.stt.vosk import VoskSttConfig
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
