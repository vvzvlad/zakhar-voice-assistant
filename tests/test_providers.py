import httpx
import pytest
import respx
from pydantic import ValidationError

import src.plugins  # noqa: F401  register all providers
from src.plugins.base import Deps, get_provider
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import LlmConfig
from src.plugins.llm.groq import GROQ_API_URL
from src.plugins.llm.openrouter import OPENROUTER_API_URL
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
    assert cfg.reply_empty == "Я тебя не расслышала, повтори."
    assert cfg.reply_rate_limit == (
        "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, "
        "обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии"
    )


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
