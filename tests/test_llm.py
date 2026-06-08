import httpx
import pytest
import respx

from src.core_config import CoreConfig, PromptConfig, WeatherConfig
from src.llm import call_llm_api
from src.plugins.llm.base import LlmConfig
from src.text import processing_response
from src.weather import OPENWEATHERMAP_URL

MAX_TOOL_ROUNDS = 5


class StubHub:
    """Tool hub double: advertises a fixed tool list and records call()s."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self.calls = []  # records (name, args)

    async def ensure_tools(self):
        return None

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return "ok"


class FakeLlmBackend:
    """LLM backend double: returns scripted JSON per complete() call.

    Each scripted item is either a dict (returned as the provider JSON) or an
    Exception instance (raised), so tests can drive both happy and error paths.
    Records the (messages, tools) of each call for assertions.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.seen = []  # list of (messages, tools)

    async def complete(self, messages, tools):
        self.seen.append((list(messages), tools))
        item = self._responses[self._i]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


SET_LIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "set_light",
        "description": "Turn a light on or off.",
        "parameters": {"type": "object", "properties": {"device_id": {"type": "string"}}},
    },
}


def _final(content):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "model": "x",
        "usage": {},
    }


def _tool_call(name, arguments_json):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_json},
                        }
                    ],
                }
            }
        ],
        "model": "x",
        "usage": {},
    }


def _http_status_error(status_code, json_body=None):
    """Build an httpx.HTTPStatusError carrying a Response with the given status."""
    request = httpx.Request("POST", "https://llm.test/chat")
    response = httpx.Response(status_code, json=json_body, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def _core(tmp_path):
    """A CoreConfig whose prompt file lives in tmp_path (so the real
    build_system_prompt reads it without touching data/)."""
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("PROMPT BODY <<<<<TDW>>>>>", encoding="utf-8")
    return CoreConfig(
        prompt=PromptConfig(system_prompt_path=str(prompt_path)),
        weather=WeatherConfig(api_key="w-key", city="Moscow"),
    )


def _mock_weather_404():
    """Mock OpenWeatherMap to 404 so weather is omitted from the prompt (deterministic)."""
    respx.get(OPENWEATHERMAP_URL).mock(return_value=httpx.Response(404, json={}))


async def _call(backend, hub, text, core, *, history=None, max_tool_rounds=MAX_TOOL_ROUNDS):
    async with httpx.AsyncClient() as weather_client:
        return await call_llm_api(
            backend,
            hub,
            text,
            weather_client=weather_client,
            core=core,
            llm_cfg=LlmConfig(max_tool_rounds=max_tool_rounds),
            history=history,
        )


@respx.mock
async def test_tool_path(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", '{"device_id":"bright_room_light","state":"on"}'),
        _final("Готово."),
    ])

    result = await _call(backend, hub, "включи свет", _core(tmp_path))

    assert hub.calls == [
        ("set_light", {"device_id": "bright_room_light", "state": "on"})
    ]
    # processing_response is applied to the final content.
    assert result == processing_response("Готово.")


@respx.mock
async def test_no_tool_path(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([_final("Привет, мясной мешок.")])

    result = await _call(backend, hub, "привет", _core(tmp_path))

    assert hub.calls == []
    assert result == processing_response("Привет, мясной мешок.")


@respx.mock
async def test_rate_limit_path(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(429)])

    result = await _call(backend, hub, "привет", _core(tmp_path))

    assert result == LlmConfig().reply_rate_limit


@respx.mock
async def test_non_2xx_returns_error_message(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(500, {"error": {"message": "boom"}})])

    result = await _call(backend, hub, "привет", _core(tmp_path))

    assert result == "Ошибка: boom"


@respx.mock
async def test_httpx_error_returns_error_prefix(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([httpx.ConnectError("down")])

    result = await _call(backend, hub, "привет", _core(tmp_path))

    assert result.startswith("Ошибка:")


@respx.mock
async def test_max_tool_rounds_exhausted(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}") for _ in range(MAX_TOOL_ROUNDS + 1)
    ])

    result = await _call(backend, hub, "включи свет", _core(tmp_path))

    assert result == "Ошибка: слишком много вызовов инструментов"


@respx.mock
async def test_empty_final_reply_uses_fallback(tmp_path):
    # No tool ever ran -> empty final content falls back to the "didn't hear" line.
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final(None)])

    result = await _call(backend, hub, "...", _core(tmp_path))

    assert result == LlmConfig().reply_empty


@respx.mock
async def test_empty_reply_after_tools_uses_done(tmp_path):
    # A tool ran, then the model produced empty content -> "Готово." (not the
    # "didn't hear" fallback).
    _mock_weather_404()
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final(None),
    ])

    result = await _call(backend, hub, "включи свет", _core(tmp_path))

    assert hub.calls == [("set_light", {})]
    assert result == LlmConfig().reply_empty_after_tools


@respx.mock
async def test_history_is_included(tmp_path):
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ответ")])

    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]
    await _call(backend, hub, "новый вопрос", _core(tmp_path), history=history)

    # System prompt + history + the new user turn, in order. The 404 weather is
    # omitted, so the system message is just the prompt body with the time prefix.
    messages = backend.seen[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1:] == [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "новый вопрос"},
    ]


@respx.mock
async def test_no_tools_passes_none_to_backend(tmp_path):
    # hub.tools is [] -> the loop passes None (not []) to complete().
    _mock_weather_404()
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ок")])

    await _call(backend, hub, "привет", _core(tmp_path))

    assert backend.seen[0][1] is None
