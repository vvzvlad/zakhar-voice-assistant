import json

import httpx
import respx

from src import llm
from src.llm import (
    EMPTY_REPLY_FALLBACK,
    MAX_TOOL_ROUNDS,
    OPENROUTER_API_URL,
    call_llm_api,
)
from src.text import processing_response


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


def _patch_prompt(monkeypatch):
    async def fake_build_system_prompt(client):
        return "SYS"

    monkeypatch.setattr(llm, "build_system_prompt", fake_build_system_prompt)


@respx.mock
async def test_tool_path(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    respx.post(OPENROUTER_API_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json=_tool_call(
                    "set_light", '{"device_id":"bright_room_light","state":"on"}'
                ),
            ),
            httpx.Response(200, json=_final("Готово.")),
        ]
    )

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "включи свет")

    assert hub.calls == [
        ("set_light", {"device_id": "bright_room_light", "state": "on"})
    ]
    assert result == processing_response("Готово.")


@respx.mock
async def test_no_tool_path(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(200, json=_final("Привет, мясной мешок."))
    )

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "привет")

    assert hub.calls == []
    assert result == processing_response("Привет, мясной мешок.")


@respx.mock
async def test_rate_limit_path(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[])
    respx.post(OPENROUTER_API_URL).mock(return_value=httpx.Response(429))

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "привет")

    assert result == (
        "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, "
        "обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии"
    )


@respx.mock
async def test_max_tool_rounds_exhausted(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    respx.post(OPENROUTER_API_URL).mock(
        side_effect=[
            httpx.Response(200, json=_tool_call("set_light", "{}"))
            for _ in range(MAX_TOOL_ROUNDS + 1)
        ]
    )

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "включи свет")

    assert result == "Ошибка: слишком много вызовов инструментов"


@respx.mock
async def test_non_200_returns_error_message(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[])
    respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom"}})
    )

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "привет")

    assert result == "Ошибка: boom"


@respx.mock
async def test_httpx_error_returns_error_prefix(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[])
    respx.post(OPENROUTER_API_URL).mock(side_effect=httpx.ConnectError("down"))

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "привет")

    assert result.startswith("Ошибка:")


@respx.mock
async def test_history_is_included(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[])
    route = respx.post(OPENROUTER_API_URL).mock(
        return_value=httpx.Response(200, json=_final("ответ"))
    )

    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]
    async with httpx.AsyncClient() as client_ext:
        await call_llm_api(client_ext, hub, "новый вопрос", history=history)

    sent = json.loads(route.calls.last.request.content)
    assert sent["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "новый вопрос"},
    ]


@respx.mock
async def test_empty_final_reply_uses_fallback(monkeypatch):
    _patch_prompt(monkeypatch)
    hub = StubHub(tools=[])
    respx.post(OPENROUTER_API_URL).mock(return_value=httpx.Response(200, json=_final(None)))

    async with httpx.AsyncClient() as client_ext:
        result = await call_llm_api(client_ext, hub, "...")

    assert result == EMPTY_REPLY_FALLBACK
