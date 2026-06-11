import httpx
import pytest

from src.core_config import CoreConfig, OpenWeatherMapConfig, PromptConfig
from src.llm import call_llm_api
from src.plugins.llm.base import LlmConfig
from src.stage_errors import StageError
from src.text import processing_response
from src.tool_hub import ToolHub

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


class FakeToolSource:
    """Minimal ToolSource double for driving a REAL ToolHub end to end.

    Advertises one tool under its raw name and records call()s, so a test can prove the
    name the model emits is routed unchanged through the hub to the owning source.
    """

    def __init__(self, id, tool):
        self.id = id
        self._tools = [tool]
        self.calls = []  # records (name, args)

    async def start(self):
        return None

    async def ensure(self):
        return None

    def raw_tools(self):
        return self._tools

    async def call(self, name, args):
        self.calls.append((name, args))
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


def _tool_call_with_content(name, arguments_json, content):
    """A tool-call round that ALSO carries non-empty spoken content (the persona's
    'I'll go check it' filler line next to the tool request)."""
    resp = _tool_call(name, arguments_json)
    resp["choices"][0]["message"]["content"] = content
    return resp


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
        openweathermap=OpenWeatherMapConfig(api_key="w-key", city="Moscow"),
    )


async def _call(backend, hub, text, core, *, history=None, max_tool_rounds=MAX_TOOL_ROUNDS):
    return await call_llm_api(
        backend,
        hub,
        text,
        core=core,
        llm_cfg=LlmConfig(max_tool_rounds=max_tool_rounds),
        history=history,
    )


async def test_tool_path(tmp_path):
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


async def test_tool_path_through_real_tool_hub(tmp_path):
    # Guard the whole loop against name-routing regressions: drive call_llm_api over a
    # REAL ToolHub (not StubHub). The hub advertises the source's raw name unchanged, so
    # the name the model emits must route straight back to the owning source.
    source = FakeToolSource("home", SET_LIGHT_TOOL)
    hub = ToolHub([source])
    await hub.start()

    # The hub advertises the bare name the system prompt teaches.
    assert [t["function"]["name"] for t in hub.tools] == ["set_light"]

    backend = FakeLlmBackend([
        _tool_call("set_light", '{"device_id":"bright_room_light","state":"on"}'),
        _final("Готово."),
    ])

    result = await _call(backend, hub, "включи свет", _core(tmp_path))

    # The raw tool name + args reached the owning source via the hub.
    assert source.calls == [
        ("set_light", {"device_id": "bright_room_light", "state": "on"})
    ]
    assert result == processing_response("Готово.")


async def test_no_tool_path(tmp_path):
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([_final("Привет, мясной мешок.")])

    result = await _call(backend, hub, "привет", _core(tmp_path))

    assert hub.calls == []
    assert result == processing_response("Привет, мясной мешок.")


async def test_rate_limit_raises_stage_error(tmp_path):
    # HTTP 429 -> StageError(kind="rate_limit"); the pipeline maps it to the
    # configured reply_rate_limit phrase (tested at the pipeline level).
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(429)])

    with pytest.raises(StageError) as ei:
        await _call(backend, hub, "привет", _core(tmp_path))

    assert ei.value.stage == "llm"
    assert ei.value.kind == "rate_limit"


async def test_non_2xx_raises_stage_error_with_reason(tmp_path):
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(500, {"error": {"message": "boom"}})])

    with pytest.raises(StageError) as ei:
        await _call(backend, hub, "привет", _core(tmp_path))

    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"
    # The JSON error message is extracted as the StageError text.
    assert str(ei.value) == "boom"


async def test_httpx_error_raises_stage_error(tmp_path):
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([httpx.ConnectError("down")])

    with pytest.raises(StageError) as ei:
        await _call(backend, hub, "привет", _core(tmp_path))

    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"


async def test_max_tool_rounds_exhausted_raises_stage_error(tmp_path):
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}") for _ in range(MAX_TOOL_ROUNDS + 1)
    ])

    with pytest.raises(StageError) as ei:
        await _call(backend, hub, "включи свет", _core(tmp_path))

    assert ei.value.stage == "llm"
    assert ei.value.kind == "tool_rounds"


async def test_empty_final_reply_uses_fallback(tmp_path):
    # No tool ever ran -> empty final content falls back to the "didn't hear" line.
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final(None)])

    result = await _call(backend, hub, "...", _core(tmp_path))

    assert result == LlmConfig().reply_empty


async def test_empty_reply_after_tools_uses_done(tmp_path):
    # A tool ran, then the model produced empty content -> "Готово." (not the
    # "didn't hear" fallback).
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final(None),
    ])

    result = await _call(backend, hub, "включи свет", _core(tmp_path))

    assert hub.calls == [("set_light", {})]
    assert result == LlmConfig().reply_empty_after_tools


async def test_history_is_included(tmp_path):
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ответ")])

    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]
    await _call(backend, hub, "новый вопрос", _core(tmp_path), history=history)

    # System prompt + history + the new user turn, in order. Weather is no longer in
    # the prompt (it is a tool now), so the system message is just the prompt body
    # with the time prefix.
    messages = backend.seen[0][0]
    assert messages[0]["role"] == "system"
    assert messages[1:] == [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "новый вопрос"},
    ]


async def test_no_tools_passes_none_to_backend(tmp_path):
    # hub.tools is [] -> the loop passes None (not []) to complete().
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ок")])

    await _call(backend, hub, "привет", _core(tmp_path))

    assert backend.seen[0][1] is None


def _tool_call_usage(name, arguments_json, *, model, total_tokens):
    resp = _tool_call(name, arguments_json)
    resp["model"] = model
    resp["usage"] = {"total_tokens": total_tokens}
    return resp


def _final_usage(content, *, model, total_tokens):
    resp = _final(content)
    resp["model"] = model
    resp["usage"] = {"total_tokens": total_tokens}
    return resp


async def test_trace_is_populated(tmp_path):
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_usage("set_light", '{"device_id":"lamp","state":"on"}',
                         model="m-tool", total_tokens=30),
        _final_usage("Готово.", model="m-final", total_tokens=12),
    ])

    trace: dict = {}
    result = await call_llm_api(
        backend, hub, "включи свет",
        core=_core(tmp_path), llm_cfg=LlmConfig(max_tool_rounds=MAX_TOOL_ROUNDS),
        trace=trace,
    )
    assert result == processing_response("Готово.")

    # model = last seen; tokens summed across rounds.
    assert trace["model"] == "m-final"
    assert trace["tokens"] == 42

    # The model input is captured once into trace["request"].
    req = trace["request"]
    assert isinstance(req["system_prompt"], str) and req["system_prompt"]
    assert isinstance(req["context"], list)
    assert req["user_text"] == "включи свет"
    assert req["tools"] == [SET_LIGHT_TOOL]

    # Two rounds: a tool-call round carrying the executed call, then a final answer.
    assert [r["note"] for r in trace["rounds"]] == ["tool call", "final answer"]
    tool_round = trace["rounds"][0]
    assert tool_round["round"] == 1
    assert tool_round["tokens"] == 30
    assert tool_round["calls"] == [
        {"name": "set_light", "args": {"device_id": "lamp", "state": "on"}, "result": "ok"}
    ]
    final_round = trace["rounds"][1]
    assert final_round["round"] == 2
    assert final_round["calls"] == []
    # Each round carries the RAW model content (the final spoken reply is
    # processing_response("Готово."), but the stored content is the raw "Готово.").
    assert final_round["content"] == "Готово."
    assert "content" in tool_round


async def test_trace_none_is_a_noop(tmp_path):
    # Omitting trace must not change behavior or raise.
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ответ")])
    result = await _call(backend, hub, "привет", _core(tmp_path))
    assert result == processing_response("ответ")


async def test_no_choices_raises_stage_error(tmp_path):
    # Provider returned a response with no choices -> StageError, no KeyError.
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([{"model": "x", "usage": {}}])  # no "choices"
    with pytest.raises(StageError) as ei:
        await _call(backend, hub, "привет", _core(tmp_path))
    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"


async def test_malformed_tool_args_degrade_to_empty_dict(tmp_path):
    # Model emitted invalid JSON in tool arguments -> args fall back to {} and the
    # tool still runs; the loop then returns the final reply.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{not json"),
        _final("Готово."),
    ])
    result = await _call(backend, hub, "включи свет", _core(tmp_path))
    assert hub.calls == [("set_light", {})]
    assert result == processing_response("Готово.")


async def test_on_filler_called_for_round_with_content_and_tool_calls(tmp_path):
    # A tool-requesting round that ALSO carries spoken content -> on_filler is invoked
    # once with (content, [tool_name]); the final reply is still returned correctly.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "Щас гляну…"),
        _final("Готово."),
    ])

    seen = []  # records (content, tool_names) per callback invocation

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await call_llm_api(
        backend, hub, "включи свет",
        core=_core(tmp_path), llm_cfg=LlmConfig(max_tool_rounds=MAX_TOOL_ROUNDS),
        on_filler=recorder,
    )

    assert seen == [("Щас гляну…", ["set_light"])]
    assert result == processing_response("Готово.")


async def test_on_filler_not_called_without_tool_calls(tmp_path):
    # A plain final answer (no tool_calls) must NOT invoke on_filler.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([_final("Привет, мясной мешок.")])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await call_llm_api(
        backend, hub, "привет",
        core=_core(tmp_path), llm_cfg=LlmConfig(max_tool_rounds=MAX_TOOL_ROUNDS),
        on_filler=recorder,
    )

    assert seen == []
    assert result == processing_response("Привет, мясной мешок.")


async def test_on_filler_not_called_when_content_empty(tmp_path):
    # A tool-call round with no spoken content (content=None) must NOT invoke on_filler.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final("Готово."),
    ])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await call_llm_api(
        backend, hub, "включи свет",
        core=_core(tmp_path), llm_cfg=LlmConfig(max_tool_rounds=MAX_TOOL_ROUNDS),
        on_filler=recorder,
    )

    assert seen == []
    assert result == processing_response("Готово.")


async def test_on_filler_failure_does_not_break_loop(tmp_path):
    # A callback that raises must be swallowed: the loop still runs the tool and
    # returns the final reply.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "Щас гляну…"),
        _final("Готово."),
    ])

    async def boom(content, tool_names):
        raise RuntimeError("filler boom")

    result = await call_llm_api(
        backend, hub, "включи свет",
        core=_core(tmp_path), llm_cfg=LlmConfig(max_tool_rounds=MAX_TOOL_ROUNDS),
        on_filler=boom,
    )

    assert hub.calls == [("set_light", {})]
    assert result == processing_response("Готово.")
