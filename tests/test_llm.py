import dataclasses
import inspect

import httpx
import pytest

from src.llm import LlmRequest, LlmResult, LlmStage
from src.plugins.llm.base import LlmConfig
from src.llm_text import clean_llm_output
from src.stage_errors import StageError
from src.tool_hub import ToolHub

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = "SYSTEM PROMPT BODY"


class StubHub:
    """Tool hub double: advertises a fixed tool list and records call()s.

    Mirrors ToolHub.call's keyword-only `device` parameter and records the device
    each call rode in with, so tests can assert the stage passes req.device through.
    """

    def __init__(self, tools=None):
        self.tools = tools or []
        self.calls = []    # records (name, args)
        self.devices = []  # records the device= of each call

    async def ensure_tools(self):
        return None

    async def call(self, name, arguments, *, device=None):
        self.calls.append((name, arguments))
        self.devices.append(device)
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


async def _respond(backend, hub, text, *, history=None, max_tool_rounds=MAX_TOOL_ROUNDS,
                   device=None, on_filler=None):
    """Drive an LlmStage with a ready-made request and return the LlmResult.

    The system prompt is passed directly (assembled by the orchestrator in
    production), so no CoreConfig/prompt file is involved at this level.
    """
    stage = LlmStage(backend, hub, LlmConfig(max_tool_rounds=max_tool_rounds))
    return await stage.respond(
        LlmRequest(
            system_prompt=SYSTEM_PROMPT,
            history=history or [],
            user_text=text,
            device=device,
        ),
        on_filler=on_filler,
    )


async def test_tool_path():
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", '{"device_id":"bright_room_light","state":"on"}'),
        _final("Готово."),
    ])

    result = await _respond(backend, hub, "включи свет")

    assert hub.calls == [
        ("set_light", {"device_id": "bright_room_light", "state": "on"})
    ]
    # clean_llm_output is applied to the final content.
    assert result.reply == clean_llm_output("Готово.")
    assert result.tool_used is True


async def test_tool_path_through_real_tool_hub():
    # Guard the whole loop against name-routing regressions: drive LlmStage over a
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

    result = await _respond(backend, hub, "включи свет")

    # The raw tool name + args reached the owning source via the hub.
    assert source.calls == [
        ("set_light", {"device_id": "bright_room_light", "state": "on"})
    ]
    assert result.reply == clean_llm_output("Готово.")


async def test_device_is_passed_to_hub_call():
    # req.device travels to the hub as hub.call(..., device=...) — the stage no
    # longer touches the current_device ContextVar itself (the hub owns it).
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final("Готово."),
    ])

    await _respond(backend, hub, "включи свет", device="kitchen")

    assert hub.devices == ["kitchen"]


async def test_device_defaults_to_none_in_hub_call():
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final("Готово."),
    ])

    await _respond(backend, hub, "включи свет")

    assert hub.devices == [None]


async def test_no_tool_path():
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([_final("Привет, мясной мешок.")])

    result = await _respond(backend, hub, "привет")

    assert hub.calls == []
    assert result.reply == clean_llm_output("Привет, мясной мешок.")
    assert result.tool_used is False


async def test_rate_limit_raises_stage_error():
    # HTTP 429 -> StageError(kind="rate_limit"); the pipeline maps it to the
    # configured reply_rate_limit phrase (tested at the pipeline level).
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(429)])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "привет")

    assert ei.value.stage == "llm"
    assert ei.value.kind == "rate_limit"
    # The partial observability data is attached so the orchestrator can still
    # record the model input even though the run failed before any round completed.
    partial = ei.value.partial
    assert partial["model"] is None and partial["tokens"] is None
    assert partial["rounds"] == []
    assert partial["request"]["user_text"] == "привет"


async def test_non_2xx_raises_stage_error_with_reason():
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_http_status_error(500, {"error": {"message": "boom"}})])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "привет")

    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"
    # The JSON error message is extracted as the StageError text.
    assert str(ei.value) == "boom"


async def test_httpx_error_raises_stage_error():
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([httpx.ConnectError("down")])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "привет")

    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"


async def test_max_tool_rounds_exhausted_raises_stage_error():
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}") for _ in range(MAX_TOOL_ROUNDS + 1)
    ])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "включи свет")

    assert ei.value.stage == "llm"
    assert ei.value.kind == "tool_rounds"
    # All executed tool rounds are preserved on the error for observability.
    partial = ei.value.partial
    assert len(partial["rounds"]) == MAX_TOOL_ROUNDS
    assert all(r["note"] == "tool call" for r in partial["rounds"])


async def test_max_tool_rounds_exhausted_with_content_returns_cleaned_text():
    # Rounds exhausted but every round carried spoken content: the loop must
    # RETURN the cleaned last content (clean_llm_output strips <think> blocks)
    # instead of raising StageError(kind="tool_rounds").
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "<think>x</think>щас")
        for _ in range(MAX_TOOL_ROUNDS + 1)
    ])

    result = await _respond(backend, hub, "включи свет")

    assert result.reply == "щас"
    # Rounds bookkeeping still reflects every executed tool round.
    assert len(result.rounds) == MAX_TOOL_ROUNDS
    assert result.tool_used is True


async def test_empty_final_reply_uses_fallback():
    # No tool ever ran -> empty final content falls back to the "didn't hear" line.
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final(None)])

    result = await _respond(backend, hub, "...")

    assert result.reply == LlmConfig().reply_empty
    assert result.tool_used is False


async def test_empty_reply_after_tools_uses_done():
    # A tool ran, then the model produced empty content -> "Готово." (not the
    # "didn't hear" fallback).
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final(None),
    ])

    result = await _respond(backend, hub, "включи свет")

    assert hub.calls == [("set_light", {})]
    assert result.reply == LlmConfig().reply_empty_after_tools
    assert result.tool_used is True


async def test_history_is_included():
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ответ")])

    history = [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
    ]
    await _respond(backend, hub, "новый вопрос", history=history)

    # System prompt (verbatim from the request) + history + the new user turn,
    # in order. The stage no longer assembles the prompt itself.
    messages = backend.seen[0][0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:] == [
        {"role": "user", "content": "старый вопрос"},
        {"role": "assistant", "content": "старый ответ"},
        {"role": "user", "content": "новый вопрос"},
    ]


async def test_no_tools_passes_none_to_backend():
    # hub.tools is [] -> the loop passes None (not []) to complete().
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([_final("ок")])

    await _respond(backend, hub, "привет")

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


async def test_result_observability_is_populated():
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_usage("set_light", '{"device_id":"lamp","state":"on"}',
                         model="m-tool", total_tokens=30),
        _final_usage("Готово.", model="m-final", total_tokens=12),
    ])

    result = await _respond(backend, hub, "включи свет")
    assert result.reply == clean_llm_output("Готово.")

    # model = last seen; tokens summed across rounds.
    assert result.model == "m-final"
    assert result.tokens == 42

    # The model input is captured once into request_debug.
    req = result.request_debug
    assert req["system_prompt"] == SYSTEM_PROMPT
    assert isinstance(req["context"], list)
    assert req["user_text"] == "включи свет"
    assert req["tools"] == [SET_LIGHT_TOOL]

    # Two rounds: a tool-call round carrying the executed call, then a final answer.
    assert [r["note"] for r in result.rounds] == ["tool call", "final answer"]
    tool_round = result.rounds[0]
    assert tool_round["round"] == 1
    assert tool_round["tokens"] == 30
    assert tool_round["calls"] == [
        {"name": "set_light", "args": {"device_id": "lamp", "state": "on"}, "result": "ok"}
    ]
    final_round = result.rounds[1]
    assert final_round["round"] == 2
    assert final_round["calls"] == []
    # Each round carries the RAW model content (the final spoken reply is
    # clean_llm_output("Готово."), but the stored content is the raw "Готово.").
    assert final_round["content"] == "Готово."
    assert "content" in tool_round
    assert result.tool_used is True


async def test_no_choices_raises_stage_error():
    # Provider returned a response with no choices -> StageError, no KeyError.
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([{"model": "x", "usage": {}}])  # no "choices"
    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "привет")
    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"


async def test_malformed_tool_args_degrade_to_empty_dict():
    # Model emitted invalid JSON in tool arguments -> args fall back to {} and the
    # tool still runs; the loop then returns the final reply.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{not json"),
        _final("Готово."),
    ])
    result = await _respond(backend, hub, "включи свет")
    assert hub.calls == [("set_light", {})]
    assert result.reply == clean_llm_output("Готово.")


async def test_on_filler_called_for_round_with_content_and_tool_calls():
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

    result = await _respond(backend, hub, "включи свет", on_filler=recorder)

    assert seen == [("Щас гляну…", ["set_light"])]
    assert result.reply == clean_llm_output("Готово.")


async def test_on_filler_not_called_without_tool_calls():
    # A plain final answer (no tool_calls) must NOT invoke on_filler.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([_final("Привет, мясной мешок.")])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await _respond(backend, hub, "привет", on_filler=recorder)

    assert seen == []
    assert result.reply == clean_llm_output("Привет, мясной мешок.")


async def test_on_filler_not_called_when_content_empty():
    # A tool-call round with no spoken content (content=None) must NOT invoke on_filler.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final("Готово."),
    ])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await _respond(backend, hub, "включи свет", on_filler=recorder)

    assert seen == []
    assert result.reply == clean_llm_output("Готово.")


async def test_on_filler_failure_does_not_break_loop():
    # A callback that raises must be swallowed: the loop still runs the tool and
    # returns the final reply.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "Щас гляну…"),
        _final("Готово."),
    ])

    async def boom(content, tool_names):
        raise RuntimeError("filler boom")

    result = await _respond(backend, hub, "включи свет", on_filler=boom)

    assert hub.calls == [("set_light", {})]
    assert result.reply == clean_llm_output("Готово.")


# --- pipeline <-> LlmStage contract pin -------------------------------------------


def test_llm_stage_contract_pin():
    # Machine pin for the handwritten FakeLlmStage doubles in tests/test_pipeline.py:
    # they re-implement this exact surface (respond signature + the request/result
    # dataclass fields) by hand, so any change here must be propagated to those
    # fakes — update them when this test changes.
    sig = inspect.signature(LlmStage.respond)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "req", "on_filler"]
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[2].kind is inspect.Parameter.KEYWORD_ONLY

    assert {f.name for f in dataclasses.fields(LlmResult)} == {
        "reply", "model", "tokens", "rounds", "request_debug", "tool_used",
    }
    assert {f.name for f in dataclasses.fields(LlmRequest)} == {
        "system_prompt", "history", "user_text", "device",
    }


# --- failure-path and multi-tool-round scenarios ----------------------------------


async def test_mid_loop_failure_attaches_partial_from_completed_rounds():
    # Round 1 succeeds (tool call, model m1, 30 tokens), round 2 raises a 500:
    # the StageError's partial carries everything accumulated before the failure.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_usage("set_light", "{}", model="m1", total_tokens=30),
        _http_status_error(500, {"error": {"message": "boom"}}),
    ])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "включи свет")

    partial = ei.value.partial
    assert len(partial["rounds"]) == 1
    assert partial["rounds"][0]["note"] == "tool call"
    assert partial["tokens"] == 30
    assert partial["model"] == "m1"


async def test_non_2xx_with_non_json_body_raises_clean_stage_error():
    # A 502 whose body is plain text ("Bad Gateway"): response.json() fails inside
    # the reason-extraction try/except, which must degrade to reason=None and still
    # raise a clean StageError with a non-empty message (no secondary exception).
    request = httpx.Request("POST", "https://llm.test/chat")
    response = httpx.Response(502, text="Bad Gateway", request=request)
    err = httpx.HTTPStatusError("server error", request=request, response=response)
    hub = StubHub(tools=[])
    backend = FakeLlmBackend([err])

    with pytest.raises(StageError) as ei:
        await _respond(backend, hub, "привет")

    assert ei.value.stage == "llm"
    assert ei.value.kind == "error"
    assert str(ei.value)  # non-empty message


def _multi_tool_call(specs):
    """One assistant message carrying SEVERAL tool_calls: specs = [(id, name, args_json)]."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": id_,
                            "type": "function",
                            "function": {"name": name, "arguments": args_json},
                        }
                        for id_, name, args_json in specs
                    ],
                }
            }
        ],
        "model": "x",
        "usage": {},
    }


class PerNameHub(StubHub):
    """StubHub returning a per-tool result, so tests can pair each appended
    {"role": "tool"} message with the tool that actually produced it."""

    async def call(self, name, arguments, *, device=None):
        await super().call(name, arguments, device=device)
        return f"result:{name}"


async def test_multiple_tool_calls_in_one_round_executed_in_order():
    # One assistant round requesting TWO tools: both execute in order, and the two
    # appended tool messages pair the correct tool_call_id with the correct result.
    hub = PerNameHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _multi_tool_call([
            ("c1", "set_light", '{"state":"on"}'),
            ("c2", "get_current_weather", '{"city":"Moscow"}'),
        ]),
        _final("Готово."),
    ])

    result = await _respond(backend, hub, "свет и погода")

    assert hub.calls == [
        ("set_light", {"state": "on"}),
        ("get_current_weather", {"city": "Moscow"}),
    ]
    # The second complete() call sees both tool results, each under its own id.
    messages = backend.seen[1][0]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "c1", "name": "set_light",
         "content": "result:set_light"},
        {"role": "tool", "tool_call_id": "c2", "name": "get_current_weather",
         "content": "result:get_current_weather"},
    ]
    assert result.reply == clean_llm_output("Готово.")
    assert result.tool_used is True


async def test_tool_result_is_fed_back_to_the_model():
    # The second complete() call must END with the assistant tool-call message
    # followed by the {"role": "tool"} result — i.e. the tool output actually
    # reaches the model before it produces the final reply.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call("set_light", "{}"),
        _final("Готово."),
    ])

    await _respond(backend, hub, "включи свет")

    assert len(backend.seen) == 2
    messages = backend.seen[1][0]
    assistant_msg, tool_msg = messages[-2], messages[-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "c1"
    assert tool_msg == {
        "role": "tool", "tool_call_id": "c1", "name": "set_light", "content": "ok",
    }


async def test_on_filler_skipped_for_whitespace_only_content():
    # A tool-call round whose content is whitespace-only: on_filler is NOT invoked,
    # but the tool itself still runs.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "   "),
        _final("Готово."),
    ])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await _respond(backend, hub, "включи свет", on_filler=recorder)

    assert seen == []
    assert hub.calls == [("set_light", {})]  # the tool still executed
    assert result.reply == clean_llm_output("Готово.")


async def test_on_filler_fires_for_every_qualifying_round():
    # The stage is policy-free: it forwards EVERY tool-requesting round that carries
    # content. The at-most-once dedup belongs to the pipeline's callback, not here.
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    backend = FakeLlmBackend([
        _tool_call_with_content("set_light", "{}", "Секунду…"),
        _tool_call_with_content("set_light", "{}", "Ещё чуть-чуть…"),
        _final("Готово."),
    ])

    seen = []

    async def recorder(content, tool_names):
        seen.append((content, tool_names))

    result = await _respond(backend, hub, "включи свет", on_filler=recorder)

    assert [content for content, _ in seen] == ["Секунду…", "Ещё чуть-чуть…"]
    assert result.reply == clean_llm_output("Готово.")


async def test_tokens_survive_a_usage_less_round():
    # Round 1 reports total_tokens=30; the final round has NO "usage" key at all.
    # The sum must keep the 30 (no TypeError, not reset to None).
    hub = StubHub(tools=[SET_LIGHT_TOOL])
    final = _final("Готово.")
    del final["usage"]  # provider omitted usage entirely on the final round
    backend = FakeLlmBackend([
        _tool_call_usage("set_light", "{}", model="m1", total_tokens=30),
        final,
    ])

    result = await _respond(backend, hub, "включи свет")

    assert result.tokens == 30
    assert result.reply == clean_llm_output("Готово.")
