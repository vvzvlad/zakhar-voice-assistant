import json
from pathlib import Path

import httpx
import respx

import src.mcp_client as mcp_client
from src.mcp_client import McpToolHub


class StubTool:
    """Minimal stand-in for an mcp.Tool (name/description/inputSchema)."""

    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class StubContent:
    def __init__(self, text):
        self.text = text


class StubResult:
    def __init__(self, content):
        self.content = content


class StubSession:
    """Async-context-manager double for mcp.ClientSession."""

    def __init__(self, *, tools=None, result=None):
        self._tools = tools or []
        self._result = result
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        class _Resp:
            pass

        resp = _Resp()
        resp.tools = self._tools
        return resp

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._result


class StubTransport:
    """Async-context-manager double for streamablehttp_client."""

    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers

    async def __aenter__(self):
        return ("read", "write", None)

    async def __aexit__(self, *exc):
        return False


class StubSseTransport:
    """Async-context-manager double for sse_client (yields a 2-tuple)."""

    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers

    async def __aenter__(self):
        return ("read", "write")

    async def __aexit__(self, *exc):
        return False


def test_to_groq_tool_conversion():
    tool = StubTool(
        "set_light",
        "Turn a light on or off.",
        {"type": "object", "properties": {"device_id": {"type": "string"}}},
    )
    result = McpToolHub._to_groq_tool(tool)
    assert result == {
        "type": "function",
        "function": {
            "name": "set_light",
            "description": "Turn a light on or off.",
            "parameters": {"type": "object", "properties": {"device_id": {"type": "string"}}},
        },
    }


def test_to_groq_tool_defaults_for_missing_fields():
    tool = StubTool("noop", None, None)
    result = McpToolHub._to_groq_tool(tool)
    assert result["function"]["description"] == ""
    assert result["function"]["parameters"] == {"type": "object", "properties": {}}


async def test_call_returns_error_string_when_connection_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(mcp_client, "streamablehttp_client", boom)
    hub = McpToolHub("http://mcp.test:8201/mcp")
    out = await hub.call("set_light", {"device_id": "x", "state": "on"})
    assert out.startswith("error calling set_light")


async def test_ensure_tools_is_noop_when_cached(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("would error if called")

    monkeypatch.setattr(mcp_client, "streamablehttp_client", boom)
    hub = McpToolHub("http://mcp.test:8201/mcp")
    cached = [
        {"type": "function", "function": {"name": "set_light", "description": "", "parameters": {}}}
    ]
    hub._tools = cached
    await hub.ensure_tools()
    assert hub.tools == cached


async def test_call_flattens_content_text(monkeypatch):
    session = StubSession(result=StubResult([StubContent("done")]))

    monkeypatch.setattr(mcp_client, "streamablehttp_client", lambda url, headers=None: StubTransport(url, headers))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    hub = McpToolHub("http://mcp.test:8201/mcp")
    out = await hub.call("set_light", {"device_id": "x", "state": "on"})
    assert out == "done"
    assert session.calls == [("set_light", {"device_id": "x", "state": "on"})]


async def test_call_returns_no_output_when_empty_content(monkeypatch):
    session = StubSession(result=StubResult([]))

    monkeypatch.setattr(mcp_client, "streamablehttp_client", lambda url, headers=None: StubTransport(url, headers))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    hub = McpToolHub("http://mcp.test:8201/mcp")
    out = await hub.call("list_devices", {})
    assert out == "(no output)"


def test_resolve_transport_auto_detects_sse_by_url_suffix():
    assert McpToolHub._resolve_transport("auto", "http://ha:8123/mcp_server/sse") == "sse"
    assert McpToolHub._resolve_transport("auto", "http://ha:8123/mcp_server/sse/") == "sse"
    assert McpToolHub._resolve_transport("auto", "http://mcp.test:8201/mcp") == "streamable_http"


def test_resolve_transport_explicit_value_is_preserved():
    assert McpToolHub._resolve_transport("sse", "http://mcp.test:8201/mcp") == "sse"
    assert (
        McpToolHub._resolve_transport("streamable_http", "http://ha:8123/mcp_server/sse")
        == "streamable_http"
    )


async def test_call_uses_sse_transport_when_selected(monkeypatch):
    session = StubSession(result=StubResult([StubContent("done")]))

    def must_not_be_called(*a, **k):
        raise AssertionError("streamablehttp_client used while SSE transport selected")

    monkeypatch.setattr(mcp_client, "streamablehttp_client", must_not_be_called)
    monkeypatch.setattr(mcp_client, "sse_client", lambda url, headers=None: StubSseTransport(url, headers))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    hub = McpToolHub("http://ha:8123/mcp_server/sse")
    out = await hub.call("set_light", {"device_id": "x", "state": "on"})
    assert out == "done"
    assert session.calls == [("set_light", {"device_id": "x", "state": "on"})]


async def test_list_tools_converts_server_tools(monkeypatch):
    tool = StubTool(
        "set_light",
        "Turn a light on or off.",
        {"type": "object", "properties": {"device_id": {"type": "string"}}},
    )
    session = StubSession(tools=[tool])

    monkeypatch.setattr(mcp_client, "streamablehttp_client", lambda url, headers=None: StubTransport(url, headers))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    hub = McpToolHub("http://mcp.test:8201/mcp")
    await hub.start()
    assert hub.tools == [McpToolHub._to_groq_tool(tool)]


async def test_start_is_graceful_when_listing_fails(monkeypatch):
    # A smart-home outage during start() must NOT propagate: start() logs and leaves an
    # empty tool list so the assistant keeps running.
    def boom(*a, **k):
        raise RuntimeError("server down")

    monkeypatch.setattr(mcp_client, "streamablehttp_client", boom)
    hub = McpToolHub("http://mcp.test:8201/mcp")

    await hub.start()  # must not raise

    assert hub.tools == []


async def test_ensure_tools_self_heals_when_server_becomes_reachable(monkeypatch):
    # Cache is empty (start ran while the server was down). ensure_tools() should reload
    # and populate _tools once the transport/session start returning a tool.
    tool = StubTool(
        "set_light",
        "Turn a light on or off.",
        {"type": "object", "properties": {"device_id": {"type": "string"}}},
    )
    session = StubSession(tools=[tool])

    monkeypatch.setattr(mcp_client, "streamablehttp_client", lambda url, headers=None: StubTransport(url, headers))
    monkeypatch.setattr(mcp_client, "ClientSession", lambda read, write: session)

    hub = McpToolHub("http://mcp.test:8201/mcp")
    assert hub.tools == []  # empty cache precondition

    await hub.ensure_tools()

    assert hub.tools == [McpToolHub._to_groq_tool(tool)]


async def test_ensure_tools_stays_empty_when_reload_still_fails(monkeypatch):
    # The reload itself raising must be swallowed: _tools stays [] and no exception
    # escapes ensure_tools().
    def boom(*a, **k):
        raise RuntimeError("still down")

    monkeypatch.setattr(mcp_client, "streamablehttp_client", boom)
    hub = McpToolHub("http://mcp.test:8201/mcp")
    assert hub.tools == []

    await hub.ensure_tools()  # must not raise

    assert hub.tools == []


# ---------------------------------------------------------------------------
# R-MCP-1: wire-format contract test.
#
# The StubSession tests above never exercise the real mcp.ClientSession decoder
# nor any JSON-RPC/SSE framing: they hand the hub a fake session whose return
# values are already Python objects. They therefore CANNOT catch SDK-decoder
# drift or a server-side wire-format change (e.g. `content[].text` renamed to
# `content[].value`, or `inputSchema` renamed). This test pins the real Node-RED
# smart-home MCP wire format by replaying a REAL recorded server exchange
# (tests/fixtures/mcp_smarthome_wire.json) through the project's REAL code path:
#   McpToolHub.start()/call()  ->  real streamablehttp_client  ->  real
#   mcp.ClientSession / mcp.types decoder, with only the HTTP transport mocked
#   by respx (fully offline; no socket to 10.31.41.62 is ever opened).
#
# Approach used: respx transport mock (the IMPLEMENTATION PLAN's preferred path).
# It works because the recorded server is STATELESS (session_id_present == false):
# the SDK never gets an Mcp-Session-Id, so handle_get_stream() returns immediately
# (no server->client GET SSE channel to satisfy) and terminate_session() is a
# no-op (no DELETE). Only the POST request/response pairs need to be served.
#
# One SDK quirk handled: mcp.ClientSession assigns its OWN monotonic JSON-RPC
# request ids starting at 0 and correlates responses by id. The recorded payloads
# carry the original ids (1, 2, 3). If we replayed them verbatim, ClientSession
# would never match the response to its pending request and initialize() would
# hang forever. So the handler rewrites ONLY the JSON-RPC envelope `id` to echo
# the incoming request's id. The load-bearing wire payload (result.tools[],
# inputSchema, content[].text, ...) is preserved exactly as recorded.

WIRE_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_smarthome_wire.json"


def _sse_data_payload(raw_sse_block: str) -> str:
    """Extract the JSON string from the `data:` line of a recorded SSE block."""
    for line in raw_sse_block.splitlines():
        if line.startswith("data:"):
            return line[len("data:"):].strip()
    raise AssertionError("recorded SSE block has no data: line")


def _reframe_sse(payload_json: str, request_id) -> bytes:
    """Re-wrap the recorded JSON payload as a clean SSE event.

    Only the JSON-RPC envelope id is replaced (to match the SDK-chosen request
    id); every other field is the exact recorded wire payload.
    """
    obj = json.loads(payload_json)
    obj["id"] = request_id  # echo the SDK's request id so ClientSession correlates
    payload = json.dumps(obj, ensure_ascii=False)
    return f"event: message\r\ndata: {payload}\r\n\r\n".encode()


@respx.mock
async def test_real_session_decodes_recorded_smarthome_wire():
    fixture = json.loads(WIRE_FIXTURE.read_text(encoding="utf-8"))
    url = fixture["url"]
    # Sanity-pin the recording: stateless server, negotiated protocol.
    assert fixture["session_id_present"] is False
    assert fixture["protocol_negotiated"] == "2025-03-26"

    payload_by_method = {
        "initialize": _sse_data_payload(fixture["initialize_raw"]),
        "tools/list": _sse_data_payload(fixture["tools_list_raw"]),
        "tools/call": _sse_data_payload(fixture["tools_call_raw"]),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # The SDK may open a server->client GET SSE channel and a DELETE on close;
        # answer tolerantly so the streamable-http handshake always completes.
        if request.method == "GET":
            return httpx.Response(405)  # Method Not Allowed -> no GET stream
        if request.method == "DELETE":
            return httpx.Response(200)
        body = json.loads(request.content)
        method = body.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)  # notifications get no body
        payload = payload_by_method.get(method)
        if payload is None:
            return httpx.Response(400)
        return httpx.Response(
            200,
            headers={"content-type": fixture[f"{method.split('/')[0]}_content_type"]
                     if method == "initialize" else "text/event-stream"},
            content=_reframe_sse(payload, body.get("id")),
        )

    respx.route(url=url).mock(side_effect=handler)

    # Drive the REAL hub over the REAL streamable_http transport + ClientSession.
    hub = McpToolHub(url, None, "auto")
    await hub.start()

    # [tool names] survive the real decoder, in the recorded order.
    names = [(t.get("function") or t)["name"] for t in hub.tools]
    assert names == ["set_light", "set_dimmer", "set_climate", "set_switch", "set_lock", "set_scene"]

    # [schema survives the real decoder] for set_light.
    light = next(t for t in hub.tools if (t.get("function") or t)["name"] == "set_light")
    fn = light.get("function") or light
    schema = fn.get("parameters") or fn.get("inputSchema")
    assert "table_light" in schema["properties"]["device_id"]["enum"]
    assert schema["required"] == ["device_id", "state"]
    assert fn["description"]  # non-empty description

    # [call text extraction through real decoder] content[].text == "ok".
    out = await hub.call("set_light", {"device_id": "table_light", "state": "on"})
    assert out == "ok"
