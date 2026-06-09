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
