from loguru import logger

from src.tool_hub import BuiltinMcpSource, HttpMcpSource, ToolHub


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"desc {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class FakeSource:
    """Minimal ToolSource double: fixed tool list, records call()s.

    `fail_start` makes start() raise so we can test failure isolation. `slow`
    mirrors the real ToolSource declaration (read by ToolHub.is_slow/describe).
    """

    def __init__(self, id, tool_names, *, fail_start=False, slow=False):
        self.id = id
        self.slow = slow
        self._tools = [_tool(n) for n in tool_names]
        self.fail_start = fail_start
        self.calls = []          # records (raw_name, args)
        self.started = False
        self.ensured = False
        self.stopped = False

    async def start(self):
        if self.fail_start:
            raise RuntimeError("boom")
        self.started = True

    async def ensure(self):
        self.ensured = True

    def raw_tools(self):
        return self._tools

    async def call(self, raw_name, args):
        self.calls.append((raw_name, args))
        return f"{self.id}:{raw_name}"

    async def stop(self):
        self.stopped = True


async def test_advertised_names_are_raw_and_merged():
    home = FakeSource("home", ["set_light", "list_devices"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    # Advertised names are the sources' RAW names, in source order — no prefixing,
    # so the system prompt's bare names (set_light, ...) still match.
    names = [t["function"]["name"] for t in hub.tools]
    assert names == [
        "set_light",
        "list_devices",
        "get_current_weather",
    ]


async def test_call_routes_to_owning_source_with_raw_name():
    home = FakeSource("home", ["set_light"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    out = await hub.call("get_current_weather", {"city": "Moscow"})

    assert out == "weather:get_current_weather"
    # The owning source received the raw name.
    assert weather.calls == [("get_current_weather", {"city": "Moscow"})]
    assert home.calls == []


async def test_unknown_tool_returns_error_string():
    hub = ToolHub([FakeSource("home", ["set_light"])])
    await hub.start()

    out = await hub.call("nope", {})
    assert out == "error: unknown tool nope"


async def test_failing_source_does_not_break_others():
    broken = FakeSource("broken", ["x"], fail_start=True)
    healthy = FakeSource("home", ["set_light"])
    hub = ToolHub([broken, healthy])

    # One source raising in start() must NOT prevent the others from loading.
    await hub.start()

    names = [t["function"]["name"] for t in hub.tools]
    # The healthy source still serves its tool under the raw name.
    assert "set_light" in names
    assert healthy.started is True
    # The healthy source is still callable.
    assert await hub.call("set_light", {}) == "home:set_light"


async def test_source_dicts_are_not_mutated():
    home = FakeSource("home", ["set_light"])
    hub = ToolHub([home])
    await hub.start()

    # No name rewriting: the advertised name equals the source's raw name, and the
    # source dict is left untouched.
    assert home.raw_tools()[0]["function"]["name"] == "set_light"
    assert hub.tools[0]["function"]["name"] == "set_light"


async def test_name_collision_first_source_wins():
    # Two sources both expose a tool named "foo".
    first = FakeSource("first", ["foo"])
    second = FakeSource("second", ["foo"])
    hub = ToolHub([first, second])

    # Capture loguru output (loguru does not feed pytest's caplog by default).
    records = []
    sink_id = logger.add(records.append, level="WARNING")
    try:
        await hub.start()
    finally:
        logger.remove(sink_id)

    # "foo" is advertised exactly ONCE (deduped, first occurrence kept).
    names = [t["function"]["name"] for t in hub.tools]
    assert names == ["foo"]

    # call("foo", ...) routes to the FIRST source, not the second.
    out = await hub.call("foo", {"x": 1})
    assert out == "first:foo"
    assert first.calls == [("foo", {"x": 1})]
    assert second.calls == []

    # A warning was logged naming the tool and both source ids.
    logged = "".join(records)
    assert "WARNING" in logged
    assert "foo" in logged
    assert "first" in logged
    assert "second" in logged


async def test_ensure_tools_refreshes_each_source():
    home = FakeSource("home", ["set_light"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    await hub.ensure_tools()
    assert home.ensured is True
    assert weather.ensured is True


async def test_stop_calls_sources():
    home = FakeSource("home", ["set_light"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    await hub.stop()
    assert home.stopped is True
    assert weather.stopped is True


# --- set_sources(): hot-swap the live source set -----------------------------

async def test_set_sources_swaps_advertised_routes_and_lifecycle():
    # Start with one source, then hot-swap to two new sources. The advertised list and
    # routing reflect the NEW sources; the new sources were start()ed and the old one
    # stop()ped.
    old = FakeSource("old", ["old_tool"])
    hub = ToolHub([old])
    await hub.start()
    assert [t["function"]["name"] for t in hub.tools] == ["old_tool"]

    new_a = FakeSource("a", ["alpha"])
    new_b = FakeSource("b", ["beta"])
    await hub.set_sources([new_a, new_b])

    # Advertised list + routing now reflect ONLY the new sources.
    names = [t["function"]["name"] for t in hub.tools]
    assert names == ["alpha", "beta"]
    assert await hub.call("alpha", {}) == "a:alpha"
    assert await hub.call("beta", {}) == "b:beta"
    # The old tool no longer routes.
    assert await hub.call("old_tool", {}) == "error: unknown tool old_tool"

    # New sources were started; the old source was stopped.
    assert new_a.started is True
    assert new_b.started is True
    assert old.stopped is True


async def test_set_sources_start_failure_does_not_abort_swap():
    # One new source failing start() must NOT abort the swap: the other new sources are
    # still started/advertised, the swap completes, and the old source is still stopped.
    old = FakeSource("old", ["old_tool"])
    hub = ToolHub([old])
    await hub.start()

    broken = FakeSource("broken", ["x"], fail_start=True)
    healthy = FakeSource("healthy", ["good"])
    await hub.set_sources([broken, healthy])

    # The healthy source was started and is advertised/callable despite the sibling's
    # start() raising (the loop logs and continues, then swaps).
    assert healthy.started is True
    assert broken.started is False
    assert await hub.call("good", {}) == "healthy:good"
    names = [t["function"]["name"] for t in hub.tools]
    assert "good" in names
    # The old source no longer routes and was stopped despite the start failure.
    assert await hub.call("old_tool", {}) == "error: unknown tool old_tool"
    assert old.stopped is True


# --- describe(): per-source info for the admin panel -------------------------

async def test_describe_returns_one_entry_per_source():
    home = FakeSource("home", ["set_light", "list_devices"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    described = hub.describe()
    assert [s["id"] for s in described] == ["home", "weather"]
    # Every entry carries the full id/kind/online/slow/tools shape.
    for entry in described:
        assert set(entry) == {"id", "kind", "online", "slow", "tools"}

    home_entry = described[0]
    assert home_entry["online"] is True
    assert [t["name"] for t in home_entry["tools"]] == ["set_light", "list_devices"]
    # Tool descriptions come straight from the groq-shape function dicts.
    assert home_entry["tools"][0]["description"] == "desc set_light"


async def test_describe_online_reflects_having_tools():
    # A source advertising zero tools (e.g. a configured-but-unreachable MCP)
    # reports online=False; one with tools reports online=True.
    empty = FakeSource("empty", [])
    full = FakeSource("full", ["x"])
    hub = ToolHub([empty, full])
    await hub.start()

    described = {s["id"]: s for s in hub.describe()}
    assert described["empty"]["online"] is False
    assert described["empty"]["tools"] == []
    assert described["full"]["online"] is True


async def test_describe_kind_for_http_and_builtin_wrappers():
    # The real wrappers expose kind "http"/"builtin"; describe() surfaces it.
    class _FakeMcpHub:
        tools = [_tool("set_light")]

        async def start(self):
            return None

        async def ensure_tools(self):
            return None

        async def call(self, name, args):
            return ""

        async def stop(self):
            return None

    class _FakeFastMcp:
        async def list_tools(self):
            return []

        async def call_tool(self, name, args):
            return ""

    http = HttpMcpSource("home", _FakeMcpHub())
    builtin = BuiltinMcpSource("weather", _FakeFastMcp())
    hub = ToolHub([http, builtin])
    await hub.start()

    described = {s["id"]: s for s in hub.describe()}
    assert described["home"]["kind"] == "http"
    assert described["weather"]["kind"] == "builtin"


# --- is_slow(): the owning source declares whether its tools are slow ---------


async def test_is_slow_follows_owning_source_declaration():
    # A tool from a slow source is slow; one from a fast source is not; a name
    # that routes nowhere is not slow either (no filler for unknown tools).
    fast = FakeSource("home", ["set_light"])
    slow = FakeSource("websearch", ["search_web"], slow=True)
    hub = ToolHub([fast, slow])
    await hub.start()

    assert hub.is_slow("search_web") is True
    assert hub.is_slow("set_light") is False
    assert hub.is_slow("unknown_tool") is False


async def test_describe_surfaces_per_source_slow_flag():
    fast = FakeSource("home", ["set_light"])
    slow = FakeSource("websearch", ["search_web"], slow=True)
    hub = ToolHub([fast, slow])
    await hub.start()

    described = {s["id"]: s for s in hub.describe()}
    assert described["home"]["slow"] is False
    assert described["websearch"]["slow"] is True


async def test_wrapper_sources_accept_slow_keyword():
    # HttpMcpSource/BuiltinMcpSource carry the slow flag through to the hub.
    class _FakeMcpHub:
        tools = [_tool("search_web")]

        async def start(self):
            return None

    class _FakeFastMcp:
        async def list_tools(self):
            return []

    http = HttpMcpSource("websearch", _FakeMcpHub(), slow=True)
    builtin = BuiltinMcpSource("reminders", _FakeFastMcp(), slow=False)
    assert http.slow is True
    assert builtin.slow is False

    hub = ToolHub([http, builtin])
    await hub.start()
    assert hub.is_slow("search_web") is True


# --- BuiltinMcpSource._normalize: flatten varied FastMCP call_tool shapes -----


class _Content:
    """Stand-in for an mcp.types.TextContent block (only .text is read)."""

    def __init__(self, text):
        self.text = text


class _FakeFastMcpServer:
    """FastMCP double for BuiltinMcpSource: returns/raises a preset call_tool value.

    `call_result` is what call_tool returns; `raise_on_call` makes it raise instead,
    so we can drive the error path without touching src/.
    """

    def __init__(self, *, tools=None, call_result=None, raise_on_call=None):
        self._tools = tools or []
        self._call_result = call_result
        self._raise_on_call = raise_on_call
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return self._call_result


def test_builtin_normalize_tuple_shape_joins_content_text():
    # This SDK shape: (content_list, {"result": ...}). _normalize takes element 0
    # (the content list) and joins each block's .text with newlines.
    res = ([_Content("a"), _Content("b")], {"result": "ignored-meta"})
    assert BuiltinMcpSource._normalize(res) == "a\nb"


def test_builtin_normalize_dict_with_result_key():
    # A bare dict carrying a "result" key stringifies that value.
    assert BuiltinMcpSource._normalize({"result": "ok"}) == "ok"


def test_builtin_normalize_dict_without_result_key_stringifies_whole_dict():
    # No "result" key -> stringify the whole dict (the .get default is res itself).
    d = {"status": "done"}
    assert BuiltinMcpSource._normalize(d) == str(d)


def test_builtin_normalize_non_iterable_falls_back_to_str():
    # A non-iterable, non-tuple, non-dict object hits the TypeError fallback in the
    # join branch and is stringified.
    assert BuiltinMcpSource._normalize(123) == "123"


async def test_builtin_call_returns_error_string_on_server_failure():
    # call() must NOT raise: a server error surfaces as text for the model.
    server = _FakeFastMcpServer(raise_on_call=RuntimeError("boom"))
    src = BuiltinMcpSource("weather", server)
    out = await src.call("get_current_weather", {"city": "Moscow"})
    assert out == "error calling get_current_weather: boom"
    # The server was actually invoked with the raw name/args.
    assert server.calls == [("get_current_weather", {"city": "Moscow"})]


async def test_builtin_call_happy_path_returns_normalized_text():
    # Happy path: tuple shape from call_tool is normalized to joined content text.
    server = _FakeFastMcpServer(
        call_result=([_Content("sunny"), _Content("25C")], {"result": "meta"})
    )
    src = BuiltinMcpSource("weather", server)
    out = await src.call("get_current_weather", {"city": "Moscow"})
    assert out == "sunny\n25C"
    assert server.calls == [("get_current_weather", {"city": "Moscow"})]


# --- ToolHub fault isolation: ensure_tools / stop / set_sources --------------


class FaultSource(FakeSource):
    """FakeSource whose ensure()/stop() can be made to raise, to test isolation."""

    def __init__(self, id, tool_names, *, fail_ensure=False, fail_stop=False):
        super().__init__(id, tool_names)
        self.fail_ensure = fail_ensure
        self.fail_stop = fail_stop

    async def ensure(self):
        if self.fail_ensure:
            raise RuntimeError("ensure boom")
        await super().ensure()

    async def stop(self):
        if self.fail_stop:
            raise RuntimeError("stop boom")
        await super().stop()


class NoStopSource:
    """ToolSource double WITHOUT a stop attribute, to exercise the getattr-None skip."""

    def __init__(self, id, tool_names):
        self.id = id
        self._tools = [_tool(n) for n in tool_names]
        self.started = False
        self.ensured = False

    async def start(self):
        self.started = True

    async def ensure(self):
        self.ensured = True

    def raw_tools(self):
        return self._tools

    async def call(self, raw_name, args):
        return f"{self.id}:{raw_name}"


async def test_ensure_tools_isolates_failing_source():
    # One source's ensure() raising must NOT stop the others from being refreshed,
    # and no exception escapes ensure_tools().
    broken = FaultSource("broken", ["x"], fail_ensure=True)
    healthy = FakeSource("home", ["set_light"])
    hub = ToolHub([broken, healthy])
    await hub.start()

    await hub.ensure_tools()  # must not raise

    # The healthy source was still refreshed despite the sibling raising.
    assert healthy.ensured is True
    # The healthy tool stays advertised/routable after the refresh+rebuild.
    assert "set_light" in [t["function"]["name"] for t in hub.tools]


async def test_stop_isolates_failing_source_and_skips_no_stop_source():
    # Mix a source whose stop() raises, a source with NO stop attribute, and a healthy
    # one. stop() must not raise and the healthy source is still stopped.
    broken = FaultSource("broken", ["x"], fail_stop=True)
    no_stop = NoStopSource("nostop", ["y"])
    healthy = FakeSource("home", ["set_light"])
    hub = ToolHub([broken, no_stop, healthy])
    await hub.start()

    await hub.stop()  # must not raise even though broken.stop() throws and no_stop lacks stop

    # The healthy source was stopped despite the earlier failure and the skipped source.
    assert healthy.stopped is True


async def test_set_sources_isolates_old_source_stop_failure():
    # Old set mixes a source whose stop() raises with one lacking stop. The swap must
    # complete (new routes in place) and the surviving old source is still stopped.
    old_broken = FaultSource("old_broken", ["old_a"], fail_stop=True)
    old_no_stop = NoStopSource("old_nostop", ["old_b"])
    old_healthy = FaultSource("old_healthy", ["old_c"])
    hub = ToolHub([old_broken, old_no_stop, old_healthy])
    await hub.start()

    new = FakeSource("new", ["fresh"])
    await hub.set_sources([new])  # must not raise

    # The swap completed: new source advertised/routable, old tools gone.
    assert [t["function"]["name"] for t in hub.tools] == ["fresh"]
    assert await hub.call("fresh", {}) == "new:fresh"
    assert await hub.call("old_a", {}) == "error: unknown tool old_a"
    # The surviving old source (after the raising one) was still stopped.
    assert old_healthy.stopped is True
