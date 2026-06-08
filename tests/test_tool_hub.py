from loguru import logger

from src.tool_hub import ToolHub


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

    `fail_start` makes start() raise so we can test failure isolation.
    """

    def __init__(self, id, tool_names, *, fail_start=False):
        self.id = id
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
