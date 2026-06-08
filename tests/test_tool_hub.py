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


async def test_advertised_names_are_prefixed_and_merged():
    home = FakeSource("home", ["set_light", "list_devices"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    names = [t["function"]["name"] for t in hub.tools]
    assert names == [
        "home__set_light",
        "home__list_devices",
        "weather__get_current_weather",
    ]


async def test_call_routes_to_owning_source_with_raw_name():
    home = FakeSource("home", ["set_light"])
    weather = FakeSource("weather", ["get_current_weather"])
    hub = ToolHub([home, weather])
    await hub.start()

    out = await hub.call("weather__get_current_weather", {"city": "Moscow"})

    assert out == "weather:get_current_weather"
    # The owning source received the RAW (un-prefixed) name.
    assert weather.calls == [("get_current_weather", {"city": "Moscow"})]
    assert home.calls == []


async def test_unknown_tool_returns_error_string():
    hub = ToolHub([FakeSource("home", ["set_light"])])
    await hub.start()

    out = await hub.call("home__nope", {})
    assert out == "error: unknown tool home__nope"


async def test_failing_source_does_not_break_others():
    broken = FakeSource("broken", ["x"], fail_start=True)
    healthy = FakeSource("home", ["set_light"])
    hub = ToolHub([broken, healthy])

    # One source raising in start() must NOT prevent the others from loading.
    await hub.start()

    names = [t["function"]["name"] for t in hub.tools]
    # The broken source advertised nothing extra; the healthy source still serves.
    assert "home__set_light" in names
    assert healthy.started is True
    # The healthy source is still callable.
    assert await hub.call("home__set_light", {}) == "home:set_light"


async def test_source_dicts_are_not_mutated():
    home = FakeSource("home", ["set_light"])
    hub = ToolHub([home])
    await hub.start()

    # The hub clones tool dicts before prefixing, so the source keeps the raw name.
    assert home.raw_tools()[0]["function"]["name"] == "set_light"
    assert hub.tools[0]["function"]["name"] == "home__set_light"


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
