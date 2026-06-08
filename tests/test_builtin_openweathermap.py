import src.builtin_mcp.openweathermap as owm_mcp
from src.builtin_mcp.openweathermap import build_openweathermap_server
from src.tool_hub import BuiltinMcpSource


async def test_server_advertises_get_current_weather():
    server = build_openweathermap_server(object(), "key", "Moscow")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    tools = source.raw_tools()
    names = [t["function"]["name"] for t in tools]
    assert "get_current_weather" in names

    tool = next(t for t in tools if t["function"]["name"] == "get_current_weather")
    # Groq shape: a function with a parameters JSON schema (object).
    assert tool["type"] == "function"
    assert tool["function"]["parameters"]["type"] == "object"
    assert "city" in tool["function"]["parameters"]["properties"]


async def test_call_returns_weather_text(monkeypatch):
    async def fake_summary(client, city, api_key):
        assert city == "Moscow"
        return "10 градусов, ясно"

    monkeypatch.setattr(owm_mcp, "get_weather_summary", fake_summary)

    server = build_openweathermap_server(object(), "key", "DefaultCity")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    # Implicitly exercises the tuple-return normalization in BuiltinMcpSource.call.
    out = await source.call("get_current_weather", {"city": "Moscow"})
    assert out == "10 градусов, ясно"


async def test_call_no_data_path(monkeypatch):
    async def fake_summary(client, city, api_key):
        return None

    monkeypatch.setattr(owm_mcp, "get_weather_summary", fake_summary)

    server = build_openweathermap_server(object(), "key", "Moscow")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    out = await source.call("get_current_weather", {})
    assert out == "Нет данных о погоде."
