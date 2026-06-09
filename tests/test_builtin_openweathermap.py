from datetime import date

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


async def test_server_advertises_get_weather_forecast():
    server = build_openweathermap_server(object(), "key", "Moscow")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    tools = source.raw_tools()
    names = [t["function"]["name"] for t in tools]
    assert "get_weather_forecast" in names

    tool = next(t for t in tools if t["function"]["name"] == "get_weather_forecast")
    assert tool["type"] == "function"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert "date" in params["properties"]
    assert "city" in params["properties"]
    assert "date" in params.get("required", [])


async def test_forecast_call_returns_text(monkeypatch):
    async def fake_forecast(client, city, api_key, target_date):
        assert isinstance(target_date, date)
        assert target_date == date(2030, 1, 2)
        assert city == "Moscow"
        return "Прогноз на 2030-01-02: 5 градусов, ясно"

    monkeypatch.setattr(owm_mcp, "get_weather_forecast", fake_forecast)

    server = build_openweathermap_server(object(), "key", "DefaultCity")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    out = await source.call("get_weather_forecast", {"date": "2030-01-02", "city": "Moscow"})
    assert out == "Прогноз на 2030-01-02: 5 градусов, ясно"


async def test_forecast_invalid_date_returns_error(monkeypatch):
    async def fake_forecast(client, city, api_key, target_date):  # pragma: no cover - must not be reached
        raise AssertionError("get_weather_forecast must not be called for an invalid date")

    monkeypatch.setattr(owm_mcp, "get_weather_forecast", fake_forecast)

    server = build_openweathermap_server(object(), "key", "Moscow")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    out = await source.call("get_weather_forecast", {"date": "not-a-date"})
    assert out == "Неверный формат даты. Ожидается ГГГГ-ММ-ДД."


async def test_forecast_no_data_path(monkeypatch):
    async def fake_forecast(client, city, api_key, target_date):
        return None

    monkeypatch.setattr(owm_mcp, "get_weather_forecast", fake_forecast)

    server = build_openweathermap_server(object(), "key", "Moscow")
    source = BuiltinMcpSource("openweathermap", server)
    await source.start()

    out = await source.call("get_weather_forecast", {"date": "2030-01-02"})
    assert out == "Нет данных о погоде."
