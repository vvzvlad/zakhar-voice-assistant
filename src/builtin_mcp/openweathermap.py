"""Built-in OpenWeatherMap MCP server (in-process FastMCP).

Exposes two on-demand tools instead of always injecting weather into the system
prompt: current weather and a forecast for a specific date. The server closes over
the proxied http client, the OWM api key and the default city; the actual OWM calls
still live in src.openweathermap.get_weather_summary / get_weather_forecast.
"""

from datetime import date as date_cls

from mcp.server.fastmcp import FastMCP

from src.openweathermap import get_weather_forecast, get_weather_summary


def build_openweathermap_server(client, api_key: str, default_city: str) -> FastMCP:
    """Build a FastMCP server exposing get_current_weather and get_weather_forecast tools.

    `client` is the shared (proxied) httpx client used for the OWM request; `api_key`
    and `default_city` come from core.openweathermap. The returned server is wrapped in
    a BuiltinMcpSource by the caller.
    """
    mcp = FastMCP("openweathermap")

    @mcp.tool(
        name="get_current_weather",
        description=(
            "Текущая погода: температура, осадки, ветер. "
            "Аргумент city — город (если не указан, берётся город по умолчанию)."
        ),
    )
    async def get_current_weather(city: str | None = None) -> str:
        summary = await get_weather_summary(client, city or default_city, api_key)
        return summary or "Нет данных о погоде."

    @mcp.tool(
        name="get_weather_forecast",
        description=(
            "Прогноз погоды на конкретную дату (до 5 дней вперёд): температура, "
            "осадки, ветер. Аргумент date — дата в формате ГГГГ-ММ-ДД. "
            "Аргумент city — город (если не указан, берётся город по умолчанию)."
        ),
    )
    async def get_weather_forecast_tool(date: str, city: str | None = None) -> str:
        try:
            # `date` is the tool argument; use the aliased class to parse it.
            target = date_cls.fromisoformat(date)
        except (ValueError, TypeError):
            return "Неверный формат даты. Ожидается ГГГГ-ММ-ДД."
        summary = await get_weather_forecast(client, city or default_city, api_key, target)
        return summary or "Нет данных о погоде."

    return mcp
