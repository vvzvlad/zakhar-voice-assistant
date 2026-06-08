"""Built-in weather MCP server (in-process FastMCP).

Exposes current weather as an on-demand tool instead of always injecting it into the
system prompt. The server closes over the proxied http client, the OWM api key and the
default city; the actual OWM call still lives in src.weather.get_weather_summary.
"""

from mcp.server.fastmcp import FastMCP

from src.weather import get_weather_summary


def build_weather_server(client, api_key: str, default_city: str) -> FastMCP:
    """Build a FastMCP server exposing a single get_current_weather tool.

    `client` is the shared (proxied) httpx client used for the OWM request; `api_key`
    and `default_city` come from core.weather. The returned server is wrapped in a
    BuiltinMcpSource by the caller.
    """
    mcp = FastMCP("weather")

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

    return mcp
