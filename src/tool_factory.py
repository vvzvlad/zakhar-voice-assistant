"""Build the ToolHub source list from the current config.

Reused at boot (app.py) and on hot-reload (reconfig). Keeps the source-construction
logic in one place so a runtime rebuild produces exactly the same set as boot.
"""

from src.builtin_mcp.openweathermap import build_openweathermap_server
from src.mcp_client import McpToolHub
from src.tool_hub import BuiltinMcpSource, HttpMcpSource


def build_sources(core, http_cloud, scheduler):
    """Return the list of ToolSources for `core`.

    `http_cloud` is the proxied client the OpenWeatherMap built-in uses; `scheduler`
    (a ReminderScheduler or None) gates the reminders source — None omits it."""
    sources = []
    for srv in core.mcp_servers:
        if srv.url and srv.name:
            sources.append(HttpMcpSource(srv.name, McpToolHub(srv.url, srv.token or None, srv.transport)))
    if core.openweathermap.api_key:
        sources.append(BuiltinMcpSource(
            "openweathermap",
            build_openweathermap_server(http_cloud, core.openweathermap.api_key, core.openweathermap.city),
        ))
    if core.calendar.url and core.calendar.username:
        from src.builtin_mcp.calendar import CalendarClient, build_calendar_server
        cal_client = CalendarClient(core.calendar.url, core.calendar.username,
                                    core.calendar.password, core.calendar.calendar)
        sources.append(BuiltinMcpSource("calendar", build_calendar_server(cal_client)))
    if scheduler is not None:
        from src.builtin_mcp.reminders import build_reminders_server
        sources.append(BuiltinMcpSource("reminders", build_reminders_server(scheduler)))
    return sources
