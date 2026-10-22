"""Build the ToolHub source list from the current config.

Called by the Reconfigurator for the boot connect (request_initial_tools_load) and for
every live tool-config change, so both build the set the same way.
"""

from src.builtin_mcp.openweathermap import build_openweathermap_server
from src.builtin_mcp.volume import build_volume_server
from src.mcp_client import McpToolHub
from src.tool_hub import BuiltinMcpSource, HttpMcpSource


def build_sources(core, http_cloud, scheduler, assistant_tasks, get_manager, memory):
    """Return the list of ToolSources for `core`.

    `http_cloud` is the proxied client the OpenWeatherMap built-in uses; `scheduler`
    (a ReminderScheduler or None) gates the reminders source — None omits it;
    `assistant_tasks` (an AssistantTasks or None) backs the assistant_tasks source,
    which also needs core.assistant_tasks enabled with a url and a secret;
    `get_manager` (a zero-arg callable returning the DeviceManager) backs the volume
    source, which is always present; `memory` (a MemoryStore or None) backs the memory
    source — None omits it."""
    sources = []
    for srv in core.mcp_servers:
        if srv.enabled and srv.url and srv.name:
            # The operator marks slow external servers (web search, ...) in config.
            sources.append(HttpMcpSource(srv.name, McpToolHub(srv.url, srv.token or None, srv.transport),
                                         slow=srv.slow))
    if core.openweathermap.enabled and core.openweathermap.api_key:
        # Weather hits the network: slow, warrants the early spoken filler.
        sources.append(BuiltinMcpSource(
            "openweathermap",
            build_openweathermap_server(http_cloud, core.openweathermap.api_key, core.openweathermap.city),
            slow=True,
        ))
    if core.calendar.enabled and core.calendar.url and core.calendar.username:
        from src.builtin_mcp.calendar import CalendarClient, build_calendar_server
        cal_client = CalendarClient(core.calendar.url, core.calendar.username,
                                    core.calendar.password, core.calendar.calendar)
        # CalDAV round-trips: slow, warrants the early spoken filler.
        sources.append(BuiltinMcpSource("calendar", build_calendar_server(cal_client), slow=True))
    if scheduler is not None:
        from src.builtin_mcp.reminders import build_reminders_server
        # Reminders are an in-process store: fast, no filler (explicit for clarity).
        sources.append(BuiltinMcpSource("reminders", build_reminders_server(scheduler), slow=False))
    cfg = core.assistant_tasks
    if assistant_tasks is not None and cfg.enabled and cfg.url and cfg.secret:
        from src.builtin_mcp.assistant_tasks import build_assistant_tasks_server
        # Handing off is one short webhook POST (ZeroClaw only queues the job): no filler.
        sources.append(BuiltinMcpSource("assistant_tasks", build_assistant_tasks_server(assistant_tasks),
                                        slow=False))
    # One message to the speaker: fast, no filler.
    sources.append(BuiltinMcpSource("volume", build_volume_server(get_manager), slow=False))
    if memory is not None:
        from src.builtin_mcp.memory import build_memory_server
        # Memory is an in-process file: fast, no filler.
        sources.append(BuiltinMcpSource("memory", build_memory_server(memory), slow=False))
    return sources
