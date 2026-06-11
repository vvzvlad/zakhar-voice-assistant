"""Unit tests for build_sources: the shared ToolHub source-list factory.

build_sources must produce the SAME set at boot (app.py) and on hot-reload (reconfig),
so these tests drive it directly off a real CoreConfig and assert the resulting source
ids/kinds for each gating combination.
"""

from src.core_config import CoreConfig
from src.tool_factory import build_sources


def _ids(sources):
    return [s.id for s in sources]


def _kinds(sources):
    return {s.id: s.kind for s in sources}


def test_build_sources_empty_config_yields_no_sources():
    # No mcp_servers, no OWM key, no calendar, no scheduler -> empty source list.
    core = CoreConfig()
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert sources == []


def test_build_sources_mcp_servers_one_http_source_each():
    # Each mcp_servers entry with url+name becomes one HttpMcpSource; the source id is
    # the (unique) server name.
    core = CoreConfig(mcp_servers=[
        {"name": "home", "url": "http://home/mcp"},
        {"name": "garage", "url": "http://garage/mcp", "token": "t"},
    ])
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert _ids(sources) == ["home", "garage"]
    assert _kinds(sources) == {"home": "http", "garage": "http"}


def test_build_sources_skips_mcp_entries_missing_url_or_name():
    # Entries without a url are skipped (a name without a url is not a usable server).
    core = CoreConfig(mcp_servers=[
        {"name": "ok", "url": "http://ok/mcp"},
        {"name": "no-url", "url": ""},
    ])
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert _ids(sources) == ["ok"]


def test_build_sources_openweathermap_gated_on_api_key():
    # An OWM api key adds a builtin "openweathermap" source; the captured http_cloud is
    # the proxied client the builtin uses.
    core = CoreConfig(openweathermap={"api_key": "k", "city": "Berlin"})
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert _ids(sources) == ["openweathermap"]
    assert _kinds(sources)["openweathermap"] == "builtin"


def test_build_sources_no_openweathermap_without_key():
    core = CoreConfig(openweathermap={"api_key": "", "city": "Berlin"})
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert sources == []


def test_build_sources_reminders_gated_on_scheduler():
    # scheduler is not None -> a builtin "reminders" source is added (build_reminders_server
    # only stores the scheduler, so any object satisfies it).
    core = CoreConfig()
    sources = build_sources(core, http_cloud=object(), scheduler=object())
    assert _ids(sources) == ["reminders"]
    assert _kinds(sources)["reminders"] == "builtin"


def test_build_sources_no_reminders_when_scheduler_none():
    core = CoreConfig()
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert "reminders" not in _ids(sources)


def test_build_sources_calendar_gated_on_url_and_username():
    # Calendar is gated on BOTH url AND username: a builtin "calendar" source is added
    # only when both are set; either missing omits it.
    both = CoreConfig(calendar={"url": "https://dav.example/cal", "username": "u",
                                "password": "p"})
    sources = build_sources(both, http_cloud=object(), scheduler=None)
    assert _ids(sources) == ["calendar"]
    assert _kinds(sources)["calendar"] == "builtin"

    no_user = CoreConfig(calendar={"url": "https://dav.example/cal", "username": ""})
    assert "calendar" not in _ids(build_sources(no_user, http_cloud=object(), scheduler=None))

    no_url = CoreConfig(calendar={"url": "", "username": "u"})
    assert "calendar" not in _ids(build_sources(no_url, http_cloud=object(), scheduler=None))


def test_build_sources_calendar_ordered_between_openweathermap_and_reminders():
    # In the combined order the calendar source sits between openweathermap and reminders.
    core = CoreConfig(
        openweathermap={"api_key": "k", "city": "Moscow"},
        calendar={"url": "https://dav.example/cal", "username": "u", "password": "p"},
    )
    sources = build_sources(core, http_cloud=object(), scheduler=object())
    assert _ids(sources) == ["openweathermap", "calendar", "reminders"]


def test_build_sources_combined_order_matches_boot():
    # Combined config: external MCP servers first, then openweathermap, then reminders —
    # the exact order app.py's boot path produces.
    core = CoreConfig(
        mcp_servers=[{"name": "home", "url": "http://home/mcp"}],
        openweathermap={"api_key": "k", "city": "Moscow"},
    )
    sources = build_sources(core, http_cloud=object(), scheduler=object())
    assert _ids(sources) == ["home", "openweathermap", "reminders"]


# --- enabled flag: the master switch drops a source from the ToolHub ----------


def test_build_sources_skips_disabled_mcp_server():
    # A fully configured external server with enabled=False is omitted; the
    # enabled one next to it still builds (missing flag defaults to True).
    core = CoreConfig(mcp_servers=[
        {"name": "home", "url": "http://home/mcp", "enabled": False},
        {"name": "garage", "url": "http://garage/mcp"},
    ])
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert _ids(sources) == ["garage"]


def test_build_sources_skips_disabled_openweathermap():
    # enabled=False omits the OWM source even though the api key is set.
    core = CoreConfig(openweathermap={"enabled": False, "api_key": "k", "city": "Berlin"})
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert sources == []


def test_build_sources_skips_disabled_calendar():
    # enabled=False omits the calendar source even with full credentials.
    core = CoreConfig(calendar={"enabled": False, "url": "https://dav.example/cal",
                                "username": "u", "password": "p"})
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert sources == []


# --- slow declarations: the source factory bakes in who is slow ---------------


def _slow(sources):
    return {s.id: s.slow for s in sources}


def test_build_sources_builtin_slow_flags():
    # Weather and calendar do network round-trips -> slow (filler fires); the
    # reminders store is in-process -> fast (explicitly slow=False).
    core = CoreConfig(
        openweathermap={"api_key": "k", "city": "Moscow"},
        calendar={"url": "https://dav.example/cal", "username": "u", "password": "p"},
    )
    sources = build_sources(core, http_cloud=object(), scheduler=object())
    assert _slow(sources) == {
        "openweathermap": True,
        "calendar": True,
        "reminders": False,
    }


def test_build_sources_external_slow_from_config():
    # External servers pick up McpServerConfig.slow; the default is fast.
    core = CoreConfig(mcp_servers=[
        {"name": "home", "url": "http://home/mcp"},
        {"name": "websearch", "url": "http://search/mcp", "slow": True},
    ])
    sources = build_sources(core, http_cloud=object(), scheduler=None)
    assert _slow(sources) == {"home": False, "websearch": True}
