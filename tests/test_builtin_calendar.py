"""Tests for the built-in CalDAV calendar MCP server.

Two layers, both fully offline (no network / no real CalDAV):
  * CalendarClient unit tests patch the module-level caldav.DAVClient with a MagicMock,
    so connection, calendar selection, search mapping, create and delete are exercised
    against fake DAV objects only.
  * Server/source tests wrap build_calendar_server(fake_client) in a BuiltinMcpSource and
    drive it through the real FastMCP list_tools/call_tool path (exercising asyncio.to_thread
    and the tuple-result normalization in BuiltinMcpSource).
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from caldav.lib.error import NotFoundError
from icalendar import Event as IEvent

import src.builtin_mcp.calendar as cal_mod
from src.builtin_mcp.calendar import CalendarClient, build_calendar_server
from src.tool_hub import BuiltinMcpSource


def _fake_calendar(name="Personal", url="https://dav.example/cal/personal/"):
    """A MagicMock standing in for a caldav Calendar with a name and url."""
    cal = MagicMock()
    cal.get_display_name.return_value = name
    cal.url = url
    return cal


def _fake_event(summary, start, end, uid, location=None):
    """A fake caldav Event exposing icalendar_component as a real icalendar Event."""
    comp = IEvent()
    comp.add("summary", summary)
    comp.add("dtstart", start)
    comp.add("dtend", end)
    comp.add("uid", uid)
    if location:
        comp.add("location", location)
    ev = MagicMock()
    ev.icalendar_component = comp
    return ev


# --- CalendarClient unit tests ----------------------------------------------


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_calendar_connects_with_creds(dav_client):
    cal = _fake_calendar()
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "alice", "secret")
    resolved = client._get_calendar()

    dav_client.assert_called_once_with(
        url="https://dav.example", username="alice", password="secret"
    )
    assert resolved is cal
    # Cached: a second call must not reconnect.
    assert client._get_calendar() is cal


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_calendar_selects_by_name(dav_client):
    work = _fake_calendar(name="Work")
    home = _fake_calendar(name="Home")
    dav_client.return_value.principal.return_value.calendars.return_value = [work, home]

    client = CalendarClient("u", "user", "pw", calendar_name="Home")
    assert client._get_calendar() is home


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_calendar_raises_when_none(dav_client):
    dav_client.return_value.principal.return_value.calendars.return_value = []

    client = CalendarClient("u", "user", "pw")
    try:
        client._get_calendar()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "no calendars" in str(e)


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_list_calendars_maps_name_and_url(dav_client):
    dav_client.return_value.principal.return_value.calendars.return_value = [
        _fake_calendar(name="Work", url="https://dav.example/work/"),
        _fake_calendar(name="Home", url="https://dav.example/home/"),
    ]

    client = CalendarClient("u", "user", "pw")
    assert client.list_calendars() == [
        {"name": "Work", "url": "https://dav.example/work/"},
        {"name": "Home", "url": "https://dav.example/home/"},
    ]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_events_maps_search_results(dav_client):
    cal = _fake_calendar()
    cal.search.return_value = [
        _fake_event(
            "Standup",
            datetime(2026, 6, 8, 10, 0),
            datetime(2026, 6, 8, 10, 30),
            "u1@zakhar",
            location="Room 1",
        )
    ]
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("u", "user", "pw")
    start, end = datetime(2026, 6, 8), datetime(2026, 6, 9)
    events = client.get_events(start, end)

    cal.search.assert_called_once_with(start=start, end=end, event=True, expand=True)
    assert events == [
        {
            "uid": "u1@zakhar",
            "summary": "Standup",
            "start": "2026-06-08T10:00:00",
            "end": "2026-06-08T10:30:00",
            "location": "Room 1",
        }
    ]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_events_skips_incomplete_components(dav_client):
    cal = _fake_calendar()
    # A component with no SUMMARY/DTSTART must be skipped, not crash.
    bad = MagicMock()
    bad.icalendar_component = IEvent()  # empty
    good = _fake_event(
        "Lunch", datetime(2026, 6, 8, 13, 0), datetime(2026, 6, 8, 14, 0), "g@zakhar"
    )
    cal.search.return_value = [bad, good]
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("u", "user", "pw")
    events = client.get_events(datetime(2026, 6, 8), datetime(2026, 6, 9))
    assert [e["summary"] for e in events] == ["Lunch"]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_create_event_saves_ical_with_summary_and_uid(dav_client):
    cal = _fake_calendar()
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("u", "user", "pw")
    result = client.create_event(
        "Dentist",
        datetime(2026, 6, 10, 9, 0),
        datetime(2026, 6, 10, 9, 30),
        description="checkup",
        location="Clinic",
    )

    cal.add_event.assert_called_once()
    ical_text = cal.add_event.call_args.args[0]
    assert "Dentist" in ical_text
    assert result["uid"] in ical_text
    assert "UID" in ical_text
    # DTSTAMP is required by RFC 5545; guard against regressing the fix.
    assert "DTSTAMP" in ical_text
    assert result["summary"] == "Dentist"
    assert result["start"] == "2026-06-10T09:00:00"


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_delete_event_calls_delete(dav_client):
    cal = _fake_calendar()
    event = MagicMock()
    cal.get_event_by_uid.return_value = event
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("u", "user", "pw")
    result = client.delete_event("u1@zakhar")

    cal.get_event_by_uid.assert_called_once_with("u1@zakhar")
    event.delete.assert_called_once()
    assert result == {"deleted": True, "uid": "u1@zakhar"}


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_delete_event_not_found(dav_client):
    cal = _fake_calendar()
    cal.get_event_by_uid.side_effect = NotFoundError("missing")
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("u", "user", "pw")
    result = client.delete_event("nope@zakhar")
    assert result == {"deleted": False, "uid": "nope@zakhar", "error": "not found"}


# --- Server / BuiltinMcpSource tests ----------------------------------------


def _fake_client():
    """A MagicMock CalendarClient — never a real DAVClient (offline)."""
    client = MagicMock(spec=CalendarClient)
    client.get_today_events.return_value = [
        {
            "uid": "u1@zakhar",
            "summary": "Standup",
            "start": "2026-06-08T10:00:00",
            "end": "2026-06-08T10:30:00",
            "location": "",
        }
    ]
    client.list_calendars.return_value = [{"name": "Personal", "url": "https://x/"}]
    client.create_event.return_value = {
        "uid": "new@zakhar",
        "summary": "Meeting",
        "start": "2026-06-09T09:00:00",
        "end": "2026-06-09T10:00:00",
    }
    client.delete_event.return_value = {"deleted": True, "uid": "u1@zakhar"}
    return client


async def test_server_advertises_all_tools():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    names = [t["function"]["name"] for t in source.raw_tools()]
    for expected in (
        "get_today_events",
        "get_week_events",
        "get_events",
        "create_event",
        "delete_event",
        "list_calendars",
    ):
        assert expected in names

    # Groq shape sanity-check on one tool that takes arguments.
    get_events = next(
        t for t in source.raw_tools() if t["function"]["name"] == "get_events"
    )
    assert get_events["type"] == "function"
    params = get_events["function"]["parameters"]
    assert params["type"] == "object"
    assert "start" in params["properties"]
    assert "end" in params["properties"]


async def test_get_today_events_formats_text():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    out = await source.call("get_today_events", {})
    assert "Standup" in out
    assert "2026-06-08T10:00:00..2026-06-08T10:30:00" in out


async def test_get_today_events_empty():
    client = _fake_client()
    client.get_today_events.return_value = []
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_today_events", {})
    assert out == "На сегодня событий нет."


async def test_create_event_returns_uid():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call(
        "create_event",
        {"summary": "Meeting", "start": "2026-06-09T09:00:00", "end": "2026-06-09T10:00:00"},
    )
    assert "new@zakhar" in out
    assert "Meeting" in out
    # _parse_dt turned the ISO strings into datetimes before the client saw them.
    args = client.create_event.call_args.args
    assert args[0] == "Meeting"
    assert args[1] == datetime(2026, 6, 9, 9, 0)


async def test_delete_event_tool():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("delete_event", {"uid": "u1@zakhar"})
    client.delete_event.assert_called_once_with("u1@zakhar")
    assert "u1@zakhar" in out and "удал" in out.lower()


async def test_list_calendars_tool():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    out = await source.call("list_calendars", {})
    assert "Personal" in out


async def test_tool_error_returns_string_not_crash():
    client = _fake_client()
    client.get_today_events.side_effect = RuntimeError("dav down")
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_today_events", {})
    assert isinstance(out, str)
    assert "dav down" in out
