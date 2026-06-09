"""Tests for the built-in CalDAV calendar MCP server.

Two layers, both fully offline (no network / no real CalDAV):
  * CalendarClient unit tests patch the module-level caldav.DAVClient with a MagicMock,
    so connection, calendar selection, search mapping, create and delete are exercised
    against fake DAV objects only.
  * Server/source tests wrap build_calendar_server(fake_client) in a BuiltinMcpSource and
    drive it through the real FastMCP list_tools/call_tool path (exercising asyncio.to_thread
    and the tuple-result normalization in BuiltinMcpSource).
"""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from caldav.lib.error import NotFoundError
from icalendar import Calendar as ICalendar
from icalendar import Event as IEvent

from src.builtin_mcp.calendar import (
    CalendarClient,
    _format_rrule,
    _is_yandex,
    build_calendar_server,
)
from src.tool_hub import BuiltinMcpSource


def _ical_text(*events) -> bytes:
    """Serialize one or more icalendar Events into a VCALENDAR byte string."""
    cal = ICalendar()
    cal.add("version", "2.0")
    cal.add("prodid", "-//zakhar//test//EN")
    for ev in events:
        cal.add_component(ev)
    return cal.to_ical()


def _component_from(ev) -> IEvent:
    """Round-trip an Event through serialization, returning the parsed VEVENT.

    Used so parse-side tests see the same objects icalendar produces when reading a
    component back off the wire.
    """
    parsed = ICalendar.from_ical(_ical_text(ev))
    return next(iter(parsed.walk("VEVENT")))


def _fake_calendar(name="Personal", url="https://dav.example/cal/personal/"):
    """A MagicMock standing in for a caldav Calendar with a name and url."""
    cal = MagicMock()
    cal.get_display_name.return_value = name
    cal.url = url
    return cal


def _fake_event(summary, start, end, uid, location=None, comp=None):
    """A fake caldav Event exposing icalendar_component as a real icalendar Event.

    Pass a prebuilt `comp` to wrap an arbitrary VEVENT; otherwise a minimal one is
    built from summary/start/end/uid/location.
    """
    if comp is None:
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


# --- Pure helper unit tests (no mocks) --------------------------------------


def test_format_rrule_freq_keyword():
    assert _format_rrule("weekly") == {"FREQ": ["WEEKLY"]}
    assert _format_rrule("DAILY") == {"FREQ": ["DAILY"]}


def test_format_rrule_full_string():
    assert _format_rrule("FREQ=WEEKLY;INTERVAL=2;COUNT=10;BYDAY=MO,WE") == {
        "FREQ": ["WEEKLY"],
        "INTERVAL": [2],
        "COUNT": [10],
        "BYDAY": ["MO", "WE"],
    }


def test_format_rrule_dict():
    assert _format_rrule({"freq": "monthly", "interval": 3}) == {
        "FREQ": ["MONTHLY"],
        "INTERVAL": [3],
    }


def test_format_rrule_unknown_freq_raises():
    for bad in ("nope", "FREQ=FORTNIGHTLY", {"interval": 2}):
        try:
            _format_rrule(bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_format_rrule_serializes_into_event():
    # The dict must be usable directly by event.add("rrule", ...).
    ev = IEvent()
    ev.add("rrule", _format_rrule("FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE"))
    text = _ical_text(ev).decode()
    assert "RRULE:FREQ=WEEKLY" in text
    assert "INTERVAL=2" in text
    assert "BYDAY=MO,WE" in text


def test_format_rrule_until_compact_datetime():
    # iCalendar compact UNTIL must become a tz-aware datetime (UTC), not a string,
    # otherwise icalendar's to_ical() raises TypeError when serializing the RRULE.
    result = _format_rrule("FREQ=DAILY;UNTIL=20261231T000000Z")
    until = result["UNTIL"][0]
    assert isinstance(until, datetime)
    assert until.tzinfo is not None
    assert until == datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc)


def test_format_rrule_until_iso_datetime():
    # ISO form with a trailing Z is also normalized to a tz-aware UTC datetime.
    result = _format_rrule("FREQ=DAILY;UNTIL=2026-12-31T00:00:00Z")
    until = result["UNTIL"][0]
    assert isinstance(until, datetime)
    assert until == datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc)


def test_format_rrule_until_date_only():
    # A date-only UNTIL (compact or ISO) yields a plain date, not a datetime.
    compact = _format_rrule("FREQ=DAILY;UNTIL=20261231")["UNTIL"][0]
    assert type(compact) is date
    assert compact == date(2026, 12, 31)
    iso = _format_rrule("FREQ=DAILY;UNTIL=2026-12-31")["UNTIL"][0]
    assert type(iso) is date
    assert iso == date(2026, 12, 31)


def test_format_rrule_until_existing_datetime_passthrough():
    # A dict that already carries UNTIL as a datetime/date is left untouched.
    dt = datetime(2026, 12, 31, tzinfo=timezone.utc)
    assert _format_rrule({"FREQ": "DAILY", "UNTIL": dt})["UNTIL"] == [dt]


def test_format_rrule_empty_freq_raises():
    # An empty FREQ (e.g. "FREQ=;INTERVAL=2") must raise ValueError, not IndexError.
    try:
        _format_rrule("FREQ=;INTERVAL=2")
        assert False, "expected ValueError for empty FREQ"
    except ValueError as e:
        assert "FREQ" in str(e)


def test_is_yandex():
    assert _is_yandex("https://caldav.yandex.ru/") is True
    assert _is_yandex("https://CalDAV.Yandex.RU/dav/") is True
    assert _is_yandex("https://dav.example.com/") is False
    assert _is_yandex("") is False


# --- create_event rich-feature client tests ---------------------------------


def _saved_ical(create_kwargs, url="https://dav.example"):
    """Run create_event against a mocked DAVClient and return the saved ical text."""
    with patch("src.builtin_mcp.calendar.caldav.DAVClient") as dav_client:
        cal = _fake_calendar()
        dav_client.return_value.principal.return_value.calendars.return_value = [cal]
        client = CalendarClient(url, "user", "pw")
        result = client.create_event(**create_kwargs)
        return cal.add_event.call_args.args[0], result


def test_create_event_with_reminders_emits_valarms():
    ical, _ = _saved_ical(
        {
            "summary": "Standup",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 9, 30),
            "reminders": [15, {"minutes": 5, "action": "AUDIO"}],
        }
    )
    assert ical.count("BEGIN:VALARM") == 2
    assert "ACTION:DISPLAY" in ical
    assert "ACTION:AUDIO" in ical
    assert "TRIGGER:-PT15M" in ical
    assert "TRIGGER:-PT5M" in ical


def test_create_event_with_priority():
    ical, _ = _saved_ical(
        {
            "summary": "Plan",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 10, 0),
            "priority": 2,
        }
    )
    assert "PRIORITY:2" in ical


def test_create_event_priority_clamped():
    ical, _ = _saved_ical(
        {
            "summary": "Plan",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 10, 0),
            "priority": 99,
        }
    )
    assert "PRIORITY:9" in ical


def test_create_event_with_recurrence():
    ical, _ = _saved_ical(
        {
            "summary": "Weekly",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 10, 0),
            "recurrence": "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE",
        }
    )
    assert "RRULE:FREQ=WEEKLY" in ical
    assert "INTERVAL=2" in ical
    assert "BYDAY=MO,WE" in ical


def test_create_event_recurrence_with_until_serializes():
    # End-to-end guard: a recurrence with UNTIL must reach add_event with a real
    # ical body. If UNTIL were left as a string, cal.to_ical() raises TypeError and
    # add_event is never called — so reaching this assertion proves the fix.
    cal_obj = None
    with patch("src.builtin_mcp.calendar.caldav.DAVClient") as dav_client:
        cal_obj = _fake_calendar()
        dav_client.return_value.principal.return_value.calendars.return_value = [cal_obj]
        client = CalendarClient("https://dav.example", "user", "pw")
        client.create_event(
            "Daily",
            datetime(2026, 6, 10, 9, 0),
            datetime(2026, 6, 10, 10, 0),
            recurrence="FREQ=DAILY;UNTIL=20261231T000000Z",
        )
    # add_event was actually called => to_ical() did not raise.
    cal_obj.add_event.assert_called_once()
    ical = cal_obj.add_event.call_args.args[0]
    assert "RRULE" in ical
    assert "FREQ=DAILY" in ical
    assert "UNTIL=20261231" in ical


def test_create_event_default_times():
    # Omitting start/end yields start = next full hour, end = start + 1h.
    # Capture "now" at call time and assert the result is consistent with it rather
    # than monkeypatching datetime (which would also break DTSTAMP / fromisoformat).
    before = datetime.now().astimezone()
    ical, result = _saved_ical({"summary": "NoTimes"})
    after = datetime.now().astimezone()

    start = datetime.fromisoformat(result["start"])
    end = datetime.fromisoformat(result["end"])
    # Start is on a full hour, within one hour after "now", and 1h before end.
    assert start.minute == 0 and start.second == 0 and start.microsecond == 0
    expected_floor = before.replace(minute=0, second=0, microsecond=0)
    assert expected_floor + timedelta(hours=1) <= start <= after + timedelta(hours=1)
    assert (end - start).total_seconds() == 3600
    assert "DTSTART" in ical


def test_create_event_all_day_default_end():
    # A bare date start with no end -> all-day VEVENT: DTSTART/DTEND as VALUE=DATE,
    # end defaulting to the next day (RFC 5545 DTEND is exclusive).
    ical, result = _saved_ical({"summary": "Holiday", "start": date(2026, 6, 10)})
    assert "DTSTART;VALUE=DATE:20260610" in ical
    assert "DTEND;VALUE=DATE:20260611" in ical
    assert result["start"] == "2026-06-10"
    assert result["end"] == "2026-06-11"


def test_create_event_all_day_explicit_multiday():
    ical, _ = _saved_ical(
        {"summary": "Trip", "start": date(2026, 6, 10), "end": date(2026, 6, 12)}
    )
    assert "DTSTART;VALUE=DATE:20260610" in ical
    assert "DTEND;VALUE=DATE:20260612" in ical


def test_create_event_all_day_zero_length_bumped():
    # end == start would be a zero-length all-day event (rejected by servers); the
    # client bumps DTEND to the next day.
    ical, _ = _saved_ical(
        {"summary": "Day", "start": date(2026, 6, 10), "end": date(2026, 6, 10)}
    )
    assert "DTEND;VALUE=DATE:20260611" in ical


def test_create_event_all_day_yandex_keeps_date():
    # All-day events have no time to normalize; the Yandex UTC path must not turn a
    # date into a datetime (which would emit a timed DTSTART).
    ical, _ = _saved_ical(
        {"summary": "Y all-day", "start": date(2026, 6, 10)},
        url="https://caldav.yandex.ru/",
    )
    dtstart = next(l for l in ical.splitlines() if l.startswith("DTSTART"))
    assert "DTSTART;VALUE=DATE:20260610" in ical
    # The serialized value (after the ':') must carry no time component: an all-day
    # date is "20260610", a timed value would be "20260610T000000Z". (The "T" in the
    # property name "DTSTART" itself is irrelevant.)
    assert "T" not in dtstart.split(":", 1)[1]


def test_create_event_escaping_round_trip():
    # Special chars (; , \\ and newline) must survive create -> parse via icalendar,
    # which handles RFC 5545 escaping itself (no manual escape helper).
    tricky_summary = "Meet; with, a\\backslash"
    tricky_desc = "line one\nline two; still, here"
    tricky_loc = "Room, B; floor\\3"
    ical, result = _saved_ical(
        {
            "summary": tricky_summary,
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 10, 0),
            "description": tricky_desc,
            "location": tricky_loc,
        }
    )
    # Parse the saved text back and confirm the original values are recovered intact.
    parsed = ICalendar.from_ical(ical)
    comp = next(iter(parsed.walk("VEVENT")))
    assert str(comp.get("SUMMARY")) == tricky_summary
    assert str(comp.get("DESCRIPTION")) == tricky_desc
    assert str(comp.get("LOCATION")) == tricky_loc


def test_create_event_yandex_serializes_utc():
    # Yandex client normalizes times to UTC so DTSTART/DTEND carry a trailing Z.
    ical, _ = _saved_ical(
        {
            "summary": "Y",
            "start": datetime(2026, 6, 10, 9, 0, tzinfo=timezone.utc),
            "end": datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc),
        },
        url="https://caldav.yandex.ru/",
    )
    dtstart = next(l for l in ical.splitlines() if l.startswith("DTSTART"))
    dtend = next(l for l in ical.splitlines() if l.startswith("DTEND"))
    assert dtstart.endswith("Z")
    assert dtend.endswith("Z")
    assert dtstart == "DTSTART:20260610T090000Z"


def test_create_event_timed_start_date_end_promoted_to_datetime():
    # Public-method contract: a timed (datetime) start with a bare `date` end must not
    # crash and must not emit a VALUE=DATE end on a timed event. The date end is
    # promoted to midnight; here it lands after start, so it is kept as the end.
    ical, result = _saved_ical(
        {
            "summary": "Mixed types",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": date(2026, 6, 12),
        }
    )
    assert "DTSTART:20260610T090000" in ical
    assert "DTEND:20260612T000000" in ical
    assert "VALUE=DATE" not in ical


def test_create_event_timed_start_date_end_before_start_bumped():
    # A date end that promotes to a midnight at/before start bumps to a 1-hour event.
    ical, result = _saved_ical(
        {
            "summary": "Degenerate",
            "start": datetime(2026, 6, 12, 9, 0),
            "end": date(2026, 6, 12),
        }
    )
    start = datetime.fromisoformat(result["start"])
    end = datetime.fromisoformat(result["end"])
    assert (end - start) == timedelta(hours=1)
    assert "VALUE=DATE" not in ical


def test_create_event_timed_end_before_start_bumped():
    # A timed start with an end at or before it (e.g. a date-only end widened to
    # midnight on the same day) must fall back to a 1-hour event, not a negative one.
    ical, result = _saved_ical(
        {
            "summary": "Bad end",
            "start": datetime(2026, 6, 12, 9, 0),
            "end": datetime(2026, 6, 12, 0, 0),
        }
    )
    start = datetime.fromisoformat(result["start"])
    end = datetime.fromisoformat(result["end"])
    assert (end - start) == timedelta(hours=1)
    assert "DTEND:20260612T100000" in ical


def test_create_event_timed_end_equals_start_bumped():
    ical, result = _saved_ical(
        {
            "summary": "Zero len",
            "start": datetime(2026, 6, 12, 9, 0),
            "end": datetime(2026, 6, 12, 9, 0),
        }
    )
    end = datetime.fromisoformat(result["end"])
    start = datetime.fromisoformat(result["start"])
    assert (end - start) == timedelta(hours=1)


def test_create_event_timed_end_aware_start_naive_end_bumped():
    # tz-awareness mismatch (aware start, naive end <= start) must not raise; the
    # non-positive duration guard falls back to a wall-clock comparison and bumps.
    ical, result = _saved_ical(
        {
            "summary": "Mixed tz",
            "start": datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc),
            "end": datetime(2026, 6, 12, 0, 0),
        }
    )
    end = datetime.fromisoformat(result["end"])
    start = datetime.fromisoformat(result["start"])
    assert (end - start) == timedelta(hours=1)


def test_create_event_all_day_reminder_anchored_to_morning():
    # All-day reminders are anchored to 09:00 on the event day: 15 min before fires
    # at 08:45 (trigger = 9h - 15m relative to midnight DTSTART).
    ical, _ = _saved_ical(
        {"summary": "All day", "start": date(2026, 6, 10), "reminders": [15]}
    )
    assert "DTSTART;VALUE=DATE:20260610" in ical
    assert "TRIGGER:PT8H45M" in ical


def test_create_event_all_day_reminder_zero_minutes_is_nine_am():
    ical, _ = _saved_ical(
        {"summary": "All day", "start": date(2026, 6, 10), "reminders": [0]}
    )
    assert "TRIGGER:PT9H" in ical


def test_create_event_all_day_reminder_full_day_before():
    ical, _ = _saved_ical(
        {"summary": "All day", "start": date(2026, 6, 10), "reminders": [1440]}
    )
    # 1440 min (1 day) before 09:00 == 09:00 the previous day == 15h before midnight.
    assert "TRIGGER:-PT15H" in ical


def test_create_event_timed_reminder_unchanged():
    # Regression: timed events keep "N minutes before start" semantics.
    ical, _ = _saved_ical(
        {
            "summary": "Standup",
            "start": datetime(2026, 6, 10, 9, 0),
            "end": datetime(2026, 6, 10, 9, 30),
            "reminders": [15],
        }
    )
    assert "TRIGGER:-PT15M" in ical


# --- get_event_by_uid client tests ------------------------------------------


def _rich_component():
    """A VEVENT carrying location, description and priority, round-tripped."""
    ev = IEvent()
    ev.add("summary", "Review")
    ev.add("uid", "rich@zakhar")
    ev.add("dtstart", datetime(2026, 6, 10, 9, 0))
    ev.add("dtend", datetime(2026, 6, 10, 10, 0))
    ev.add("location", "Room 5")
    ev.add("description", "quarterly review")
    ev.add("priority", 3)
    return _component_from(ev)


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_event_by_uid_found_full_dict(dav_client):
    cal = _fake_calendar()
    cal.get_event_by_uid.return_value = _fake_event(
        None, None, None, None, comp=_rich_component()
    )
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    ev = client.get_event_by_uid("rich@zakhar")

    cal.get_event_by_uid.assert_called_once_with("rich@zakhar")
    assert ev["uid"] == "rich@zakhar"
    assert ev["summary"] == "Review"
    assert ev["location"] == "Room 5"
    assert ev["description"] == "quarterly review"
    assert ev["priority"] == 3
    assert "categories" not in ev
    assert "attendees" not in ev


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_event_by_uid_not_found(dav_client):
    cal = _fake_calendar()
    cal.get_event_by_uid.side_effect = NotFoundError("missing")
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    assert client.get_event_by_uid("nope@zakhar") == {
        "found": False,
        "uid": "nope@zakhar",
    }


# --- search_events client tests ---------------------------------------------


def _search_corpus():
    """Two events with distinct summary/location for filter tests."""
    a = _fake_event(
        "Dentist appointment",
        datetime(2026, 6, 10, 9, 0),
        datetime(2026, 6, 10, 9, 30),
        "a@zakhar",
        location="Clinic",
    )
    b = _fake_event(
        "Team lunch",
        datetime(2026, 6, 11, 13, 0),
        datetime(2026, 6, 11, 14, 0),
        "b@zakhar",
        location="Cafe Downtown",
    )
    return [a, b]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_search_events_filters_summary(dav_client):
    cal = _fake_calendar()
    cal.search.return_value = _search_corpus()
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    out = client.search_events("dentist")
    assert [e["summary"] for e in out] == ["Dentist appointment"]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_search_events_filters_location(dav_client):
    cal = _fake_calendar()
    cal.search.return_value = _search_corpus()
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    out = client.search_events("downtown")
    assert [e["summary"] for e in out] == ["Team lunch"]


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_search_events_default_range(dav_client):
    cal = _fake_calendar()
    cal.search.return_value = []
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    client.search_events("anything")

    # No explicit dates -> default window roughly now-30d .. now+90d.
    kwargs = cal.search.call_args.kwargs
    span = kwargs["end"] - kwargs["start"]
    assert 119 <= span.days <= 121


# --- gap #1: all-day events ---------------------------------------------------


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_get_events_all_day_date_only(dav_client):
    # A date-only DTSTART (all-day event) must map without crashing.
    cal = _fake_calendar()
    ev = IEvent()
    ev.add("summary", "Birthday")
    ev.add("uid", "ad@zakhar")
    ev.add("dtstart", date(2026, 6, 10))
    ev.add("dtend", date(2026, 6, 11))
    cal.search.return_value = [_fake_event(None, None, None, None, comp=ev)]
    dav_client.return_value.principal.return_value.calendars.return_value = [cal]

    client = CalendarClient("https://dav.example", "user", "pw")
    events = client.get_events(datetime(2026, 6, 10), datetime(2026, 6, 11))
    assert events[0]["summary"] == "Birthday"
    assert events[0]["start"] == "2026-06-10"
    assert events[0]["end"] == "2026-06-11"


# --- gap #2: connect failure surfaces, tool returns error string -------------


@patch("src.builtin_mcp.calendar.caldav.DAVClient")
def test_connect_failure_surfaces_in_client(dav_client):
    dav_client.side_effect = ConnectionError("dns boom")
    client = CalendarClient("https://dav.example", "user", "pw")
    try:
        client.get_today_events()
        assert False, "expected the connection error to surface"
    except ConnectionError as e:
        assert "dns boom" in str(e)


async def test_connect_failure_tool_returns_error_string():
    # A real CalendarClient whose DAVClient raises on connect: the tool must catch it
    # and return a Russian error string rather than crashing.
    with patch("src.builtin_mcp.calendar.caldav.DAVClient") as dav_client:
        dav_client.side_effect = ConnectionError("dns boom")
        client = CalendarClient("https://dav.example", "user", "pw")
        source = BuiltinMcpSource("calendar", build_calendar_server(client))
        await source.start()
        out = await source.call("get_today_events", {})
    assert isinstance(out, str)
    assert "dns boom" in out


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
    client.get_week_events.return_value = [
        {
            "uid": "w1@zakhar",
            "summary": "Planning",
            "start": "2026-06-09T11:00:00",
            "end": "2026-06-09T12:00:00",
            "location": "Room 2",
        }
    ]
    client.get_events.return_value = [
        {
            "uid": "r1@zakhar",
            "summary": "Range event",
            "start": "2026-06-09T11:00:00",
            "end": "2026-06-09T12:00:00",
            "location": "",
        }
    ]
    client.search_events.return_value = [
        {
            "uid": "s1@zakhar",
            "summary": "Found event",
            "start": "2026-06-09T11:00:00",
            "end": "2026-06-09T12:00:00",
            "location": "Hall",
        }
    ]
    client.get_event_by_uid.return_value = {
        "uid": "rich@zakhar",
        "summary": "Review",
        "start": "2026-06-10T09:00:00",
        "end": "2026-06-10T10:00:00",
        "location": "Room 5",
        "description": "quarterly review",
        "priority": 3,
    }
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
        "search_events",
        "get_event_by_uid",
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
    assert "понедельник, 8 июня 2026, 10:00–10:30" in out


async def test_all_day_single_day_event_collapses():
    # All-day events carry an exclusive DTEND (2026-06-14 == day after 06-13), so a
    # single-day all-day event must collapse to just the start date, not span two days.
    client = MagicMock(spec=CalendarClient)
    client.get_week_events.return_value = [
        {
            "uid": "a1",
            "summary": "ДР Хлои",
            "start": "2026-06-13",
            "end": "2026-06-14",
            "location": "",
        }
    ]
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_week_events", {})
    assert "ДР Хлои — суббота, 13 июня 2026, весь день" in out


async def test_all_day_multi_day_event_spans():
    # Exclusive end 06-16 -> inclusive end 06-15: a real multi-day all-day span.
    client = MagicMock(spec=CalendarClient)
    client.get_week_events.return_value = [
        {
            "uid": "a2",
            "summary": "Отпуск",
            "start": "2026-06-13",
            "end": "2026-06-16",
            "location": "",
        }
    ]
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_week_events", {})
    # Both endpoints carry the year (per _ru_date), so the inclusive span reads in full.
    assert "суббота, 13 июня 2026 – понедельник, 15 июня 2026, весь день" in out


async def test_timed_event_no_timezone_conversion():
    # A tz-aware start (20:00+03:00) must render as 20:00 — the wall-clock time the user
    # set — with no conversion to UTC or any other zone.
    client = MagicMock(spec=CalendarClient)
    client.get_week_events.return_value = [
        {
            "uid": "a3",
            "summary": "Счётчики",
            "start": "2026-06-15T20:00:00+03:00",
            "end": "2026-06-15T22:00:00+03:00",
            "location": "",
        }
    ]
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_week_events", {})
    assert "понедельник, 15 июня 2026, 20:00–22:00" in out


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


async def test_create_event_tool_all_day_from_date_string():
    # A date-only string (no time) must reach the client as a `date`, not a midnight
    # datetime, so the event is created all-day.
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    await source.call("create_event", {"summary": "Vacation", "start": "2026-06-10"})
    args = client.create_event.call_args.args
    assert args[1] == date(2026, 6, 10)
    assert not isinstance(args[1], datetime)


async def test_create_event_tool_timed_start_date_end_stays_timed():
    # Mixed input: a timed start with a date-only end. The end must reach the client
    # as a datetime (midnight), not a bare date — otherwise a timed DTSTART would pair
    # with a DTEND;VALUE=DATE, which RFC 5545 forbids.
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    await source.call(
        "create_event",
        {"summary": "Mixed", "start": "2026-06-10T09:00:00", "end": "2026-06-12"},
    )
    args = client.create_event.call_args.args
    assert args[1] == datetime(2026, 6, 10, 9, 0)
    assert args[2] == datetime(2026, 6, 12, 0, 0)
    assert isinstance(args[2], datetime)


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


# --- gap #3: tool-level coverage for get_events / get_week_events ------------


async def test_get_week_events_tool():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    out = await source.call("get_week_events", {})
    assert "Planning" in out
    assert "Room 2" in out


async def test_get_week_events_tool_empty():
    client = _fake_client()
    client.get_week_events.return_value = []
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_week_events", {})
    assert out == "На ближайшую неделю событий нет."


async def test_get_events_tool_parses_dates():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call(
        "get_events", {"start": "2026-06-09", "end": "2026-06-10"}
    )
    assert "Range event" in out
    args = client.get_events.call_args.args
    assert args[0] == datetime(2026, 6, 9)
    # Date-only end is advanced to the next midnight (whole day inclusive).
    assert args[1] == datetime(2026, 6, 11)


async def test_get_events_tool_bad_dates():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    out = await source.call("get_events", {"start": "not-a-date", "end": "also-bad"})
    assert "Не понял даты" in out


async def test_get_events_tool_same_day_range_covers_whole_day():
    # Regression: a single-day query (start == end, date-only) must search the
    # whole day, not a zero-width [start, end) window that finds nothing.
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    await source.call("get_events", {"start": "2026-06-10", "end": "2026-06-10"})
    args = client.get_events.call_args.args
    assert args[0] == datetime(2026, 6, 10)
    assert args[1] == datetime(2026, 6, 11)


async def test_get_events_tool_datetime_end_kept_exact():
    # An END that carries an explicit time must be used verbatim, not advanced.
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    await source.call(
        "get_events",
        {"start": "2026-06-10T08:00:00", "end": "2026-06-10T18:00:00"},
    )
    args = client.get_events.call_args.args
    assert args[1] == datetime(2026, 6, 10, 18, 0, 0)


# --- search_events tool ------------------------------------------------------


async def test_search_events_tool():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("search_events", {"query": "found"})
    assert "Found event" in out
    args = client.search_events.call_args.args
    assert args[0] == "found"
    # Dates omitted -> the client receives None (it applies its own default range).
    assert args[1] is None and args[2] is None


async def test_search_events_tool_with_dates():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    await source.call(
        "search_events",
        {"query": "x", "start": "2026-06-01", "end": "2026-06-30"},
    )
    args = client.search_events.call_args.args
    assert args[1] == datetime(2026, 6, 1)
    # Date-only end is advanced to the next midnight (whole day inclusive).
    assert args[2] == datetime(2026, 7, 1)


async def test_search_events_tool_empty():
    client = _fake_client()
    client.search_events.return_value = []
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("search_events", {"query": "nothing"})
    assert "nothing" in out


# --- get_event_by_uid tool ---------------------------------------------------


async def test_get_event_by_uid_tool_found():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_event_by_uid", {"uid": "rich@zakhar"})
    client.get_event_by_uid.assert_called_once_with("rich@zakhar")
    assert "Review" in out
    assert "Room 5" in out
    assert "quarterly review" in out


async def test_get_event_by_uid_tool_not_found():
    client = _fake_client()
    client.get_event_by_uid.return_value = {"found": False, "uid": "nope@zakhar"}
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("get_event_by_uid", {"uid": "nope@zakhar"})
    assert "nope@zakhar" in out
    assert "не найден" in out.lower()


# --- create_event tool with the new optional params --------------------------


async def test_create_event_tool_with_rich_params():
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call(
        "create_event",
        {
            "summary": "Planning",
            "start": "2026-06-09T09:00:00",
            "end": "2026-06-09T10:00:00",
            "description": "agenda",
            "location": "Room 1",
            "priority": 2,
            "recurrence": "WEEKLY",
            "reminders": [10, 30],
        },
    )
    assert "new@zakhar" in out
    # All optional params reach the client positionally in the documented order:
    # summary, start, end, description, location, priority, reminders, recurrence.
    args = client.create_event.call_args.args
    assert args[0] == "Planning"
    assert args[1] == datetime(2026, 6, 9, 9, 0)
    assert args[3] == "agenda"
    assert args[4] == "Room 1"
    assert args[5] == 2
    assert args[6] == [10, 30]
    assert args[7] == "WEEKLY"


async def test_create_event_tool_default_times():
    # Omitting start/end is allowed; the client receives None and applies defaults.
    client = _fake_client()
    source = BuiltinMcpSource("calendar", build_calendar_server(client))
    await source.start()

    out = await source.call("create_event", {"summary": "Quick"})
    assert "new@zakhar" in out
    args = client.create_event.call_args.args
    assert args[0] == "Quick"
    assert args[1] is None and args[2] is None


async def test_create_event_tool_bad_dates():
    source = BuiltinMcpSource("calendar", build_calendar_server(_fake_client()))
    await source.start()

    out = await source.call("create_event", {"summary": "X", "start": "garbage"})
    assert "Не понял даты" in out


# --- gap #4: _event_dict tolerates a non-numeric PRIORITY ---------------------


def test_event_dict_non_numeric_priority_omits_key():
    # A PRIORITY that round-trips as a non-int (e.g. "HIGH") must be swallowed:
    # the int() in _event_dict raises ValueError, which is caught, so the "priority"
    # key is simply absent rather than crashing the mapping. icalendar validates
    # PRIORITY on .add(), so we parse raw text to get the non-numeric (vBroken) value
    # a malformed server response would carry.
    raw = (
        b"BEGIN:VCALENDAR\r\n"
        b"VERSION:2.0\r\n"
        b"PRODID:-//zakhar//test//EN\r\n"
        b"BEGIN:VEVENT\r\n"
        b"SUMMARY:Plan\r\n"
        b"UID:p@zakhar\r\n"
        b"DTSTART:20260610T090000\r\n"
        b"DTEND:20260610T100000\r\n"
        b"PRIORITY:HIGH\r\n"
        b"END:VEVENT\r\n"
        b"END:VCALENDAR\r\n"
    )
    comp = next(iter(ICalendar.from_ical(raw).walk("VEVENT")))
    # Sanity: the parsed PRIORITY really is non-numeric (int() would raise).
    assert str(comp.get("PRIORITY")) == "HIGH"

    result = CalendarClient._event_dict(comp)
    assert result is not None
    assert "priority" not in result
    # The rest of the mapping is intact.
    assert result["summary"] == "Plan"
    assert result["uid"] == "p@zakhar"


# --- gap #5: _to_yandex_utc naive-datetime branch -----------------------------


def test_create_event_yandex_naive_datetime_serializes_utc():
    # A tz-NAIVE start/end on a Yandex client must be interpreted as local time and
    # normalized to UTC, so DTSTART/DTEND serialize with a trailing Z. The expected
    # UTC value is derived from the SAME naive input at runtime (.astimezone()), so the
    # assertion holds regardless of the CI machine's timezone.
    naive_start = datetime(2026, 6, 10, 9, 0)
    naive_end = datetime(2026, 6, 10, 10, 0)
    ical, _ = _saved_ical(
        {"summary": "Y-naive", "start": naive_start, "end": naive_end},
        url="https://caldav.yandex.ru/",
    )
    dtstart = next(l for l in ical.splitlines() if l.startswith("DTSTART"))
    dtend = next(l for l in ical.splitlines() if l.startswith("DTEND"))
    assert dtstart.endswith("Z")
    assert dtend.endswith("Z")
    # Compute the expected compact-UTC string from the same naive inputs at runtime.
    expected_start = naive_start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    expected_end = naive_end.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assert dtstart == f"DTSTART:{expected_start}"
    assert dtend == f"DTEND:{expected_end}"


# --- gap #6: _parse_until malformed / empty (reached via _format_rrule) -------


def test_format_rrule_until_empty_raises():
    # "UNTIL=" yields an empty value: _parse_until must raise ValueError, not pass an
    # empty string down to icalendar (which would TypeError at serialization).
    try:
        _format_rrule("FREQ=DAILY;UNTIL=")
        assert False, "expected ValueError for empty UNTIL"
    except ValueError as e:
        assert "UNTIL" in str(e) or "empty" in str(e).lower()


def test_format_rrule_until_garbage_raises():
    # A non-parseable UNTIL must raise (strptime/fromisoformat fail) rather than
    # silently producing a bad value.
    try:
        _format_rrule("FREQ=DAILY;UNTIL=garbage")
        assert False, "expected an error for a garbage UNTIL"
    except (ValueError, TypeError):
        pass


# --- gap #7: _format_rrule malformed part (no '=') ----------------------------


def test_format_rrule_malformed_part_raises():
    # A part without '=' (e.g. "JUNK") is not a KEY=VALUE pair: _format_rrule must
    # raise ValueError naming the offending part.
    try:
        _format_rrule("FREQ=DAILY;JUNK")
        assert False, "expected ValueError for a malformed RRULE part"
    except ValueError as e:
        assert "JUNK" in str(e)
