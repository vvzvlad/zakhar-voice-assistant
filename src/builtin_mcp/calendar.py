"""Built-in calendar MCP server (CalDAV, in-process FastMCP).

Exposes calendar reading/writing as on-demand tools backed by a CalDAV account
(Nextcloud / iCloud / Fastmail / Google / Yandex via app-password). The CalDAV client
lib is SYNCHRONOUS (it talks plain HTTP), so the FastMCP tools offload every client
call to a worker thread via asyncio.to_thread — the event loop is never blocked. The
thin CalendarClient wrapper below stays pure-sync and has no asyncio in it.

Feature parity with a typical CalDAV-MCP server: list/read events in a range
(today / week / arbitrary / free-text search), read a single event by uid, create a
rich event (description, location, priority, recurrence, reminders/alarms),
delete by uid, list calendars.

All event (de)serialization goes through the `icalendar` library. icalendar handles
RFC 5545 TEXT escaping (`;`, `,`, `\\`, newlines) and property serialization itself —
we never hand-escape strings, which would double-escape them.
"""

import asyncio
import uuid
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import caldav
from caldav.lib.error import NotFoundError
from icalendar import Alarm
from icalendar import Calendar as ICalendar
from icalendar import Event as IEvent
from mcp.server.fastmcp import FastMCP

# Valid RFC 5545 RRULE frequency keywords. Used to validate the freq of a recurrence.
_VALID_FREQ = {"SECONDLY", "MINUTELY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY", "YEARLY"}

# All-day events have no start time, so a "minutes before start" reminder is anchored
# to this hour on the event day (e.g. 9 -> 09:00) instead of midnight.
ALL_DAY_REMINDER_HOUR = 9


def _parse_until(value):
    """Parse an RRULE UNTIL value into a tz-aware datetime (UTC) or a date.

    icalendar's vRecur serializes UNTIL via `to_ical()` and requires a real
    date/datetime object — a leftover string raises TypeError at serialization. We
    accept the common shapes a model or caller might pass:
      * iCalendar compact form: "YYYYMMDDTHHMMSSZ" / "YYYYMMDDTHHMMSS" (datetime),
        "YYYYMMDD" (date-only);
      * ISO form: "2026-12-31T00:00:00Z", "2026-12-31T00:00:00+00:00" (datetime),
        "2026-12-31" (date-only).
    A trailing `Z` is treated as UTC. Datetime results are returned tz-aware in UTC;
    date-only inputs return a `date`. An already-parsed date/datetime passes through.
    """
    # Already a date/datetime — leave it untouched (datetime is a subclass of date).
    if isinstance(value, (datetime, date)):
        return value

    text = str(value).strip()
    if not text:
        raise ValueError("UNTIL value is empty")

    # Normalize a trailing Z (UTC designator) into an explicit +00:00 offset so the
    # ISO parser produces a tz-aware datetime.
    has_z = text.endswith("Z")
    iso_text = (text[:-1] + "+00:00") if has_z else text

    # ISO form (contains '-' separators in the date part).
    if "-" in text:
        # No time component -> date-only (datetime.fromisoformat would otherwise
        # widen "2026-12-31" to a midnight datetime on 3.11+).
        if "T" not in text:
            return date.fromisoformat(text)
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    # iCalendar compact form.
    body = text[:-1] if has_z else text
    if "T" in body:
        parsed = datetime.strptime(body, "%Y%m%dT%H%M%S")
        return parsed.replace(tzinfo=timezone.utc)
    return datetime.strptime(body, "%Y%m%d").date()


def _is_yandex(url: str) -> bool:
    """True when the host of `url` contains "yandex" (case-insensitive).

    Yandex's CalDAV endpoint prefers DTSTART/DTEND in UTC (with a trailing `Z`); we use
    this flag to normalize event times before serialization.
    """
    host = urlparse(url or "").hostname or ""
    return "yandex" in host.lower()


def _format_rrule(recurrence) -> dict:
    """Normalize a recurrence spec into a vRecur-compatible dict.

    Accepts three input forms:
      * a freq keyword, e.g. "WEEKLY" (case-insensitive) -> {"FREQ": ["WEEKLY"]};
      * a full RRULE string, e.g.
        "FREQ=WEEKLY;INTERVAL=2;COUNT=10;BYDAY=MO,WE" -> the parsed dict;
      * a dict already in {KEY: value|list} form -> normalized in place.

    The result is suitable for `event.add("rrule", <dict>)`. Each value is wrapped in a
    list (icalendar's vRecur expects list values); comma-separated values (BYDAY etc.)
    are split. INTERVAL/COUNT become ints; an UNTIL value is parsed into a real
    date/datetime (icalendar's to_ical() rejects a leftover string). The FREQ key is
    required and validated — a missing/empty or unknown freq raises ValueError.
    """
    if recurrence is None:
        raise ValueError("recurrence is empty")

    parsed: dict = {}
    if isinstance(recurrence, dict):
        # Already structured: just upper-case keys and coerce values to lists.
        for key, value in recurrence.items():
            parsed[str(key).strip().upper()] = value
    else:
        text = str(recurrence).strip()
        if not text:
            raise ValueError("recurrence is empty")
        if "=" not in text:
            # Bare freq keyword form, e.g. "weekly".
            parsed["FREQ"] = text.upper()
        else:
            # Full RRULE string: split on ';' then 'KEY=VALUE'.
            for part in text.split(";"):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError(f"bad RRULE part: {part!r}")
                key, value = part.split("=", 1)
                parsed[key.strip().upper()] = value.strip()

    result: dict = {}
    for key, value in parsed.items():
        if key == "UNTIL":
            # UNTIL must become a real date/datetime; icalendar's to_ical() raises
            # TypeError on a leftover string. Parse strings; leave date/datetime as-is.
            if isinstance(value, (list, tuple)):
                items = [_parse_until(v) for v in value]
            else:
                items = [_parse_until(value)]
            result[key] = items
            continue
        if isinstance(value, (list, tuple)):
            items = [str(v).strip() for v in value if str(v).strip()]
        else:
            # Split comma-separated scalars (e.g. "MO,WE") into a list.
            items = [v.strip() for v in str(value).split(",") if v.strip()]
        # Numeric keys (INTERVAL/COUNT) become ints so vRecur serializes them cleanly.
        if key in {"INTERVAL", "COUNT"}:
            items = [int(v) for v in items]
        result[key] = items

    # FREQ is required for a usable RRULE. Guard against both a missing key and an
    # empty value (e.g. "FREQ=;INTERVAL=2") so result["FREQ"][0] never IndexErrors.
    if not result.get("FREQ"):
        raise ValueError("recurrence is missing FREQ")

    freq_value = str(result["FREQ"][0]).upper()
    if freq_value not in _VALID_FREQ:
        raise ValueError(f"unknown recurrence frequency: {freq_value!r}")
    result["FREQ"] = [freq_value]
    return result


def _nonpositive_duration(start: datetime, end: datetime) -> bool:
    """True when `end` is at or before `start` (a zero/negative-length event).

    Tolerant of a tz-awareness mismatch: if a direct comparison raises (one side
    tz-aware, the other naive), fall back to comparing the naive wall-clock values.
    """
    try:
        return end <= start
    except TypeError:
        return end.replace(tzinfo=None) <= start.replace(tzinfo=None)


class CalendarClient:
    """Thin synchronous wrapper over the `caldav` library.

    Connects lazily on first use and caches the selected calendar. All methods are
    blocking (the underlying lib uses synchronous HTTP); callers must offload them to a
    worker thread when running on an event loop.
    """

    def __init__(self, url: str, username: str, password: str, calendar_name: str = ""):
        self._url = url
        self._username = username
        self._password = password
        self._calendar_name = calendar_name
        # Yandex prefers UTC times; remember it so create_event normalizes accordingly.
        self.is_yandex = _is_yandex(url)
        # Lazily resolved on the first call so construction never touches the network.
        self._calendar = None

    def _principal(self):
        """Open a DAVClient and return the account principal."""
        client = caldav.DAVClient(
            url=self._url, username=self._username, password=self._password
        )
        return client.principal()

    def _get_calendar(self):
        """Resolve and cache the calendar to operate on.

        Picks the calendar whose name matches `calendar_name` when set, otherwise the
        first calendar on the principal. Raises if the account exposes no calendars.
        """
        if self._calendar is not None:
            return self._calendar
        calendars = self._principal().calendars()
        if not calendars:
            raise RuntimeError("CalDAV account has no calendars")
        chosen = None
        if self._calendar_name:
            for cal in calendars:
                if cal.get_display_name() == self._calendar_name:
                    chosen = cal
                    break
            if chosen is None:
                raise RuntimeError(f"calendar {self._calendar_name!r} not found")
        else:
            chosen = calendars[0]
        self._calendar = chosen
        return self._calendar

    def list_calendars(self) -> list[dict]:
        """Return every calendar available on the principal."""
        return [
            {"name": cal.get_display_name(), "url": str(cal.url)}
            for cal in self._principal().calendars()
        ]

    @staticmethod
    def _iso(value) -> str:
        """Render a date or datetime to ISO-8601 text; pass through anything else."""
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _component_of(ev):
        """Pull the icalendar component out of a caldav object, or None."""
        comp = getattr(ev, "icalendar_component", None)
        if comp is None:
            getter = getattr(ev, "get_icalendar_component", None)
            if getter is not None:
                comp = getter()
        return comp

    @classmethod
    def _event_dict(cls, comp) -> dict | None:
        """Map an icalendar VEVENT component to the plain-dict event shape.

        Returns None when the component lacks the minimum required fields (summary and
        start), so callers can skip it defensively. The dict always carries
        uid/summary/start/end/location and includes description/priority when present.
        """
        summary = comp.get("SUMMARY")
        dtstart = comp.get("DTSTART")
        if summary is None or dtstart is None:
            return None
        dtend = comp.get("DTEND")
        result: dict = {
            "uid": str(comp.get("UID", "")),
            "summary": str(summary),
            "start": cls._iso(dtstart.dt),
            "end": cls._iso(dtend.dt) if dtend is not None else "",
            "location": str(comp.get("LOCATION", "")),
        }
        description = comp.get("DESCRIPTION")
        if description is not None:
            result["description"] = str(description)
        priority = comp.get("PRIORITY")
        if priority is not None:
            try:
                result["priority"] = int(priority)
            except (TypeError, ValueError):
                pass
        return result

    def get_events(self, start: datetime, end: datetime) -> list[dict]:
        """Return events overlapping [start, end) as plain dicts.

        Recurrences are expanded by the server/lib so a weekly event shows up once per
        occurrence in range. Components missing a summary or start are skipped
        defensively rather than raising.
        """
        results = self._get_calendar().search(
            start=start, end=end, event=True, expand=True
        )
        events: list[dict] = []
        for ev in results:
            comp = self._component_of(ev)
            if comp is None:
                continue
            mapped = self._event_dict(comp)
            if mapped is not None:
                events.append(mapped)
        return events

    def get_today_events(self) -> list[dict]:
        """Events from today 00:00 up to the next midnight."""
        # Timezone-aware local time so range boundaries compare correctly against
        # events that carry a timezone.
        now = datetime.now().astimezone()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return self.get_events(start, end)

    def get_week_events(self) -> list[dict]:
        """Events from now to seven days ahead."""
        # Timezone-aware local time so range boundaries compare correctly against
        # events that carry a timezone.
        now = datetime.now().astimezone()
        return self.get_events(now, now + timedelta(days=7))

    def search_events(
        self, query: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[dict]:
        """Free-text search over events in a range (default now-30d .. now+90d).

        CalDAV has no portable full-text query, so we fetch the range and filter in
        Python: an event matches when `query` (case-insensitive) is a substring of its
        summary, description or location.
        """
        now = datetime.now().astimezone()
        if start is None:
            start = now - timedelta(days=30)
        if end is None:
            end = now + timedelta(days=90)
        needle = (query or "").strip().lower()
        events = self.get_events(start, end)
        if not needle:
            return events
        matched: list[dict] = []
        for ev in events:
            haystacks = [
                ev.get("summary", ""),
                ev.get("description", ""),
                ev.get("location", ""),
            ]
            if any(needle in str(h).lower() for h in haystacks):
                matched.append(ev)
        return matched

    def _to_yandex_utc(self, value):
        """Convert a datetime to UTC for Yandex (aware->astimezone; naive->assume local).

        Plain dates (all-day events) are returned unchanged — they have no time to
        normalize. Non-datetime values pass through untouched.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.astimezone()  # interpret naive as local time
            return value.astimezone(timezone.utc)
        return value

    def create_event(
        self,
        summary: str,
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        description: str = "",
        location: str = "",
        priority=None,
        reminders=None,
        recurrence=None,
    ) -> dict:
        """Create a single VEVENT on the calendar and return its core fields.

        A bare `date` start (no time component) produces an all-day event
        (DTSTART;VALUE=DATE / DTEND;VALUE=DATE); a `datetime` start produces a timed
        event. Optional fields (description, location, priority, reminders, recurrence)
        are added only when provided, so existing callers that pass just
        summary/start/end keep working unchanged.
        """
        # Default start when omitted: next full hour (a timed event).
        if start is None:
            now = datetime.now().astimezone()
            start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        # A bare `date` start (no time component) means an all-day event: icalendar
        # serializes a date as DTSTART;VALUE=DATE. datetime is a subclass of date, so
        # the `not isinstance(..., datetime)` guard is required to tell them apart.
        all_day = isinstance(start, date) and not isinstance(start, datetime)

        if all_day:
            # RFC 5545 DTEND is exclusive for all-day events; a single-day event ends
            # on the next day. Keep DTSTART/DTEND the same value type (date) and ensure
            # DTEND > DTSTART (a zero-length all-day event is rejected by servers).
            if isinstance(end, datetime):
                end = end.date()
            if end is None or end <= start:
                end = start + timedelta(days=1)
        else:
            # A bare `date` end on a timed event would mix value types (DTSTART with a
            # time + DTEND;VALUE=DATE), which RFC 5545 forbids, and the non-positive
            # check below cannot compare a date against a datetime. Promote it to
            # midnight of that day, sharing the start's tzinfo, so both ends are timed.
            if isinstance(end, date) and not isinstance(end, datetime):
                end = datetime(end.year, end.month, end.day, tzinfo=start.tzinfo)
            if end is None:
                end = start + timedelta(hours=1)
            elif _nonpositive_duration(start, end):
                # A provided end at or before start — e.g. a date-only end widened to
                # midnight on (or before) the start's day — would yield a zero/negative
                # length event; fall back to the default 1-hour duration.
                end = start + timedelta(hours=1)
            # Yandex prefers UTC so DTSTART/DTEND serialize with a trailing `Z`. All-day
            # events carry no time, so this only applies to timed events.
            if self.is_yandex:
                start = self._to_yandex_utc(start)
                end = self._to_yandex_utc(end)

        uid = uuid.uuid4().hex + "@zakhar"
        cal = ICalendar()
        cal.add("prodid", "-//zakhar//calendar//EN")
        cal.add("version", "2.0")
        event = IEvent()
        event.add("uid", uid)
        event.add("summary", summary)
        event.add("dtstart", start)
        event.add("dtend", end)
        # RFC 5545 requires DTSTAMP on every VEVENT; strict servers (iCloud/Google)
        # reject events that omit it.
        event.add("dtstamp", datetime.now(timezone.utc))
        if description:
            event.add("description", description)
        if location:
            event.add("location", location)

        if priority is not None:
            # PRIORITY is 0..9 per RFC 5545; clamp to that range.
            event.add("priority", max(0, min(9, int(priority))))

        if recurrence:
            event.add("rrule", _format_rrule(recurrence))

        for alarm in self._build_alarms(reminders, all_day):
            event.add_component(alarm)

        cal.add_component(event)
        ical_text = cal.to_ical().decode()
        self._get_calendar().add_event(ical_text)
        return {
            "uid": uid,
            "summary": summary,
            "start": self._iso(start),
            "end": self._iso(end),
        }

    @staticmethod
    def _build_alarms(reminders, all_day: bool = False) -> list[Alarm]:
        """Build VALARM components from a list of reminder specs.

        Each item is either an int (minutes before start, ACTION=DISPLAY) or a dict
        {minutes, action} where action is DISPLAY or AUDIO. The TRIGGER is relative to
        DTSTART; DISPLAY alarms carry a DESCRIPTION.

        For timed events the trigger is N minutes before start (a negative timedelta).
        For all-day events DTSTART is midnight, so "N minutes before" is anchored to
        ALL_DAY_REMINDER_HOUR (e.g. 09:00) on the event day instead: the trigger is
        ALL_DAY_REMINDER_HOUR hours after midnight minus N minutes (so 0 minutes fires
        at 09:00, 60 at 08:00, 1440 at 09:00 the previous day).
        """
        alarms: list[Alarm] = []
        for item in reminders or []:
            if isinstance(item, dict):
                minutes = int(item.get("minutes", 0))
                action = str(item.get("action") or "DISPLAY").upper()
            else:
                minutes = int(item)
                action = "DISPLAY"
            if action not in {"DISPLAY", "AUDIO"}:
                action = "DISPLAY"
            if all_day:
                trigger = timedelta(hours=ALL_DAY_REMINDER_HOUR, minutes=-minutes)
            else:
                trigger = timedelta(minutes=-minutes)
            alarm = Alarm()
            alarm.add("action", action)
            alarm.add("trigger", trigger)
            if action == "DISPLAY":
                # DISPLAY alarms require a DESCRIPTION per RFC 5545.
                alarm.add("description", "Reminder")
            alarms.append(alarm)
        return alarms

    def get_event_by_uid(self, uid: str) -> dict:
        """Fetch a single event by uid and map it to the full event dict.

        Returns {"found": False, "uid": uid} when the event does not exist.
        """
        try:
            ev = self._get_calendar().get_event_by_uid(uid)
        except NotFoundError:
            return {"found": False, "uid": uid}
        comp = self._component_of(ev)
        if comp is None:
            return {"found": False, "uid": uid}
        mapped = self._event_dict(comp)
        if mapped is None:
            return {"found": False, "uid": uid}
        return mapped

    def delete_event(self, uid: str) -> dict:
        """Delete an event by its uid; report whether it existed."""
        try:
            event = self._get_calendar().get_event_by_uid(uid)
            event.delete()
        except NotFoundError:
            return {"deleted": False, "uid": uid, "error": "not found"}
        return {"deleted": True, "uid": uid}


def _parse_dt(value: str) -> datetime:
    """Parse a YYYY-MM-DD date or an ISO datetime into a datetime.

    `datetime.fromisoformat` handles both full datetimes and bare dates on 3.11, but we
    keep an explicit date fallback for robustness across inputs.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d")


def _parse_range_end(value: str) -> datetime:
    """Parse a range END boundary; a bare date covers that whole day.

    CalDAV time-range search uses a half-open [start, end) window, so a date-only
    END like "2026-06-10" must advance to the next midnight to include events on
    that day (otherwise start == end yields an empty window and finds nothing). An
    END that carries an explicit time is used verbatim.
    """
    parsed = _parse_dt(value)
    # Date-only input (no time component) -> exclusive upper bound = next midnight.
    if "T" not in value and ":" not in value:
        return parsed + timedelta(days=1)
    return parsed


def _parse_event_dt(value: str) -> date | datetime:
    """Parse an event boundary, preserving a bare calendar date as a `date`.

    A date-only input ("YYYY-MM-DD", no time component) returns a `date` so
    create_event can emit an all-day VEVENT (DTSTART;VALUE=DATE). Anything carrying a
    time returns a `datetime`. Contrast with `_parse_dt`, which always widens to a
    datetime — used by the range/search tools where a time boundary is required.
    """
    text = value.strip()
    # No time component -> treat as an all-day date. (datetime.fromisoformat would
    # otherwise widen "2026-06-10" to a midnight datetime.)
    if "T" not in text and ":" not in text:
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    return _parse_dt(text)


def _format_events(events: list[dict], empty_text: str) -> str:
    """Render an event list as readable Russian text, one event per line."""
    if not events:
        return empty_text
    lines = []
    for ev in events:
        span = ev["start"]
        if ev.get("end"):
            span = f"{ev['start']}..{ev['end']}"
        line = f"{ev['summary']} — {span}"
        if ev.get("location"):
            line += f" ({ev['location']})"
        lines.append(line)
    return "\n".join(lines)


def _format_event_details(ev: dict) -> str:
    """Render a single event dict as a multi-line Russian description."""
    lines = [f"Событие: {ev.get('summary', '')}"]
    span = ev.get("start", "")
    if ev.get("end"):
        span = f"{ev.get('start', '')}..{ev['end']}"
    if span:
        lines.append(f"Время: {span}")
    if ev.get("location"):
        lines.append(f"Место: {ev['location']}")
    if ev.get("description"):
        lines.append(f"Описание: {ev['description']}")
    if ev.get("priority") is not None:
        lines.append(f"Приоритет: {ev['priority']}")
    if ev.get("uid"):
        lines.append(f"uid: {ev['uid']}")
    return "\n".join(lines)


def build_calendar_server(client: CalendarClient) -> FastMCP:
    """Build a FastMCP server exposing the calendar tools.

    Every tool offloads the blocking caldav calls via asyncio.to_thread and returns a
    short human-readable string. Errors are caught and returned as text (never raised),
    mirroring the OpenWeatherMap tool / ToolHub contract so a failure reaches the model as a
    message instead of crashing the run.
    """
    mcp = FastMCP("calendar")

    @mcp.tool(
        name="get_today_events",
        description="События календаря на сегодня (с названием и временем).",
    )
    async def get_today_events() -> str:
        try:
            events = await asyncio.to_thread(client.get_today_events)
        except Exception as e:
            return f"Не удалось получить события: {e}"
        return _format_events(events, "На сегодня событий нет.")

    @mcp.tool(
        name="get_week_events",
        description="События календаря на ближайшие 7 дней.",
    )
    async def get_week_events() -> str:
        try:
            events = await asyncio.to_thread(client.get_week_events)
        except Exception as e:
            return f"Не удалось получить события: {e}"
        return _format_events(events, "На ближайшую неделю событий нет.")

    @mcp.tool(
        name="get_events",
        description=(
            "События календаря в заданном диапазоне. Аргументы start и end — "
            "дата (ГГГГ-ММ-ДД) или дата-время в формате ISO."
        ),
    )
    async def get_events(start: str, end: str) -> str:
        try:
            start_dt = _parse_dt(start)
            end_dt = _parse_range_end(end)
        except ValueError:
            return "Не понял даты: используйте формат ГГГГ-ММ-ДД или ISO дату-время."
        try:
            events = await asyncio.to_thread(client.get_events, start_dt, end_dt)
        except Exception as e:
            return f"Не удалось получить события: {e}"
        return _format_events(events, "В этом диапазоне событий нет.")

    @mcp.tool(
        name="search_events",
        description=(
            "Поиск событий по тексту (в названии, описании или месте). "
            "query — искомая строка; start и end необязательны (дата ГГГГ-ММ-ДД или "
            "ISO дата-время), по умолчанию ищет в диапазоне от 30 дней назад до "
            "90 дней вперёд."
        ),
    )
    async def search_events(
        query: str, start: str | None = None, end: str | None = None
    ) -> str:
        start_dt = None
        end_dt = None
        try:
            if start:
                start_dt = _parse_dt(start)
            if end:
                end_dt = _parse_range_end(end)
        except ValueError:
            return "Не понял даты: используйте формат ГГГГ-ММ-ДД или ISO дату-время."
        try:
            events = await asyncio.to_thread(
                client.search_events, query, start_dt, end_dt
            )
        except Exception as e:
            return f"Не удалось выполнить поиск: {e}"
        return _format_events(events, f"По запросу «{query}» ничего не найдено.")

    @mcp.tool(
        name="get_event_by_uid",
        description="Получить полную информацию о событии по его uid.",
    )
    async def get_event_by_uid(uid: str) -> str:
        try:
            event = await asyncio.to_thread(client.get_event_by_uid, uid)
        except Exception as e:
            return f"Не удалось получить событие: {e}"
        if not event or event.get("found") is False:
            return f"Событие {uid} не найдено."
        return _format_event_details(event)

    @mcp.tool(
        name="create_event",
        description=(
            "Создать событие в календаре. summary — название; start и end — "
            "дата (ГГГГ-ММ-ДД) или ISO дата-время (если не указаны, берётся ближайший "
            "час и длительность 1 час). Если start задан только датой (ГГГГ-ММ-ДД, без "
            "времени), событие создаётся на весь день. Необязательно: description, "
            "location, priority (0-9), recurrence (правило повторения, напр. WEEKLY или "
            "FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,WE), reminders (список напоминаний "
            "в минутах до начала)."
        ),
    )
    async def create_event(
        summary: str,
        start: str | None = None,
        end: str | None = None,
        description: str = "",
        location: str = "",
        priority: int | None = None,
        recurrence: str | None = None,
        reminders: list[int] | None = None,
    ) -> str:
        start_dt = None
        end_dt = None
        try:
            if start:
                # Date-only start -> a `date`, so the client emits an all-day event.
                start_dt = _parse_event_dt(start)
            if end:
                # End stays a datetime: the client coerces it to a date inside the
                # all-day branch, so a timed start never pairs with a VALUE=DATE end
                # (mismatched value types are RFC 5545-invalid).
                end_dt = _parse_dt(end)
        except ValueError:
            return "Не понял даты: используйте формат ГГГГ-ММ-ДД или ISO дату-время."
        try:
            created = await asyncio.to_thread(
                client.create_event,
                summary,
                start_dt,
                end_dt,
                description,
                location,
                priority,
                reminders,
                recurrence,
            )
        except Exception as e:
            return f"Не удалось создать событие: {e}"
        return f"Событие «{created['summary']}» создано (uid {created['uid']})."

    @mcp.tool(
        name="delete_event",
        description="Удалить событие из календаря по его uid.",
    )
    async def delete_event(uid: str) -> str:
        try:
            result = await asyncio.to_thread(client.delete_event, uid)
        except Exception as e:
            return f"Не удалось удалить событие: {e}"
        if result.get("deleted"):
            return f"Событие {uid} удалено."
        return f"Событие {uid} не найдено."

    @mcp.tool(
        name="list_calendars",
        description="Список доступных календарей в аккаунте.",
    )
    async def list_calendars() -> str:
        try:
            calendars = await asyncio.to_thread(client.list_calendars)
        except Exception as e:
            return f"Не удалось получить список календарей: {e}"
        if not calendars:
            return "Календари не найдены."
        return "\n".join(c["name"] for c in calendars)

    return mcp
