"""Built-in calendar MCP server (CalDAV, in-process FastMCP).

Exposes calendar reading/writing as on-demand tools backed by a CalDAV account
(Nextcloud / iCloud / Fastmail / Google via app-password). The CalDAV client lib is
SYNCHRONOUS (it talks plain HTTP), so the FastMCP tools offload every client call to a
worker thread via asyncio.to_thread — the event loop is never blocked. The thin
CalendarClient wrapper below stays pure-sync and has no asyncio in it.

Scope for v1 is intentionally small: list/read events in a range (today / week /
arbitrary), create a simple event, delete by uid, list calendars. Recurrence editing,
attendees, priority and free-text search are out of scope for now.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import caldav
from caldav.lib.error import NotFoundError
from icalendar import Calendar as ICalendar
from icalendar import Event as IEvent
from mcp.server.fastmcp import FastMCP


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

    def get_events(self, start: datetime, end: datetime) -> list[dict]:
        """Return events overlapping [start, end) as plain dicts.

        Recurrences are expanded by the server/lib so a weekly event shows up once per
        occurrence in range. Components missing a summary or both endpoints are skipped
        defensively rather than raising.
        """
        results = self._get_calendar().search(
            start=start, end=end, event=True, expand=True
        )
        events: list[dict] = []
        for ev in results:
            comp = getattr(ev, "icalendar_component", None)
            if comp is None:
                comp = ev.get_icalendar_component()
            if comp is None:
                continue
            summary = comp.get("SUMMARY")
            dtstart = comp.get("DTSTART")
            dtend = comp.get("DTEND")
            if summary is None or dtstart is None:
                # Required fields missing — not something we can present.
                continue
            events.append(
                {
                    "uid": str(comp.get("UID", "")),
                    "summary": str(summary),
                    "start": self._iso(dtstart.dt),
                    "end": self._iso(dtend.dt) if dtend is not None else "",
                    "location": str(comp.get("LOCATION", "")),
                }
            )
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

    def create_event(
        self,
        summary: str,
        start: datetime,
        end: datetime,
        description: str = "",
        location: str = "",
    ) -> dict:
        """Create a single VEVENT on the calendar and return its core fields."""
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
        cal.add_component(event)
        ical_text = cal.to_ical().decode()
        self._get_calendar().add_event(ical_text)
        return {
            "uid": uid,
            "summary": summary,
            "start": self._iso(start),
            "end": self._iso(end),
        }

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


def build_calendar_server(client: CalendarClient) -> FastMCP:
    """Build a FastMCP server exposing the calendar tools.

    Every tool offloads the blocking caldav calls via asyncio.to_thread and returns a
    short human-readable string. Errors are caught and returned as text (never raised),
    mirroring the weather tool / ToolHub contract so a failure reaches the model as a
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
            end_dt = _parse_dt(end)
        except ValueError:
            return "Не понял даты: используйте формат ГГГГ-ММ-ДД или ISO дату-время."
        try:
            events = await asyncio.to_thread(client.get_events, start_dt, end_dt)
        except Exception as e:
            return f"Не удалось получить события: {e}"
        return _format_events(events, "В этом диапазоне событий нет.")

    @mcp.tool(
        name="create_event",
        description=(
            "Создать событие в календаре. summary — название, start и end — "
            "дата (ГГГГ-ММ-ДД) или ISO дата-время; description и location необязательны."
        ),
    )
    async def create_event(
        summary: str, start: str, end: str, description: str = "", location: str = ""
    ) -> str:
        try:
            start_dt = _parse_dt(start)
            end_dt = _parse_dt(end)
        except ValueError:
            return "Не понял даты: используйте формат ГГГГ-ММ-ДД или ISO дату-время."
        try:
            created = await asyncio.to_thread(
                client.create_event, summary, start_dt, end_dt, description, location
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
