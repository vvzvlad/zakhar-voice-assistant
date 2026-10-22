"""Persistent store + scheduler for what fires on a speaker at a set time.

A row is one of four kinds, all bound to the speaker they were set on and to an
absolute due time: a reminder (text to speak), a timer and an alarm (a phrase, then a
ring until silenced; `text` is their optional label) and a deferred command (`text` is a
request run through the assistant at its due time). An alarm may repeat on weekdays
(`days`, ISO 1 = Monday .. 7 = Sunday): fired, it moves to its next occurrence; every
other row is deleted once fired. The store is SQLite (mirroring src.runs_store
conventions: a shared Connection opened with check_same_thread=False, all DB methods
synchronous and serialized by a threading.Lock, WAL journaling).

The scheduler drives delivery on the event loop with a single-timer + wakeup-event
pattern: it sleeps until the earliest due time, fires everything then due, and wakes
early whenever a row is added, changed or cancelled. Rows that came due while the process was
down are taken once at start() without firing: a repeating alarm moves on, the rest
are dropped.
"""

import asyncio
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from datetime import time as dtime

from loguru import logger

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS reminders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT,
  text TEXT,
  due_ts REAL,
  days TEXT,
  device TEXT,
  created_ts REAL
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_reminders_due_ts ON reminders(due_ts);"

# Columns returned by the dict-returning methods, in select order.
_COLS = ["id", "kind", "text", "due_ts", "days", "device", "created_ts"]


def _row(r) -> dict:
    """A selected row as a dict, its repeat days ("1,2,3", stored as text) as a list."""
    row = dict(r)
    row["days"] = [int(d) for d in row["days"].split(",")] if row["days"] else None
    return row


def next_occurrence(hour: int, minute: int, days: list[int] | None, after: float) -> float:
    """The first local time hour:minute later than the timestamp `after`, on one of
    `days` (ISO weekdays 1..7, 1 = Monday; at least one) or on any day when `days` is None."""
    start = datetime.fromtimestamp(after).date()
    for i in range(8):  # today, then a whole week: every weekday comes up once
        at = datetime.combine(start + timedelta(days=i), dtime(hour, minute))
        if (days is None or at.isoweekday() in days) and at.timestamp() > after:
            return at.timestamp()


def _plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Pick the Russian plural form for `n` (one / few / many)."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def _format_ago(elapsed_s: float) -> str:
    """Render elapsed seconds as a Russian "<duration> назад" phrase.

    Granularity is one minute; sub-minute elapsed reads as "минуту назад".
    Rolls up to hours, then days.
    """
    minutes = int(round(elapsed_s / 60))
    if minutes < 1:
        minutes = 1
    if minutes < 60:
        if minutes == 1:
            return "минуту назад"
        return f"{minutes} {_plural_ru(minutes, 'минуту', 'минуты', 'минут')} назад"
    hours = int(round(minutes / 60))
    if hours < 24:
        if hours == 1:
            return "час назад"
        return f"{hours} {_plural_ru(hours, 'час', 'часа', 'часов')} назад"
    days = int(round(hours / 24))
    if days == 1:
        return "день назад"
    return f"{days} {_plural_ru(days, 'день', 'дня', 'дней')} назад"


def format_reminder_speech(text: str, created_ts, now: float) -> str:
    """Build the spoken reminder phrase.

    "<duration> назад вы просили напомнить вам <text>", with the reminder text
    lower-cased on its first letter so it reads naturally inside the sentence.
    Falls back to the raw `text` when the creation time is unknown (legacy rows).
    """
    if created_ts is None:
        return text
    body = text[:1].lower() + text[1:] if text else text
    ago = _format_ago(max(0.0, now - created_ts))
    return f"{ago} вы просили напомнить вам {body}"


def format_duration(seconds: int) -> str:
    """Render whole seconds as a Russian duration in the accusative ("на <duration>"):
    its hours, minutes and seconds that are not zero; a lone one of a unit is the bare
    word ("минуту", "час"), as in _format_ago."""
    parts = []
    for n, forms in ((seconds // 3600, ("час", "часа", "часов")),
                     (seconds % 3600 // 60, ("минуту", "минуты", "минут")),
                     (seconds % 60, ("секунду", "секунды", "секунд"))):
        if n:
            parts.append(f"{n} {_plural_ru(n, *forms)}")
    if len(parts) == 1 and parts[0].startswith("1 "):
        return parts[0][2:]
    return " ".join(parts)


def format_timer_speech(label, due_ts: float, created_ts: float) -> str:
    """The phrase a timer fires with: "Сработал таймер на <due - created>", then its label
    ("Сработал" so it is not heard as the confirmation of setting one)."""
    phrase = f"Сработал таймер на {format_duration(round(due_ts - created_ts))}"
    return f"{phrase}, {label}" if label else phrase


def format_alarm_speech(label) -> str:
    """The phrase an alarm fires with: "Будильник", then its label."""
    return f"Будильник, {label}" if label else "Будильник"


class RemindersStore:
    """Reminder, timer, alarm and deferred command rows over a single SQLite file."""

    def __init__(self, path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def insert(self, text, due_ts, device, kind="reminder", days=None) -> int:
        """Insert one row of `kind`; stamps created_ts. Returns the new row id.

        `device` and `text` may be None and are stored as-is; `days` (an alarm's repeat
        weekdays) None makes the row one-shot.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reminders (kind, text, due_ts, days, device, created_ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, text, due_ts, ",".join(map(str, days)) if days else None, device,
                 time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_pending(self) -> list[dict]:
        """All rows ordered by due_ts ascending, as dicts."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM reminders ORDER BY due_ts ASC"
            ).fetchall()
        return [_row(r) for r in rows]

    def next_due_ts(self) -> float | None:
        """MIN(due_ts) over all rows, or None when the table is empty."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(due_ts) AS m FROM reminders"
            ).fetchone()
        return row["m"] if row is not None else None

    def pop_due(self, now) -> list[dict]:
        """Atomically take all rows with due_ts <= now and return them as they were: a
        repeating alarm moves to its next occurrence after `now` (at the local time of
        its due time), every other row is deleted. The select and the writes happen
        under one lock so a concurrent insert can't slip a row in between.
        """
        with self._lock:
            rows = [_row(r) for r in self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM reminders WHERE due_ts <= ? "
                "ORDER BY due_ts ASC",
                (now,),
            ).fetchall()]
            for r in rows:
                if r["days"]:
                    at = datetime.fromtimestamp(r["due_ts"])
                    self._conn.execute(
                        "UPDATE reminders SET due_ts = ? WHERE id = ?",
                        (next_occurrence(at.hour, at.minute, r["days"], now), r["id"]),
                    )
                else:
                    self._conn.execute("DELETE FROM reminders WHERE id = ?", (r["id"],))
            self._conn.commit()
        return rows

    def update(self, reminder_id, **fields) -> dict | None:
        """Change one row's text, due_ts and/or days (the keywords given; days None makes
        an alarm one-shot) and return the row as stored, or None when no row has that id.
        When due_ts or days changes on a row left with repeat days, its due time becomes
        the next occurrence of its local hour:minute on those days after now, so a
        repeating alarm always sits on one of its days. Read and write happen under one
        lock so pop_due can't fire or move the row in between."""
        with self._lock:
            r = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
            if r is None:
                return None
            row = {**_row(r), **fields}
            if row["days"] and ("due_ts" in fields or "days" in fields):
                at = datetime.fromtimestamp(row["due_ts"])
                row["due_ts"] = next_occurrence(at.hour, at.minute, row["days"], time.time())
            self._conn.execute(
                "UPDATE reminders SET text = ?, due_ts = ?, days = ? WHERE id = ?",
                (row["text"], row["due_ts"],
                 ",".join(map(str, row["days"])) if row["days"] else None, reminder_id),
            )
            self._conn.commit()
        return row

    def delete(self, reminder_id) -> bool:
        """Delete one reminder by id; return whether a row existed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM reminders WHERE id = ?", (reminder_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def close(self):
        self._conn.close()


class ReminderScheduler:
    """Drives the delivery of due rows on the event loop.

    Single-timer + wakeup-event loop: sleep until the earliest due time, fire
    everything then due, wake early on add()/update()/cancel(). The `deliver` callback is an
    async callable (row: dict) -> None taking the whole row (DeviceManager.fire),
    assigned late by app.py once the device manager exists.
    """

    def __init__(self, store):
        self.store = store
        # Async callable (row) -> None; assigned late from app.py.
        self.deliver = None
        self._wakeup = asyncio.Event()
        self._task = None
        # Running deliveries: each fired row is its own task, so a ringing alarm (up to
        # minutes) or a command's model call never delays the rows due after it.
        self._deliveries: set[asyncio.Task] = set()
        self._stopped = False

    def add(self, text, due_ts, device, kind="reminder", days=None) -> int:
        """Persist a row and wake the loop to re-evaluate the earliest due time."""
        rid = self.store.insert(text=text, due_ts=due_ts, device=device, kind=kind, days=days)
        self._wakeup.set()
        return rid

    def cancel(self, reminder_id) -> bool:
        """Cancel a pending reminder; wake the loop. Return whether it existed."""
        ok = self.store.delete(reminder_id)
        self._wakeup.set()
        return ok

    def update(self, reminder_id, **fields) -> dict | None:
        """Change a pending row (RemindersStore.update); wake the loop to re-evaluate the
        earliest due time. Return the row as stored, or None when it does not exist."""
        row = self.store.update(reminder_id, **fields)
        self._wakeup.set()
        return row

    def pending(self) -> list[dict]:
        return self.store.list_pending()

    async def start(self) -> None:
        """Take the rows missed while down without firing them, then start the delivery
        loop."""
        self.store.pop_due(time.time())
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the delivery loop and the running deliveries, and await their tasks."""
        self._stopped = True
        self._wakeup.set()
        tasks = list(self._deliveries)
        if self._task is not None:
            tasks.append(self._task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(self, r: dict) -> None:
        """Hand one fired row to the deliver callback; a failure is logged, never raised."""
        if self.deliver is None:
            logger.warning("reminder scheduler has no deliver callback; dropping")
            return
        try:
            await self.deliver(r)
        except Exception as e:
            # The firing is lost (pop_due already deleted or moved the row), so log it
            # so the drop is visible to an operator.
            logger.error(
                f"{r['kind']} delivery failed (device={r['device']!r}, "
                f"text={r['text']!r}): {e}"
            )

    async def _loop(self) -> None:
        """Sleep until the earliest due time, fire due rows, repeat."""
        while not self._stopped:
            self._wakeup.clear()
            nxt = self.store.next_due_ts()
            if nxt is None:
                await self._wakeup.wait()  # nothing pending: sleep until add()/cancel()
                continue
            delay = max(0.0, nxt - time.time())
            if delay > 0:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                    continue  # woken by add()/update()/cancel(): re-evaluate earliest
                except asyncio.TimeoutError:
                    pass  # delay elapsed: fire everything now due
            for r in self.store.pop_due(time.time()):
                task = asyncio.create_task(self._deliver(r))
                self._deliveries.add(task)
                task.add_done_callback(self._deliveries.discard)
