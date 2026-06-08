"""Persistent store + scheduler for one-shot voice reminders.

A reminder is a piece of text to speak on a specific speaker at a specific time.
Reminders are one-shot: once fired they are deleted. The store is SQLite (mirroring
src.runs_store conventions: a shared Connection opened with check_same_thread=False,
all DB methods synchronous and serialized by a threading.Lock, WAL journaling).

The scheduler drives delivery on the event loop with a single-timer + wakeup-event
pattern: it sleeps until the earliest due time, fires everything then due, and wakes
early whenever a reminder is added or cancelled. Missed reminders (due while the
process was down) are dropped once at start().
"""

import asyncio
import sqlite3
import threading
import time

from loguru import logger

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS reminders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT,
  due_ts REAL,
  device TEXT,
  created_ts REAL
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_reminders_due_ts ON reminders(due_ts);"

# Columns returned by the dict-returning methods, in select order.
_COLS = ["id", "text", "due_ts", "device", "created_ts"]


class RemindersStore:
    """One-shot reminder rows over a single SQLite file."""

    def __init__(self, path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def insert(self, text, due_ts, device) -> int:
        """Insert one reminder; stamps created_ts. Returns the new row id.

        `device` may be None and is stored as-is.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reminders (text, due_ts, device, created_ts) "
                "VALUES (?, ?, ?, ?)",
                (text, due_ts, device, time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_pending(self) -> list[dict]:
        """All reminders ordered by due_ts ascending, as dicts."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM reminders ORDER BY due_ts ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def next_due_ts(self) -> float | None:
        """MIN(due_ts) over all rows, or None when the table is empty."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(due_ts) AS m FROM reminders"
            ).fetchone()
        return row["m"] if row is not None else None

    def pop_due(self, now) -> list[dict]:
        """Atomically take all reminders with due_ts <= now: select, delete, return.

        One-shot semantics: a fired reminder is gone. The select and delete happen
        under one lock so a concurrent insert can't slip a row in between.
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM reminders WHERE due_ts <= ? "
                "ORDER BY due_ts ASC",
                (now,),
            ).fetchall()
            self._conn.execute("DELETE FROM reminders WHERE due_ts <= ?", (now,))
            self._conn.commit()
        return [dict(r) for r in rows]

    def drop_overdue(self, now) -> int:
        """Delete all reminders with due_ts <= now; return the deleted count.

        Used once at boot to drop reminders that came due while the process was down.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM reminders WHERE due_ts <= ?", (now,)
            )
            self._conn.commit()
            return cur.rowcount

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
    """Drives one-shot reminder delivery on the event loop.

    Single-timer + wakeup-event loop: sleep until the earliest due time, fire
    everything then due, wake early on add()/cancel(). The `deliver` callback is an
    async callable (device: str|None, text: str) -> None, assigned late by app.py
    once the device manager exists.
    """

    def __init__(self, store):
        self.store = store
        # Async callable (device, text) -> None; assigned late from app.py.
        self.deliver = None
        self._wakeup = asyncio.Event()
        self._task = None
        self._stopped = False

    def add(self, text, due_ts, device) -> int:
        """Persist a reminder and wake the loop to re-evaluate the earliest due time."""
        rid = self.store.insert(text=text, due_ts=due_ts, device=device)
        self._wakeup.set()
        return rid

    def cancel(self, reminder_id) -> bool:
        """Cancel a pending reminder; wake the loop. Return whether it existed."""
        ok = self.store.delete(reminder_id)
        self._wakeup.set()
        return ok

    def pending(self) -> list[dict]:
        return self.store.list_pending()

    async def start(self) -> None:
        """Drop missed reminders, then start the delivery loop."""
        self.store.drop_overdue(time.time())
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the delivery loop and await its task."""
        self._stopped = True
        self._wakeup.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        """Sleep until the earliest due time, fire due reminders, repeat."""
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
                    continue  # woken by add()/cancel(): re-evaluate earliest
                except asyncio.TimeoutError:
                    pass  # delay elapsed: fire everything now due
            for r in self.store.pop_due(time.time()):
                if self.deliver is None:
                    logger.warning(
                        "reminder scheduler has no deliver callback; dropping"
                    )
                    continue
                try:
                    await self.deliver(r["device"], r["text"])
                except Exception as e:
                    logger.error(f"reminder delivery failed: {e}")
