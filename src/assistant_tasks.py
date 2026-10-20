"""Jobs handed off to the ZeroClaw assistant and delivery of its replies.

A job goes to ZeroClaw's webhook channel as a signed POST; ZeroClaw only queues it
and runs the agent later, then posts every reply of that agent back to the panel
(POST /api/assistant-tasks/reply). The `thread_id` "zakhar-<task id>" ties a reply
to its task, and the task row remembers the speaker the job came from, so the reply
is spoken there.

The ledger is SQLite (mirroring src.reminders conventions: a shared Connection opened
with check_same_thread=False, all DB methods synchronous and serialized by a
threading.Lock, WAL journaling). It answers "how are the tasks going" and keeps a
reply that could not be spoken (delivered=0).
"""

import asyncio
import hashlib
import hmac
import json
import re
import sqlite3
import threading
import time

import httpx
from loguru import logger

from src.run_events import record_run
from src.runs_store import new_record

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT,
  device TEXT,
  created_ts REAL,
  answer TEXT,
  answered_ts REAL,
  delivered INTEGER DEFAULT 0
);
"""

# Columns returned by the dict-returning methods, in select order.
_COLS = ["id", "text", "device", "created_ts", "answer", "answered_ts", "delivered"]

# The thread id a task is sent with, and that ZeroClaw echoes on its replies.
_THREAD_RE = re.compile(r"zakhar-(\d+)")


class AssistantTasksStore:
    """Handed-off task rows over a single SQLite file."""

    def __init__(self, path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def insert(self, text, device) -> int:
        """Insert one task; stamps created_ts. Returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (text, device, created_ts) VALUES (?, ?, ?)",
                (text, device, time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def delete(self, task_id) -> bool:
        """Delete one task by id; return whether a row existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, task_id) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def add_answer(self, task_id, content) -> float:
        """Append one reply to the task's answer (blank line between replies) and stamp
        answered_ts, which is returned. delivered drops to 0: the appended reply has not
        been spoken yet."""
        answered_ts = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT answer FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            answer = content if not row["answer"] else f"{row['answer']}\n\n{content}"
            self._conn.execute(
                "UPDATE tasks SET answer = ?, answered_ts = ?, delivered = 0 WHERE id = ?",
                (answer, answered_ts, task_id),
            )
            self._conn.commit()
        return answered_ts

    def mark_delivered(self, task_id, answered_ts) -> None:
        """Mark the task's answer spoken, unless a newer reply (another answered_ts) has
        been appended since: that one is not spoken yet."""
        with self._lock:
            self._conn.execute("UPDATE tasks SET delivered = 1 WHERE id = ? AND answered_ts = ?",
                               (task_id, answered_ts))
            self._conn.commit()

    def recent(self, limit) -> list[dict]:
        """The newest `limit` tasks, newest first, as dicts."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self._conn.close()


def _task_id(payload) -> int | None:
    """The task id from a reply's thread_id, else its recipient; None when neither
    is "zakhar-<int>"."""
    if not isinstance(payload, dict):
        return None
    for key in ("thread_id", "recipient"):
        value = payload.get(key)
        if isinstance(value, str):
            m = _THREAD_RE.fullmatch(value)
            if m:
                return int(m.group(1))
    return None


class AssistantTasks:
    """Hands jobs to ZeroClaw and speaks its replies on the speaker each job came from.

    Config (`rt.core.assistant_tasks`) and the speakers (`rt.manager`) are read at call
    time, so a config change or a reconnect needs nothing here. `rt` is assigned late by
    app.py: the tool source built off this service exists before the Runtime does."""

    def __init__(self, store):
        self.store = store
        self.rt = None  # Runtime; assigned late from app.py
        # Replies being spoken: referenced until done so the event loop keeps them.
        self._speaking: set[asyncio.Task] = set()
        # One reply at a time, so two replies of one task are spoken in order.
        self._speak_lock = asyncio.Lock()

    async def delegate(self, text, device) -> int:
        """Record the task and post it to ZeroClaw; return the task id. On a network
        error or a non-2xx answer the row is deleted and the error raised."""
        cfg = self.rt.core.assistant_tasks
        task_id = self.store.insert(text, device)
        # Sign exactly the bytes that are sent.
        body = json.dumps({
            "sender": device or "zakhar",
            "content": f"{text}\n\n{cfg.task_suffix}",
            "thread_id": f"zakhar-{task_id}",
        }, ensure_ascii=False).encode("utf-8")
        signature = hmac.new(cfg.secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(cfg.url, content=body, headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": f"sha256={signature}",
                })
            resp.raise_for_status()
        except Exception:
            self.store.delete(task_id)
            raise
        logger.info(f"assistant task №{task_id} handed off (device={device!r}): {text!r}")
        return task_id

    async def on_reply(self, payload) -> tuple[int, dict]:
        """Take one ZeroClaw reply: store it on its task and start speaking it on the
        task's speaker. Returns (HTTP status, JSON body) at once, without waiting for the
        speech: ZeroClaw re-sends a reply whose request fails, so a request held for the
        whole phrase could be retried and the reply spoken twice. A reply that cannot be
        spoken stays stored with delivered=0."""
        task_id = _task_id(payload)
        task = self.store.get(task_id) if task_id is not None else None
        if task is None:
            return 404, {"error": "unknown task"}
        content = payload["content"]
        answered_ts = self.store.add_answer(task_id, content)
        speak = asyncio.create_task(self._speak(task, content, answered_ts))
        self._speaking.add(speak)
        speak.add_done_callback(self._speaking.discard)
        return 200, {"id": task_id}

    async def _speak(self, task, content, answered_ts) -> None:
        """Speak one stored reply on the task's speaker; mark it delivered when spoken.
        A device without a speaker still gets the text on its screen (announce skips the
        speech there), but the reply stays undelivered: nobody heard it.

        Device.announce records the reply as a run (reason "assistant", the task text as
        the request) and adds it to the speaker's conversation context. A reply with no
        speaker to announce on is recorded here as a "device" error run."""
        task_id, device = task["id"], task["device"]
        async with self._speak_lock:
            # The first online speaker when the task has none, as DeviceManager.announce does.
            clients = self.rt.manager.clients
            if device is not None:
                client = next((c for c in clients if c.name == device), None)
            else:
                client = next((c for c in clients if c.online), None)
            if client is None:
                logger.warning(f"assistant task №{task_id}: speaker {device!r} unavailable; "
                               f"reply stored, not spoken")
                await record_run(self.rt, new_record(
                    device, "assistant", result="error", stt_text=task["text"], llm_text=content,
                    error_stage="device", error_text="device unavailable", t_total=0))
                return
            try:
                await client.announce(content, "assistant", task["text"])
            except Exception as e:
                logger.error(f"assistant task №{task_id}: speaking the reply on {client.name} failed: {e}")
                return
            if client.pipeline.has_speaker:
                self.store.mark_delivered(task_id, answered_ts)

    def recent(self, limit) -> list[dict]:
        return self.store.recent(limit)
