"""Named system-prompt profiles over a single SQLite file.

The user keeps several named prompts (profiles) and activates one; the active
profile's text is what the pipeline uses as the LLM system prompt. The store
mirrors src.runs_store / src.reminders conventions: a shared Connection opened
with check_same_thread=False, all methods synchronous and serialized by a
threading.Lock, WAL journaling.

Invariant: exactly one row has is_active=1, enforced inside the store under the
lock (activate clears every flag, then sets one, in a single transaction).

On first boot the empty table is seeded with one active "default" profile from
the legacy prompt file (data/system_prompt.md) when it exists, otherwise from
the committed default template. The legacy file is never deleted (it stays as a
safety backup).
"""

import os
import sqlite3
import threading
import time

from loguru import logger

DEFAULT_PROMPT_PATH = "templates/default_prompt.md"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prompt_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  text TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_ts REAL,
  updated_ts REAL
);
"""

# Columns of the full-row payload (get/active), in select order.
_COLS = ["id", "name", "text", "is_active", "created_ts", "updated_ts"]


def _read_default_prompt() -> str:
    """Text of the committed default prompt template."""
    with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _to_dict(row) -> dict:
    """sqlite3.Row -> plain dict with is_active coerced to bool."""
    rec = dict(row)
    if "is_active" in rec:
        rec["is_active"] = bool(rec["is_active"])
    return rec


class PromptStore:
    """Named system-prompt profiles over a single SQLite file."""

    def __init__(self, path, seed_path=None):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        self._seed_if_empty(seed_path)

    def _seed_if_empty(self, seed_path) -> None:
        """Insert the initial active "default" profile when the table is empty.

        The text comes from the legacy prompt file at `seed_path` when it exists
        (kept on disk as a safety backup, never deleted), otherwise from the
        committed default template.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM prompt_profiles"
            ).fetchone()
            if row["n"] > 0:
                return
            if seed_path and os.path.exists(seed_path):
                with open(seed_path, "r", encoding="utf-8") as f:
                    text = f.read()
                source = seed_path
            else:
                text = _read_default_prompt()
                source = DEFAULT_PROMPT_PATH
            now = time.time()
            self._conn.execute(
                "INSERT INTO prompt_profiles (name, text, is_active, created_ts, updated_ts) "
                "VALUES (?, ?, 1, ?, ?)",
                ("default", text, now, now),
            )
            self._conn.commit()
            logger.info(f"prompt store seeded with 'default' profile from {source}")

    def list_profiles(self) -> list[dict]:
        """All profiles ordered by name, WITHOUT the full text (chars = LENGTH(text))."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, is_active, created_ts, updated_ts, "
                "LENGTH(text) AS chars FROM prompt_profiles ORDER BY name ASC"
            ).fetchall()
        return [_to_dict(r) for r in rows]

    def get(self, profile_id) -> dict | None:
        """Full row (incl. text) for one profile, or None when not found."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM prompt_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
        return _to_dict(row) if row is not None else None

    def create(self, name, text) -> dict:
        """Insert one profile (inactive); stamps created_ts/updated_ts.

        Raises ValueError on a duplicate name (UNIQUE constraint).
        """
        now = time.time()
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO prompt_profiles (name, text, is_active, created_ts, updated_ts) "
                    "VALUES (?, ?, 0, ?, ?)",
                    (name, text, now, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                raise ValueError(f"profile name {name!r} is already in use")
            new_id = cur.lastrowid
        return self.get(new_id)

    def update(self, profile_id, *, name=None, text=None) -> dict | None:
        """Partial update (name and/or text); stamps updated_ts.

        Returns the refreshed row, or None when the id does not exist.
        Raises ValueError on a duplicate name.
        """
        sets, params = [], []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if text is not None:
            sets.append("text = ?")
            params.append(text)
        if not sets:
            return self.get(profile_id)  # nothing to change
        sets.append("updated_ts = ?")
        params.extend([time.time(), profile_id])
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"UPDATE prompt_profiles SET {', '.join(sets)} WHERE id = ?",
                    params,
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                raise ValueError(f"profile name {name!r} is already in use")
            if cur.rowcount == 0:
                return None
        return self.get(profile_id)

    def delete(self, profile_id) -> bool:
        """Delete one profile by id; return whether a row existed.

        REFUSES to delete the active profile (the exactly-one-active invariant
        would break) — raises ValueError instead.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT is_active FROM prompt_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if row is None:
                return False
            if row["is_active"]:
                raise ValueError("cannot delete the active profile")
            self._conn.execute(
                "DELETE FROM prompt_profiles WHERE id = ?", (profile_id,)
            )
            self._conn.commit()
            return True

    def activate(self, profile_id) -> bool:
        """Make one profile the active one (clears every other flag, one transaction).

        Returns False when no such id exists.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM prompt_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("UPDATE prompt_profiles SET is_active = 0")
            self._conn.execute(
                "UPDATE prompt_profiles SET is_active = 1 WHERE id = ?",
                (profile_id,),
            )
            self._conn.commit()
            return True

    def active(self) -> dict | None:
        """The active row (incl. text), or None when (defensively) none is flagged."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {', '.join(_COLS)} FROM prompt_profiles WHERE is_active = 1"
            ).fetchone()
        return _to_dict(row) if row is not None else None

    def active_text(self) -> str:
        """Text of the active profile — what the pipeline uses as the system prompt.

        Defensive fallback: when no active row exists (should be impossible — the
        store seeds itself and activate() preserves the invariant), the committed
        default template is returned so a run never goes out with an empty prompt.
        """
        row = self.active()
        if row is not None:
            return row["text"]
        logger.warning("prompt store has no active profile; using the default template")
        return _read_default_prompt()

    def close(self):
        with self._lock:
            self._conn.close()
