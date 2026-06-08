"""SQLite-backed log of finalized pipeline runs (observability backend).

Each finalized voice run (STT -> LLM -> TTS) is appended as one row. Rounds (the
agentic tool-calling steps) are stored as a JSON string column to avoid a join.

All methods are synchronous; callers in async code offload them to a worker thread
via `asyncio.to_thread`. The store is shared across pipelines/devices, so the
connection is opened with `check_same_thread=False` and writes are serialized with
a `threading.Lock`. WAL journaling keeps concurrent readers from blocking writers.
"""

import json
import sqlite3
import threading

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL,
  device TEXT,
  result TEXT,
  reason TEXT,
  stt_text TEXT,
  llm_text TEXT,
  model TEXT,
  tokens INTEGER,
  t_vad INTEGER, t_stt INTEGER, t_llm INTEGER, t_ruaccent INTEGER, t_tts INTEGER, t_total INTEGER,
  audio_ms INTEGER, audio_bytes INTEGER, audio_fmt TEXT,
  error_stage TEXT, error_text TEXT,
  rounds_json TEXT
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);"

# Columns persisted on insert, in order. `id` autoincrements; `rounds_json` is
# derived from rec["rounds"] separately, so it is appended last by insert().
_INSERT_COLS = [
    "ts", "device", "result", "reason", "stt_text", "llm_text", "model", "tokens",
    "t_vad", "t_stt", "t_llm", "t_ruaccent", "t_tts", "t_total",
    "audio_ms", "audio_bytes", "audio_fmt", "error_stage", "error_text",
]

# Summary columns returned by list() (rounds_json is intentionally omitted).
_LIST_COLS = [
    "id", "ts", "device", "result", "reason", "stt_text", "llm_text", "tokens",
    "t_vad", "t_stt", "t_llm", "t_ruaccent", "t_tts", "t_total",
]

_DAY_SECONDS = 86400


class RunsStore:
    """Append-only run log over a single SQLite file."""

    def __init__(self, path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def insert(self, rec: dict) -> int:
        """Insert one run; rec["rounds"] (a list) is JSON-encoded into rounds_json.

        Missing keys default to None. Returns the new row id.
        """
        values = [rec.get(col) for col in _INSERT_COLS]
        values.append(json.dumps(rec.get("rounds") or [], ensure_ascii=False))
        cols = _INSERT_COLS + ["rounds_json"]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
        with self._lock:
            cur = self._conn.execute(sql, values)
            self._conn.commit()
            return cur.lastrowid

    def list(self, *, device=None, result=None, search=None, limit=100) -> list[dict]:
        """Recent runs (newest first) as summary dicts, with optional filters.

        - device: exact match.
        - result: "errors" -> result='error'; "ok" -> result IN ('ok','tool');
          anything else -> exact match.
        - search: LIKE on stt_text or llm_text.
        rounds_json is omitted from the summary payload.
        """
        where = []
        params: list = []
        if device:
            where.append("device = ?")
            params.append(device)
        if result:
            if result == "errors":
                where.append("result = ?")
                params.append("error")
            elif result == "ok":
                where.append("result IN ('ok', 'tool')")
            else:
                where.append("result = ?")
                params.append(result)
        if search:
            where.append("(stt_text LIKE ? OR llm_text LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (
            f"SELECT {', '.join(_LIST_COLS)} FROM runs{clause} "
            "ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)
        # Reads share the same Connection as writes and run on to_thread worker
        # threads, so they must hold the write lock too: concurrent execute()/
        # cursor use on one sqlite3.Connection from multiple threads is a race.
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get(self, run_id) -> dict | None:
        """Full row for one run, with rounds_json parsed back into `rounds`."""
        # Same Connection as writes from a worker thread: hold the write lock.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        raw = rec.pop("rounds_json", None)
        try:
            rec["rounds"] = json.loads(raw) if raw else []
        except (ValueError, json.JSONDecodeError):
            rec["rounds"] = []
        return rec

    def metrics(self, *, now: float) -> dict:
        """Aggregate metrics over the last 24h. Returns zeros/None when empty."""
        since = now - _DAY_SECONDS
        # Same Connection as writes from a worker thread: hold the write lock.
        with self._lock:
            rows = self._conn.execute(
                "SELECT result, t_total, t_vad, t_stt, t_llm, t_tts "
                "FROM runs WHERE ts >= ?",
                (since,),
            ).fetchall()
        requests = len(rows)
        if requests == 0:
            return {
                "requests_24h": 0,
                "p50_ms": None,
                "p95_ms": None,
                "error_rate": 0.0,
                "per_stage_avg_ms": {"vad": None, "stt": None, "llm": None, "tts": None},
            }

        totals = sorted(r["t_total"] for r in rows if r["t_total"] is not None)
        errors = sum(1 for r in rows if r["result"] == "error")

        return {
            "requests_24h": requests,
            "p50_ms": _percentile(totals, 50),
            "p95_ms": _percentile(totals, 95),
            "error_rate": errors / requests,
            "per_stage_avg_ms": {
                "vad": _avg(r["t_vad"] for r in rows),
                "stt": _avg(r["t_stt"] for r in rows),
                "llm": _avg(r["t_llm"] for r in rows),
                "tts": _avg(r["t_tts"] for r in rows),
            },
        }

    def prune(self, *, now: float, retention_days: int):
        """Delete runs older than `retention_days`."""
        cutoff = now - retention_days * _DAY_SECONDS
        with self._lock:
            self._conn.execute("DELETE FROM runs WHERE ts < ?", (cutoff,))
            self._conn.commit()

    def close(self):
        self._conn.close()


def _percentile(sorted_values: list, pct: int):
    """Approximate percentile of a pre-sorted list, or None when empty.

    Picks the value at the nearest index into the sorted list — it maps the
    percentile onto the [0, N-1] index range and rounds to the closest index.
    This is a nearest-index pick, not the canonical nearest-rank percentile.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Map pct onto the [0, N-1] index range and round to the nearest index.
    idx = round((pct / 100) * (len(sorted_values) - 1))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def _avg(values):
    """Mean of the non-None values, or None when there are none."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)
