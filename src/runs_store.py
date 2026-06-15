"""SQLite-backed log of finalized pipeline runs (observability backend).

Each finalized voice run (STT -> LLM -> TTS) is appended as one row. Rounds (the
agentic tool-calling steps) are stored as a JSON string column to avoid a join.

All methods are synchronous; callers in async code offload them to a worker thread
via `asyncio.to_thread`. The store is shared across pipelines/devices, so the
connection is opened with `check_same_thread=False` and writes are serialized with
a `threading.Lock`. WAL journaling keeps concurrent readers from blocking writers.
"""

import json
import os
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
  stress_text TEXT,
  filler_text TEXT,
  model TEXT,
  tokens INTEGER,
  t_filler INTEGER,
  t_vad INTEGER, t_stt INTEGER, t_llm INTEGER, t_stress INTEGER, t_tts INTEGER, t_total INTEGER,
  audio_ms INTEGER, audio_bytes INTEGER, audio_fmt TEXT,
  error_stage TEXT, error_text TEXT,
  rounds_json TEXT,
  request_json TEXT
);
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);"

_CREATE_AUDIO_TABLE = """
CREATE TABLE IF NOT EXISTS run_audio (
  run_id INTEGER PRIMARY KEY,
  wav BLOB NOT NULL
);
"""

_CREATE_TTS_AUDIO_TABLE = """
CREATE TABLE IF NOT EXISTS run_tts_audio (
  run_id INTEGER PRIMARY KEY,
  audio BLOB NOT NULL,
  mime TEXT NOT NULL
);
"""

# Columns persisted on insert, in order. `id` autoincrements; `rounds_json` is
# derived from rec["rounds"] separately, so it is appended last by insert().
_INSERT_COLS = [
    "ts", "device", "result", "reason", "stt_text", "llm_text", "stress_text",
    "filler_text",
    "model", "tokens",
    "t_vad", "t_stt", "t_llm", "t_stress", "t_tts", "t_filler", "t_total",
    "audio_ms", "audio_bytes", "audio_fmt", "error_stage", "error_text",
]

# Summary columns returned by list() (rounds_json is intentionally omitted).
_LIST_COLS = [
    "id", "ts", "device", "result", "reason", "stt_text", "llm_text", "filler_text",
    "tokens",
    "t_vad", "t_stt", "t_llm", "t_stress", "t_tts", "t_filler", "t_total",
]

_DAY_SECONDS = 86400


class RunsStore:
    """Append-only run log over a single SQLite file."""

    def __init__(self, path):
        self._lock = threading.Lock()
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.execute(_CREATE_AUDIO_TABLE)
        self._conn.execute(_CREATE_TTS_AUDIO_TABLE)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema to a pre-existing DB.

        SQLite has no 'ADD COLUMN IF NOT EXISTS', so read the live column set first
        and ALTER only what's missing. Column names are fixed literals (not user
        input), so the f-string interpolation is safe. Idempotent."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(runs)")}
        # Rename the reserved accent-timing column to its provider-neutral name (the
        # stage is now `stress`, not the RuAccent provider). SQLite >= 3.25 supports
        # RENAME COLUMN; the bundled sqlite is newer. Only when the old name is present
        # and the new one is not, so it's idempotent.
        if "t_ruaccent" in existing and "t_stress" not in existing:
            self._conn.execute("ALTER TABLE runs RENAME COLUMN t_ruaccent TO t_stress")
            existing.discard("t_ruaccent")
            existing.add("t_stress")
        for col, decl in (("filler_text", "TEXT"), ("t_filler", "INTEGER"),
                          ("request_json", "TEXT"), ("t_stress", "INTEGER"),
                          ("stress_text", "TEXT")):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
        self._conn.commit()

    def insert(self, rec: dict) -> int:
        """Insert one run; rec["rounds"] (a list) is JSON-encoded into rounds_json.

        Missing keys default to None. Returns the new row id.
        """
        values = [rec.get(col) for col in _INSERT_COLS]
        values.append(json.dumps(rec.get("rounds") or [], ensure_ascii=False))
        values.append(json.dumps(rec.get("request"), ensure_ascii=False))
        cols = _INSERT_COLS + ["rounds_json", "request_json"]
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
        with self._lock:
            cur = self._conn.execute(sql, values)
            self._conn.commit()
            return cur.lastrowid

    def put_audio(self, run_id: int, wav: bytes, keep: int) -> None:
        """Store one run's utterance WAV, then prune to the newest `keep` rows.

        run_id is the runs.id AUTOINCREMENT value, so "newest" == highest run_id:
        the ring buffer keeps only the `keep` highest run_ids. INSERT OR REPLACE so a
        repeat for the same run_id updates the bytes. `keep` is clamped to >= 0 so a
        negative value can't disable pruning (SQLite treats LIMIT -1 as "no limit");
        keep == 0 keeps none.
        """
        keep = max(0, keep)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO run_audio (run_id, wav) VALUES (?, ?)",
                (run_id, sqlite3.Binary(wav)),
            )
            self._conn.execute(
                "DELETE FROM run_audio WHERE run_id NOT IN "
                "(SELECT run_id FROM run_audio ORDER BY run_id DESC LIMIT ?)",
                (keep,),
            )
            self._conn.commit()

    def get_audio(self, run_id: int) -> bytes | None:
        """Return the stored WAV bytes for one run, or None when not stored."""
        with self._lock:
            row = self._conn.execute(
                "SELECT wav FROM run_audio WHERE run_id = ?", (run_id,)
            ).fetchone()
        return bytes(row["wav"]) if row is not None else None

    def put_tts_audio(self, run_id: int, audio: bytes, mime: str, keep: int) -> None:
        """Store one run's generated TTS reply audio (native backend format) + its
        mime, then prune to the newest `keep` rows. Mirrors put_audio: an independent
        rolling ring keyed by run_id (highest id == newest). INSERT OR REPLACE so a
        repeat updates the bytes; `keep` clamped to >= 0 (keep == 0 keeps none)."""
        keep = max(0, keep)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO run_tts_audio (run_id, audio, mime) VALUES (?, ?, ?)",
                (run_id, sqlite3.Binary(audio), mime),
            )
            self._conn.execute(
                "DELETE FROM run_tts_audio WHERE run_id NOT IN "
                "(SELECT run_id FROM run_tts_audio ORDER BY run_id DESC LIMIT ?)",
                (keep,),
            )
            self._conn.commit()

    def get_tts_audio(self, run_id: int) -> tuple[bytes, str] | None:
        """Return (audio_bytes, mime) of the stored TTS reply audio, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT audio, mime FROM run_tts_audio WHERE run_id = ?", (run_id,)
            ).fetchone()
        return (bytes(row["audio"]), row["mime"]) if row is not None else None

    def audio_channels(self, run_id: int) -> int | None:
        """Channel count of the stored WAV for one run, or None when not stored.

        Reads only the first bytes of the blob (SQLite substr is 1-based and works
        on BLOBs) and parses the RIFF header, so the whole WAV is never loaded.
        Used by the panel to decide whether to render per-channel players.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT substr(wav, 1, 64) AS head FROM run_audio WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return _wav_channels(bytes(row["head"]))

    def db_size_bytes(self) -> int:
        """On-disk size of this store's SQLite file (+ WAL/SHM sidecars), in bytes."""
        return db_file_size(self._path)

    def _filter_sql(self, device, result, search):
        """Build the shared WHERE clause + params for list()/count() from the
        device/result/search filters. Returns (clause, params)."""
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
        return clause, params

    def list(self, *, device=None, result=None, search=None, limit=100, offset=0) -> list[dict]:
        """Recent runs (newest first) as summary dicts, with optional filters.

        - device: exact match.
        - result: "errors" -> result='error'; "ok" -> result IN ('ok','tool');
          anything else -> exact match.
        - search: LIKE on stt_text or llm_text.
        - offset: number of matching rows to skip before the page (for offset-based
          numbered pagination); paired with limit as LIMIT ? OFFSET ?.
        Rows are ordered by id DESC (newest-first, stable and unique).
        rounds_json is omitted from the summary payload.
        """
        clause, params = self._filter_sql(device, result, search)
        sql = (
            f"SELECT {', '.join(_LIST_COLS)}, "
            "EXISTS(SELECT 1 FROM run_audio WHERE run_audio.run_id = runs.id) "
            f"AS has_audio FROM runs{clause} ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        params.append(limit)
        params.append(offset)
        # Reads share the same Connection as writes and run on to_thread worker
        # threads, so they must hold the write lock too: concurrent execute()/
        # cursor use on one sqlite3.Connection from multiple threads is a race.
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self, *, device=None, result=None, search=None) -> int:
        """Total number of runs matching the filters (for paginated UIs)."""
        clause, params = self._filter_sql(device, result, search)
        # Reads share the write connection on worker threads — hold the lock too.
        with self._lock:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM runs{clause}", params).fetchone()
        return int(row["n"])

    def get(self, run_id) -> dict | None:
        """Full row for one run, with rounds_json/request_json parsed back into
        `rounds`/`request`."""
        # Same Connection as writes from a worker thread: hold the write lock.
        with self._lock:
            row = self._conn.execute(
                "SELECT *, "
                "EXISTS(SELECT 1 FROM run_audio WHERE run_audio.run_id = runs.id) "
                "AS has_audio, "
                "EXISTS(SELECT 1 FROM run_tts_audio WHERE run_tts_audio.run_id = runs.id) "
                "AS has_tts_audio FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        rec = dict(row)
        raw = rec.pop("rounds_json", None)
        try:
            rec["rounds"] = json.loads(raw) if raw else []
        except (ValueError, json.JSONDecodeError):
            rec["rounds"] = []
        raw_req = rec.pop("request_json", None)
        try:
            rec["request"] = json.loads(raw_req) if raw_req else None
        except (ValueError, json.JSONDecodeError):
            rec["request"] = None
        # Channel count of the stored utterance WAV (panel renders per-channel
        # players for stereo). Called OUTSIDE the lock block above: audio_channels
        # re-acquires self._lock itself and the lock is not reentrant.
        rec["audio_channels"] = self.audio_channels(run_id) if rec.get("has_audio") else None
        return rec

    def metrics(self, *, now: float) -> dict:
        """Aggregate metrics over the last 24h. Returns zeros/None when empty."""
        since = now - _DAY_SECONDS
        # Same Connection as writes from a worker thread: hold the write lock.
        with self._lock:
            rows = self._conn.execute(
                "SELECT result, t_total, t_vad, t_stt, t_llm, t_stress, t_tts "
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
                "per_stage_avg_ms": {"vad": None, "stt": None, "llm": None, "stress": None, "tts": None},
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
                "stress": _avg(r["t_stress"] for r in rows),
                "tts": _avg(r["t_tts"] for r in rows),
            },
        }

    def prune(self, *, now: float, retention_days: int):
        """Delete runs older than `retention_days`. retention_days<=0 keeps all rows."""
        if retention_days <= 0:
            return  # 0 (or negative) means "keep forever" — never prune.
        cutoff = now - retention_days * _DAY_SECONDS
        with self._lock:
            self._conn.execute("DELETE FROM runs WHERE ts < ?", (cutoff,))
            # Drop audio whose run row was just deleted so it can't outlive its run.
            self._conn.execute(
                "DELETE FROM run_audio WHERE run_id NOT IN (SELECT id FROM runs)"
            )
            self._conn.execute(
                "DELETE FROM run_tts_audio WHERE run_id NOT IN (SELECT id FROM runs)"
            )
            self._conn.commit()

    def close(self):
        # Serialize close with insert/list/get/metrics/prune: those run on
        # to_thread workers under the same lock on the shared sqlite3.Connection.
        # Closing without the lock would be a use-after-close race when
        # _rebuild_runs disables runs while a concurrent DB op is still running.
        with self._lock:
            self._conn.close()


def _wav_channels(header: bytes) -> int | None:
    """Channel count parsed from a WAV header prefix, or None when malformed.

    `header` is only the first bytes of the file (a prefix), not the whole WAV.
    Walks the RIFF chunk list looking for the "fmt " chunk instead of assuming a
    fixed layout, so files with extra chunks before "fmt " still parse as long as
    the chunk fits into the prefix.
    """
    if len(header) < 12 or header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        return None
    pos = 12
    # Each chunk: 4-byte id + 4-byte LE size + payload.
    while pos + 8 <= len(header):
        chunk_id = header[pos:pos + 4]
        size = int.from_bytes(header[pos + 4:pos + 8], "little")
        if chunk_id == b"fmt ":
            # Channel count is the LE uint16 at payload offset 2.
            if pos + 8 + 4 > len(header):
                return None  # fmt chunk truncated by the prefix
            return int.from_bytes(header[pos + 10:pos + 12], "little")
        pos += 8 + size + (size & 1)  # chunks are word-aligned
    return None


def db_file_size(path: str) -> int:
    """Total on-disk size in bytes of a SQLite DB: the main file plus its WAL and
    SHM sidecars. Missing files count as 0, so a not-yet-created DB reports 0."""
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += os.path.getsize(path + suffix)
        except OSError:
            pass
    return total


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


def summary_row(rec: dict, run_id: int, has_audio: bool = False) -> dict:
    """Build a list()-shaped summary dict from an insert record + its new row id.

    Mirrors the columns list() returns (_LIST_COLS plus the computed has_audio), so a
    run pushed live over the WebSocket carries exactly the same shape the panel's
    GET /api/runs returns.
    """
    row = {col: (run_id if col == "id" else rec.get(col)) for col in _LIST_COLS}
    row["has_audio"] = 1 if has_audio else 0
    return row


def live_row(rec: dict) -> dict:
    """Build an in-progress (live) run row for incremental WS streaming.

    Same summary shape as summary_row but for a run that has NOT been persisted
    yet: it has no DB id (id stays None) and is flagged `live` so the panel can
    upsert the same row (keyed by device) as later stages stream in.
    """
    row = {col: rec.get(col) for col in _LIST_COLS}  # id absent in rec -> None
    row["has_audio"] = 0
    row["live"] = 1
    return row
