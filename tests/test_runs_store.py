"""Unit tests for the SQLite run log (src.runs_store.RunsStore)."""

import asyncio
import sqlite3
import time

from src.runs_store import RunsStore, _LIST_COLS, live_row, summary_row


def _store(tmp_path):
    return RunsStore(str(tmp_path / "runs.db"))


def _rec(**kw):
    """A minimal-but-complete run record; override fields via kwargs."""
    base = {
        "ts": time.time(),
        "device": "dev",
        "result": "ok",
        "reason": "endpoint",
        "stt_text": "включи свет",
        "llm_text": "Готово.",
        "model": "anthropic/claude-haiku-4.5",
        "tokens": 42,
        "t_vad": 1000, "t_stt": 300, "t_llm": 500, "t_ruaccent": 0, "t_tts": 200,
        "t_total": 2000,
        "audio_ms": None, "audio_bytes": 1234, "audio_fmt": "mp3",
        "error_stage": None, "error_text": None,
        "rounds": [],
    }
    base.update(kw)
    return base


def test_insert_and_get_round_trip(tmp_path):
    store = _store(tmp_path)
    rounds = [
        {"round": 1, "note": "tool call", "tokens": 30,
         "calls": [{"name": "set_light", "args": {"state": "on"}, "result": "ok"}]},
        {"round": 2, "note": "final answer", "tokens": 12, "calls": []},
    ]
    rid = store.insert(_rec(result="tool", rounds=rounds))
    assert isinstance(rid, int) and rid > 0

    got = store.get(rid)
    assert got is not None
    assert got["id"] == rid
    assert got["result"] == "tool"
    assert got["stt_text"] == "включи свет"
    assert got["audio_bytes"] == 1234
    assert got["audio_fmt"] == "mp3"
    # rounds_json parsed back into a list of dicts.
    assert got["rounds"] == rounds
    assert "rounds_json" not in got
    store.close()


def test_insert_and_get_request_round_trip(tmp_path):
    # A record carrying a `request` debug dict persists and round-trips through get().
    store = _store(tmp_path)
    request = {
        "system_prompt": "You are a helpful assistant.\n[MCP tools]",
        "context": [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "Здравствуйте."}],
        "user_text": "включи свет",
        "tools": [{"type": "function", "function": {"name": "set_light", "description": "Toggle a light"}}],
    }
    rid = store.insert(_rec(request=request))
    got = store.get(rid)
    assert got["request"] == request
    assert "request_json" not in got
    store.close()


def test_get_request_is_none_when_absent(tmp_path):
    # A record WITHOUT a `request` key yields request=None (column defaults to NULL).
    store = _store(tmp_path)
    rec = _rec()  # the base record carries no `request`
    assert "request" not in rec
    rid = store.insert(rec)
    got = store.get(rid)
    assert got["request"] is None
    store.close()


def test_get_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get(999) is None
    store.close()


def test_fresh_db_has_filler_columns(tmp_path):
    # A fresh DB must carry the filler columns straight from _CREATE_TABLE.
    store = _store(tmp_path)
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(runs)")}
    assert "filler_text" in cols
    assert "t_filler" in cols
    store.close()


def test_insert_and_get_filler_fields_round_trip(tmp_path):
    # A record carrying filler_text/t_filler persists and round-trips through get().
    store = _store(tmp_path)
    rid = store.insert(_rec(filler_text="Щас гляну…", t_filler=123))
    got = store.get(rid)
    assert got["filler_text"] == "Щас гляну…"
    assert got["t_filler"] == 123
    store.close()


def test_insert_without_filler_keys_defaults_to_null(tmp_path):
    # A record WITHOUT the filler keys still inserts fine; the columns default to NULL.
    store = _store(tmp_path)
    rec = _rec()  # the base record carries neither filler_text nor t_filler
    assert "filler_text" not in rec and "t_filler" not in rec
    rid = store.insert(rec)
    got = store.get(rid)
    assert got["filler_text"] is None
    assert got["t_filler"] is None
    store.close()


# The pre-filler schema, used to simulate an old DB that the migration must upgrade.
_OLD_CREATE_TABLE = """
CREATE TABLE runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, device TEXT, result TEXT, reason TEXT,
  stt_text TEXT, llm_text TEXT, model TEXT, tokens INTEGER,
  t_vad INTEGER, t_stt INTEGER, t_llm INTEGER, t_ruaccent INTEGER, t_tts INTEGER, t_total INTEGER,
  audio_ms INTEGER, audio_bytes INTEGER, audio_fmt TEXT,
  error_stage TEXT, error_text TEXT, rounds_json TEXT
);
"""


def test_migration_adds_filler_columns_to_old_db(tmp_path):
    # Simulate a pre-existing DB created BEFORE the filler columns: create the runs
    # table with the old schema, then open a RunsStore over it. The migration must
    # ALTER in the missing columns, and insert/get must round-trip the new fields.
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute(_OLD_CREATE_TABLE)
    conn.commit()
    conn.close()

    store = RunsStore(path)
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(runs)")}
    assert "filler_text" in cols
    assert "t_filler" in cols
    assert "request_json" in cols

    rid = store.insert(_rec(filler_text="Ну, погуглю…", t_filler=77,
                            request={"system_prompt": "p", "context": [], "user_text": "u", "tools": []}))
    got = store.get(rid)
    assert got["filler_text"] == "Ну, погуглю…"
    assert got["t_filler"] == 77
    assert got["request"]["user_text"] == "u"
    store.close()


def test_migration_is_idempotent(tmp_path):
    # Opening a store twice (the second time the columns already exist) must not raise:
    # _migrate ALTERs only what's missing.
    path = str(tmp_path / "twice.db")
    RunsStore(path).close()
    store = RunsStore(path)  # second open: columns already present
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(runs)")}
    assert "filler_text" in cols and "t_filler" in cols
    store.close()


def test_list_omits_rounds_and_orders_newest_first(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    store.insert(_rec(ts=now - 10, stt_text="первый"))
    store.insert(_rec(ts=now, stt_text="второй"))

    runs = store.list()
    assert [r["stt_text"] for r in runs] == ["второй", "первый"]
    # Summary payload carries timings/result but not rounds_json/rounds.
    assert "rounds_json" not in runs[0]
    assert "rounds" not in runs[0]
    assert runs[0]["t_total"] == 2000
    assert runs[0]["t_stt"] == 300
    store.close()


def test_list_filters(tmp_path):
    store = _store(tmp_path)
    store.insert(_rec(device="kitchen", result="ok", stt_text="свет на кухне"))
    store.insert(_rec(device="bedroom", result="tool", stt_text="музыка"))
    store.insert(_rec(device="bedroom", result="error", stt_text="ошибка тут",
                      error_stage="LLM"))

    # device exact.
    assert {r["stt_text"] for r in store.list(device="bedroom")} == {"музыка", "ошибка тут"}
    # result "errors" -> only error rows.
    errs = store.list(result="errors")
    assert [r["result"] for r in errs] == ["error"]
    # result "ok" -> ok + tool.
    oks = store.list(result="ok")
    assert {r["result"] for r in oks} == {"ok", "tool"}
    # result exact passthrough.
    assert [r["result"] for r in store.list(result="tool")] == ["tool"]
    # search LIKE on stt_text/llm_text.
    assert [r["stt_text"] for r in store.list(search="кухн")] == ["свет на кухне"]
    store.close()


def test_list_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.insert(_rec(ts=time.time() + i, stt_text=f"q{i}"))
    assert len(store.list(limit=2)) == 2
    store.close()


def test_metrics_over_window(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    # Five recent runs with known t_total for percentile checks.
    store.insert(_rec(ts=now, result="ok", t_total=100, t_vad=10, t_stt=20, t_llm=30, t_tts=40))
    store.insert(_rec(ts=now, result="ok", t_total=200, t_vad=10, t_stt=20, t_llm=30, t_tts=40))
    store.insert(_rec(ts=now, result="tool", t_total=300, t_vad=10, t_stt=20, t_llm=30, t_tts=40))
    store.insert(_rec(ts=now, result="error", t_total=400, t_vad=10, t_stt=20, t_llm=30, t_tts=40,
                      error_stage="TTS"))
    # An "empty" run (empty STT): t_total present, t_llm/t_tts zero.
    store.insert(_rec(ts=now, result="empty", t_total=500, t_vad=10, t_stt=20, t_llm=0, t_tts=0,
                      llm_text="", model=None, tokens=None))
    # A run older than 24h is excluded from the window.
    store.insert(_rec(ts=now - 90000, result="error", t_total=99999, error_stage="LLM"))

    m = store.metrics(now=now)
    assert m["requests_24h"] == 5
    # 5 values sorted: [100,200,300,400,500]; p50 -> 300, p95 -> 500 (nearest-rank).
    assert m["p50_ms"] == 300
    assert m["p95_ms"] == 500
    # 1 of 5 is an error.
    assert m["error_rate"] == 1 / 5
    # Per-stage averages over the window.
    assert m["per_stage_avg_ms"]["vad"] == 10
    assert m["per_stage_avg_ms"]["stt"] == 20
    assert m["per_stage_avg_ms"]["llm"] == (30 + 30 + 30 + 30 + 0) / 5
    assert m["per_stage_avg_ms"]["tts"] == (40 + 40 + 40 + 40 + 0) / 5
    store.close()


def test_metrics_all_none_timings_in_nonempty_window(tmp_path):
    # Rows that errored before any timing was set (t_total + all stage timings None)
    # still count toward requests_24h, but every percentile/average degrades to None.
    # This exercises the empty-branch of _percentile (no t_total values) and _avg (no
    # per-stage values) WITHOUT hitting the requests==0 early return — there ARE rows
    # in the window, so the None results must come from the empty aggregation branches.
    store = _store(tmp_path)
    now = time.time()
    for i in range(3):
        store.insert(_rec(
            ts=now - i, result="error", error_stage="pipeline",
            t_total=None, t_vad=None, t_stt=None, t_llm=None, t_tts=None,
        ))

    m = store.metrics(now=now)
    # There ARE rows in the window: the None results are NOT the empty early-return.
    assert m["requests_24h"] == 3
    # No t_total values -> _percentile([]) -> None for both percentiles.
    assert m["p50_ms"] is None
    assert m["p95_ms"] is None
    # No per-stage values -> _avg over all-None -> None for every stage.
    assert m["per_stage_avg_ms"] == {"vad": None, "stt": None, "llm": None, "tts": None}
    # All 3 in-window rows are errors.
    assert m["error_rate"] == 1.0
    store.close()


def test_metrics_empty(tmp_path):
    store = _store(tmp_path)
    m = store.metrics(now=time.time())
    assert m == {
        "requests_24h": 0,
        "p50_ms": None,
        "p95_ms": None,
        "error_rate": 0.0,
        "per_stage_avg_ms": {"vad": None, "stt": None, "llm": None, "tts": None},
    }
    store.close()


def test_concurrent_reads_and_writes_do_not_error(tmp_path):
    """Reads and writes share one Connection across to_thread worker threads.

    They are all guarded by the store's write lock, so hammering insert() and
    list()/get()/metrics() concurrently must not raise (no cross-thread sqlite3
    Connection race) and must leave a consistent row count.
    """
    store = _store(tmp_path)
    now = time.time()
    n_inserts = 40

    async def main():
        async def do_insert(i):
            return await asyncio.to_thread(store.insert, _rec(ts=now + i, stt_text=f"q{i}"))

        async def do_list():
            return await asyncio.to_thread(store.list)

        async def do_metrics():
            return await asyncio.to_thread(store.metrics, now=now + n_inserts)

        # Interleave many concurrent inserts with reads on the same store.
        tasks = []
        for i in range(n_inserts):
            tasks.append(do_insert(i))
            tasks.append(do_list())
            tasks.append(do_metrics())
        # Returns results in order; any worker-thread exception propagates here.
        return await asyncio.gather(*tasks)

    results = asyncio.run(main())

    # New row ids returned by every insert are all distinct (no lost/duplicated
    # writes under concurrency).
    insert_ids = [r for r in results if isinstance(r, int)]
    assert len(insert_ids) == n_inserts
    assert len(set(insert_ids)) == n_inserts

    # After all inserts settle, the store holds exactly n_inserts rows.
    final = store.list(limit=n_inserts + 10)
    assert len(final) == n_inserts
    m = store.metrics(now=now + n_inserts)
    assert m["requests_24h"] == n_inserts
    store.close()


def test_summary_row_shape_matches_list_cols():
    # summary_row builds a list()-shaped dict from an insert record + new id.
    rec = _rec(stt_text="включи свет", llm_text="Готово.", t_total=2000)
    row = summary_row(rec, 7)
    # The summary columns plus the computed has_audio flag (defaults to 0).
    assert set(row.keys()) == set(_LIST_COLS) | {"has_audio"}
    assert row["has_audio"] == 0
    # id comes from the run_id arg; other fields copied from rec.
    assert row["id"] == 7
    assert row["stt_text"] == "включи свет"
    assert row["llm_text"] == "Готово."
    assert row["t_total"] == 2000


def test_summary_row_missing_keys_become_none():
    # Keys absent from the record default to None (mirrors insert defaults).
    row = summary_row({"device": "kitchen"}, 3)
    assert row["id"] == 3
    assert row["device"] == "kitchen"
    assert row["stt_text"] is None
    assert row["t_total"] is None


def test_live_row_shape_and_flags():
    # live_row builds an in-progress row: no DB id, flagged live, summary shape.
    rec = _rec(stt_text="включи свет", llm_text="Готово.", t_total=2000)
    row = live_row(rec)
    assert row["id"] is None
    assert row["live"] == 1
    assert row["has_audio"] == 0
    assert set(row.keys()) == set(_LIST_COLS) | {"has_audio", "live"}
    # Provided fields are echoed through.
    assert row["stt_text"] == "включи свет"


def test_prune_drops_old_rows(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    keep = store.insert(_rec(ts=now, stt_text="recent"))
    old = store.insert(_rec(ts=now - 40 * 86400, stt_text="ancient"))

    store.prune(now=now, retention_days=30)

    assert store.get(keep) is not None
    assert store.get(old) is None
    assert [r["stt_text"] for r in store.list()] == ["recent"]
    store.close()


def test_prune_keeps_all_when_retention_zero(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    recent = store.insert(_rec(ts=now, stt_text="recent"))
    ancient = store.insert(_rec(ts=now - 999 * 86400, stt_text="ancient"))

    store.prune(now=now, retention_days=0)  # 0 == keep forever

    assert store.get(recent) is not None
    assert store.get(ancient) is not None
    store.close()


def test_put_get_audio_round_trip(tmp_path):
    store = _store(tmp_path)
    rid = store.insert(_rec())
    wav = b"RIFF....WAVEfmt fake-pcm-bytes"
    store.put_audio(rid, wav, keep=100)
    assert store.get_audio(rid) == wav
    # A run with no stored audio returns None.
    assert store.get_audio(999) is None
    store.close()


def test_put_audio_insert_or_replace_updates_bytes(tmp_path):
    store = _store(tmp_path)
    rid = store.insert(_rec())
    store.put_audio(rid, b"first", keep=100)
    store.put_audio(rid, b"second", keep=100)
    assert store.get_audio(rid) == b"second"
    store.close()


def test_put_audio_ring_buffer_keeps_newest(tmp_path):
    store = _store(tmp_path)
    # Insert several runs and store audio for each; run_id AUTOINCREMENTs so the
    # newest 3 ids must survive the keep=3 ring buffer, older ones get pruned.
    rids = []
    for i in range(6):
        rid = store.insert(_rec(stt_text=f"q{i}"))
        rids.append(rid)
        store.put_audio(rid, f"wav-{rid}".encode(), keep=3)

    survivors = [r for r in rids if store.get_audio(r) is not None]
    assert survivors == rids[-3:]
    # Older run_ids no longer carry audio.
    for r in rids[:-3]:
        assert store.get_audio(r) is None
    store.close()


def test_list_includes_has_audio(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    with_audio = store.insert(_rec(ts=now, stt_text="with"))
    without_audio = store.insert(_rec(ts=now - 1, stt_text="without"))
    store.put_audio(with_audio, b"wav-bytes", keep=100)

    by_id = {r["id"]: r for r in store.list()}
    assert by_id[with_audio]["has_audio"] == 1
    assert by_id[without_audio]["has_audio"] == 0
    store.close()


def test_get_includes_has_audio(tmp_path):
    store = _store(tmp_path)
    with_audio = store.insert(_rec())
    without_audio = store.insert(_rec())
    store.put_audio(with_audio, b"wav-bytes", keep=100)

    assert store.get(with_audio)["has_audio"] == 1
    assert store.get(without_audio)["has_audio"] == 0
    store.close()


def test_prune_deletes_orphaned_audio(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    keep = store.insert(_rec(ts=now, stt_text="recent"))
    old = store.insert(_rec(ts=now - 40 * 86400, stt_text="ancient"))
    store.put_audio(keep, b"recent-wav", keep=100)
    store.put_audio(old, b"old-wav", keep=100)

    store.prune(now=now, retention_days=30)

    # The old run AND its audio are gone; the recent run's audio survives.
    assert store.get(old) is None
    assert store.get_audio(old) is None
    assert store.get_audio(keep) == b"recent-wav"
    store.close()
