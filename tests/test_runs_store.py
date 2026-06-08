"""Unit tests for the SQLite run log (src.runs_store.RunsStore)."""

import asyncio
import time

from src.runs_store import RunsStore


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


def test_get_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get(999) is None
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
