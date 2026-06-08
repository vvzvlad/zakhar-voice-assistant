"""End-to-end tests for the admin-panel HTTP API.

Uses aiohttp's built-in TestServer/TestClient (no extra dependency). asyncio_mode=auto
is on, so async test functions run directly. Each test builds a ConfigService over a
tmp config file and drives a fresh PanelServer app.
"""

import asyncio
import time

import aiohttp
import httpx
import pytest

import src.plugins  # noqa: F401  register all providers
from aiohttp.test_utils import TestClient, TestServer
from src.config_service import ConfigService
from src.panel_api import PanelServer
from src.plugins.base import Deps
from src.run_events import RunEventsHub
from src.runs_store import RunsStore


def _doc(prompt_path):
    """Minimal valid config doc: selected stt/llm/tts + instances + core."""
    return {
        "version": 1,
        "stt": {"selected": "groq", "instances": {"groq": {"api_key": "gsk-x"}}},
        "llm": {
            "selected": "openrouter",
            "instances": {"openrouter": {"api_key": "sk-or-x", "model": "m1"}},
        },
        "tts": {
            "selected": "yandex",
            "instances": {"yandex": {"api_key": "AQVN-x", "voice": "jane", "speed": 1.2}},
        },
        "core": {
            "prompt": {"system_prompt_path": str(prompt_path)},
            "log_level": "DEBUG",
        },
    }


def _svc(tmp_path):
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("seed prompt", encoding="utf-8")
    deps = Deps(http_cloud=httpx.AsyncClient(), http_local=httpx.AsyncClient())
    return ConfigService(_doc(prompt_path), deps, path=str(tmp_path / "config.json"))


async def _client(tmp_path, **kw):
    svc = _svc(tmp_path)
    ev = asyncio.Event()
    srv = PanelServer(svc, "127.0.0.1", 0, version="9.9", started_at=time.time(),
                      restart_event=ev, **kw)
    client = TestClient(TestServer(srv.build_app()))
    await client.start_server()
    return client, svc, ev


async def test_get_catalog(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/catalog")
        assert resp.status == 200
        body = await resp.json()
        assert "categories" in body
    finally:
        await client.close()


async def test_get_config(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/config")
        assert resp.status == 200
        body = await resp.json()
        assert all(k in body for k in ("stt", "llm", "tts", "core"))
    finally:
        await client.close()


async def test_patch_config_valid(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        assert svc.pending_restart is False
        resp = await client.patch("/api/config",
                                  json={"core": {"context": {"max_turns": 9}}})
        assert resp.status == 200
        # GET reflects the change and pending_restart flipped on.
        got = await (await client.get("/api/config")).json()
        assert got["core"]["context"]["max_turns"] == 9
        assert svc.pending_restart is True
    finally:
        await client.close()


async def test_patch_config_invalid_returns_422(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        # vad.aggressiveness has le=3; 9 fails core validation -> ValidationError -> 422.
        resp = await client.patch("/api/config",
                                  json={"core": {"vad": {"aggressiveness": 9}}})
        assert resp.status == 422
        body = await resp.json()
        assert "error" in body and "detail" in body
        assert isinstance(body["detail"], list)
    finally:
        await client.close()


async def test_patch_config_bad_json_returns_400(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.patch("/api/config", data="not json",
                                  headers={"Content-Type": "application/json"})
        assert resp.status == 400
    finally:
        await client.close()


async def test_patch_config_invalid_leaves_config_unchanged(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        # aggressiveness defaults to 2; an invalid patch must NOT persist anything.
        assert svc.document()["core"]["vad"]["aggressiveness"] == 2
        resp = await client.patch("/api/config",
                                  json={"core": {"vad": {"aggressiveness": 9}}})
        assert resp.status == 422
        # On-disk/in-memory config kept the old value and no restart was queued.
        assert svc.document()["core"]["vad"]["aggressiveness"] == 2
        assert svc.pending_restart is False
    finally:
        await client.close()


async def test_patch_config_non_object_body_returns_400(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        for bad in ([1, 2, 3], "just a string"):
            resp = await client.patch("/api/config", json=bad)
            assert resp.status == 400
            assert "error" in await resp.json()
        # Nothing was applied.
        assert svc.pending_restart is False
    finally:
        await client.close()


async def test_get_options(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "yandex", "field": "voice"})
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body["options"], list)
        assert "zahar" in body["options"]
    finally:
        await client.close()


async def test_options_unknown_field_returns_empty_list(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "yandex", "field": "nope"})
        assert resp.status == 200
        assert (await resp.json())["options"] == []
    finally:
        await client.close()


async def test_options_missing_params_returns_400(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/options")
        assert resp.status == 400
        assert "error" in await resp.json()
    finally:
        await client.close()


async def test_options_unknown_plugin_returns_404(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "nope", "field": "voice"})
        assert resp.status == 404
        assert "error" in await resp.json()
    finally:
        await client.close()


async def test_prompt_round_trip(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/prompt")
        assert resp.status == 200
        body = await resp.json()
        assert body["text"] == "seed prompt"
        assert body["path"] == svc.core.prompt.system_prompt_path

        put = await client.put("/api/prompt", json={"text": "updated prompt"})
        assert put.status == 200
        assert (await put.json()) == {"ok": True}

        # Re-read reflects the new text.
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "updated prompt"
    finally:
        await client.close()


async def test_put_prompt_non_object_returns_400(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        resp = await client.put("/api/prompt", json="x")
        assert resp.status == 400
        assert "error" in await resp.json()
        # The prompt on disk was NOT clobbered.
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "seed prompt"
    finally:
        await client.close()


async def test_put_prompt_missing_text_returns_400(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.put("/api/prompt", json={})
        assert resp.status == 400
        assert "error" in await resp.json()
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "seed prompt"
    finally:
        await client.close()


async def test_cors_header_present_on_error_response(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        # 400 (missing options params) and 404 (unknown plugin) still carry CORS.
        bad = await client.get("/api/options")
        assert bad.status == 400
        assert bad.headers["Access-Control-Allow-Origin"] == "*"

        not_found = await client.get("/api/options",
                                     params={"category": "tts", "plugin": "nope", "field": "voice"})
        assert not_found.status == 404
        assert not_found.headers["Access-Control-Allow-Origin"] == "*"

        # A router-level 404 (raised HTTPException) also carries CORS now.
        missing = await client.get("/api/does-not-exist")
        assert missing.status == 404
        assert missing.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        await client.close()


async def test_get_system(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/system")
        assert resp.status == 200
        body = await resp.json()
        assert body["version"] == "9.9"
        assert body["running"] is True
        assert body["pending_restart"] is False
        assert body["log_level"] == "DEBUG"
        assert "started" in body
        assert isinstance(body["uptime_seconds"], int)
    finally:
        await client.close()


async def test_post_restart(tmp_path):
    client, _svc_, ev = await _client(tmp_path)
    try:
        resp = await client.post("/api/restart")
        assert resp.status == 202
        assert (await resp.json()) == {"restarting": True}
        assert ev.is_set()
    finally:
        await client.close()


async def test_get_devices(tmp_path):
    devices = [{"name": "x", "host": "y", "online": True}]
    client, _svc_, _ev = await _client(tmp_path, device_status=lambda: devices)
    try:
        resp = await client.get("/api/devices")
        assert resp.status == 200
        assert (await resp.json()) == devices
    finally:
        await client.close()


async def test_get_devices_default_empty(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/devices")
        assert resp.status == 200
        assert (await resp.json()) == []
    finally:
        await client.close()


async def test_get_tools(tmp_path):
    # tool_sources is a zero-arg callable returning ToolHub.describe() output.
    sources = [{
        "id": "home", "kind": "http", "online": True,
        "tools": [{"name": "light.set", "description": "x"}],
    }]
    client, _svc_, _ev = await _client(tmp_path, tool_sources=lambda: sources)
    try:
        resp = await client.get("/api/tools")
        assert resp.status == 200
        assert (await resp.json()) == {"sources": sources}
    finally:
        await client.close()


async def test_get_tools_default_empty(tmp_path):
    # No tool_sources -> empty list.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/tools")
        assert resp.status == 200
        assert (await resp.json()) == {"sources": []}
    finally:
        await client.close()


async def test_cors_preflight(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.options("/api/config")
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        await client.close()


# --- observability endpoints (runs + metrics) --------------------------------

def _seed_runs(tmp_path):
    """A RunsStore over a tmp db seeded with a couple of runs."""
    store = RunsStore(str(tmp_path / "runs.db"))
    now = time.time()
    store.insert({
        "ts": now, "device": "kitchen", "result": "ok", "reason": "endpoint",
        "stt_text": "включи свет", "llm_text": "Готово.", "model": "m1", "tokens": 10,
        "t_vad": 1000, "t_stt": 200, "t_llm": 300, "t_ruaccent": 0, "t_tts": 100,
        "t_total": 1600, "audio_ms": None, "audio_bytes": 500, "audio_fmt": "mp3",
        "error_stage": None, "error_text": None,
        "rounds": [{"round": 1, "note": "final answer", "tokens": 10, "calls": []}],
    })
    store.insert({
        "ts": now, "device": "bedroom", "result": "error", "reason": "endpoint",
        "stt_text": "сломайся", "llm_text": "Ошибка: boom", "model": None, "tokens": None,
        "t_vad": 1000, "t_stt": 200, "t_llm": 300, "t_ruaccent": 0, "t_tts": 0,
        "t_total": 1500, "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
        "error_stage": "LLM", "error_text": "Ошибка: boom", "rounds": [],
    })
    return store


async def test_get_runs_list_and_filters(tmp_path):
    store = _seed_runs(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs")
        assert resp.status == 200
        runs = (await resp.json())["runs"]
        assert len(runs) == 2
        # Summary payload omits the heavy rounds field.
        assert "rounds" not in runs[0]

        # device filter.
        only_kitchen = await (await client.get("/api/runs", params={"device": "kitchen"})).json()
        assert [r["device"] for r in only_kitchen["runs"]] == ["kitchen"]

        # result=errors filter.
        errs = await (await client.get("/api/runs", params={"result": "errors"})).json()
        assert [r["result"] for r in errs["runs"]] == ["error"]

        # search filter.
        found = await (await client.get("/api/runs", params={"search": "свет"})).json()
        assert [r["stt_text"] for r in found["runs"]] == ["включи свет"]
    finally:
        await client.close()
        store.close()


async def test_get_run_by_id_and_404(tmp_path):
    store = _seed_runs(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        # The first inserted run has id 1; it carries the parsed rounds list.
        resp = await client.get("/api/runs/1")
        assert resp.status == 200
        body = await resp.json()
        assert body["id"] == 1
        assert body["device"] == "kitchen"
        assert body["rounds"] == [{"round": 1, "note": "final answer", "tokens": 10, "calls": []}]

        missing = await client.get("/api/runs/9999")
        assert missing.status == 404
    finally:
        await client.close()
        store.close()


async def test_get_metrics(tmp_path):
    store = _seed_runs(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/metrics")
        assert resp.status == 200
        body = await resp.json()
        assert body["requests_24h"] == 2
        assert body["error_rate"] == 0.5
        assert "p50_ms" in body and "p95_ms" in body
        assert set(body["per_stage_avg_ms"]) == {"vad", "stt", "llm", "tts"}
    finally:
        await client.close()
        store.close()


async def test_runs_endpoints_empty_without_store(tmp_path):
    # No runs_store -> list returns empty, metrics returns zeros, get id is 404.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        runs = await (await client.get("/api/runs")).json()
        assert runs == {"runs": []}

        metrics = await (await client.get("/api/metrics")).json()
        assert metrics["requests_24h"] == 0
        assert metrics["error_rate"] == 0.0
        assert metrics["p50_ms"] is None

        missing = await client.get("/api/runs/1")
        assert missing.status == 404
    finally:
        await client.close()


# --- live run stream (WebSocket) ---------------------------------------------

async def test_runs_stream_broadcasts_to_connected_client(tmp_path):
    hub = RunEventsHub()
    client, _svc_, _ev = await _client(tmp_path, run_events=hub)
    try:
        ws = await client.ws_connect("/api/runs/stream")
        # Give the server a moment so the handler registers the socket.
        for _ in range(50):
            if hub.count() == 1:
                break
            await asyncio.sleep(0.01)
        assert hub.count() == 1

        payload = {"type": "run", "run": {"id": 7, "device": "kitchen"}}
        await hub.broadcast(payload)
        msg = await asyncio.wait_for(ws.receive_json(), 2)
        assert msg == payload

        await ws.close()
    finally:
        await client.close()


async def test_runs_stream_without_hub_closes_promptly(tmp_path):
    # No run_events hub on the server -> the handler closes the socket immediately.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        ws = await client.ws_connect("/api/runs/stream")
        # The first receive should observe the close from the server side.
        msg = await asyncio.wait_for(ws.receive(), 2)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING)
        await ws.close()
    finally:
        await client.close()
