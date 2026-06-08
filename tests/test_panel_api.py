"""End-to-end tests for the admin-panel HTTP API.

Uses aiohttp's built-in TestServer/TestClient (no extra dependency). asyncio_mode=auto
is on, so async test functions run directly. Each test builds a ConfigService over a
tmp config file and drives a fresh PanelServer app.
"""

import asyncio
import time

import httpx
import pytest

import src.plugins  # noqa: F401  register all providers
from aiohttp.test_utils import TestClient, TestServer
from src.config_service import ConfigService
from src.panel_api import PanelServer
from src.plugins.base import Deps


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


async def test_cors_preflight(tmp_path):
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.options("/api/config")
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
    finally:
        await client.close()
