"""End-to-end tests for the admin-panel HTTP API.

Uses aiohttp's built-in TestServer/TestClient (no extra dependency). asyncio_mode=auto
is on, so async test functions run directly. Each test builds a ConfigService over a
tmp config file and drives a fresh PanelServer app.
"""

import asyncio
import re
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
        resp = await client.patch("/api/config",
                                  json={"core": {"context": {"max_turns": 9}}})
        assert resp.status == 200
        # GET reflects the applied change. (pending_restart is owned by the
        # Reconfigurator now; this test wires no reconfigurator, so it stays False.)
        got = await (await client.get("/api/config")).json()
        assert got["core"]["context"]["max_turns"] == 9
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
        # On-disk/in-memory config kept the old value.
        assert svc.document()["core"]["vad"]["aggressiveness"] == 2
    finally:
        await client.close()


async def test_patch_config_non_object_body_returns_400(tmp_path):
    client, svc, _ev = await _client(tmp_path)
    try:
        for bad in ([1, 2, 3], "just a string"):
            resp = await client.patch("/api/config", json=bad)
            assert resp.status == 400
            assert "error" in await resp.json()
        # Nothing was applied: config document is untouched.
        assert svc.document()["core"]["vad"]["aggressiveness"] == 2
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
        # Default config has an empty allowlist, so error responses emit no ACAO
        # header (and never a wildcard) while status codes are unaffected.
        bad = await client.get("/api/options")
        assert bad.status == 400
        assert "Access-Control-Allow-Origin" not in bad.headers

        not_found = await client.get("/api/options",
                                     params={"category": "tts", "plugin": "nope", "field": "voice"})
        assert not_found.status == 404
        assert "Access-Control-Allow-Origin" not in not_found.headers

        # A router-level 404 (raised HTTPException) also carries no wildcard.
        missing = await client.get("/api/does-not-exist")
        assert missing.status == 404
        assert "Access-Control-Allow-Origin" not in missing.headers
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


async def test_get_system_pending_restart_true(tmp_path):
    # A reconfigurator whose flag is set surfaces as pending_restart=True.
    client, _svc_, _ev = await _client(tmp_path, pending_restart=lambda: True)
    try:
        resp = await client.get("/api/system")
        assert resp.status == 200
        body = await resp.json()
        assert body["pending_restart"] is True
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


def _wav_bytes(pcm=b"\x01\x02" * 16):
    """Build a tiny valid 16k/mono/16-bit WAV for the capture-download assertions."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


async def _poll_state(client, device, want, *, tries=200):
    """Poll GET /api/capture until the state matches `want` (one or many), then return it.

    Yields the event loop between polls so the background capture task can run; uses
    a bounded iteration count instead of wall-clock delays.
    """
    wants = {want} if isinstance(want, str) else set(want)
    last = None
    for _ in range(tries):
        last = await (await client.get("/api/capture", params={"device": device})).json()
        if last["state"] in wants:
            return last
        await asyncio.sleep(0)
    raise AssertionError(f"state {last and last['state']!r} never reached {wants}")


async def test_capture_start_status_result_lifecycle(tmp_path):
    # The recording runs as a background task: POST /api/capture returns 202
    # "recording" immediately; status reflects it; once the (event-gated) capture
    # finishes the result is downloadable as a valid WAV; status then goes idle.
    import io
    import wave

    calls = []
    wav = _wav_bytes()
    release = asyncio.Event()

    async def cap(device, seconds):
        calls.append((device, seconds))
        await release.wait()  # hold the "recording" until the test releases it
        return wav

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        # Start -> 202 with the initial snapshot (state recording).
        started = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert started.status == 202
        snap = await started.json()
        assert snap["state"] == "recording" and snap["device"] == "hall"

        # GET status shows the in-flight recording while the capture is blocked.
        st = await _poll_state(client, "hall", "recording")
        assert st["state"] == "recording"

        # Release the recording and poll until it finishes.
        release.set()
        await _poll_state(client, "hall", "done")

        # The result downloads as a valid 16k/mono/16-bit WAV with the stamped name.
        resp = await client.get("/api/capture/result", params={"device": "hall"})
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert re.fullmatch(
            r'attachment; filename="zakhar_hall_5s_\d{8}_\d{6}\.wav"',
            resp.headers["Content-Disposition"],
        )
        body = await resp.read()
        assert body == wav
        with wave.open(io.BytesIO(body), "rb") as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getframerate() == 16000
        assert calls == [("hall", 5)]

        # The result was consumed; status is now idle.
        idle = await (await client.get("/api/capture", params={"device": "hall"})).json()
        assert idle["state"] == "idle"
    finally:
        release.set()
        await client.close()


async def test_capture_result_filename_sanitizes_device(tmp_path):
    # A device name with unsafe chars is sanitized in the download filename.
    async def cap(device, seconds):
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        started = await client.post("/api/capture", json={"device": "living room/2", "seconds": 3})
        assert started.status == 202
        await _poll_state(client, "living room/2", "done")
        resp = await client.get("/api/capture/result", params={"device": "living room/2"})
        assert resp.status == 200
        # unsafe chars sanitized; trailing UTC timestamp appended (YYYYMMDD_HHMMSS)
        assert re.fullmatch(
            r'attachment; filename="zakhar_living_room_2_3s_\d{8}_\d{6}\.wav"',
            resp.headers["Content-Disposition"],
        )
    finally:
        await client.close()


async def test_capture_busy_returns_409(tmp_path):
    # While a capture is in flight on a device, a second POST /api/capture for that
    # same device is rejected with 409 and an "in progress" message.
    release = asyncio.Event()

    async def cap(device, seconds):
        await release.wait()
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        first = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert first.status == 202
        await _poll_state(client, "hall", "recording")

        second = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert second.status == 409
        assert "in progress" in (await second.json())["error"]
    finally:
        release.set()
        await client.close()


@pytest.mark.parametrize("exc, needle", [
    (RuntimeError("hall is offline"), "offline"),
    (LookupError("unknown device 'hall'"), "unknown device"),
    (TimeoutError("hall capture timed out after 13s"), "timed out"),
])
async def test_capture_failure_surfaces_via_status(tmp_path, exc, needle):
    # A capture that raises does NOT fail the start request (still 202) — the error
    # surfaces on the status endpoint as state "error" with the message, and the
    # result endpoint returns 404.
    async def cap(device, seconds):
        raise exc

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        started = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert started.status == 202

        st = await _poll_state(client, "hall", "error")
        assert st["state"] == "error"
        assert needle in st["error"]

        # No WAV to download for a failed capture.
        res = await client.get("/api/capture/result", params={"device": "hall"})
        assert res.status == 404
    finally:
        await client.close()


async def test_capture_empty_recording_surfaces_as_error(tmp_path):
    # An empty recording (CaptureEmptyError) is a capture failure too: it surfaces
    # via the status endpoint as state "error".
    from src.pipeline import CaptureEmptyError

    async def cap(device, seconds):
        raise CaptureEmptyError("capture produced no audio")

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        started = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert started.status == 202
        st = await _poll_state(client, "hall", "error")
        assert "no audio" in st["error"]
    finally:
        await client.close()


async def test_capture_bad_seconds_returns_400(tmp_path):
    called = []

    async def cap(device, seconds):
        called.append((device, seconds))

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        # Out of range, wrong type, missing device — all 400, capture never called.
        # The accepted range is 1..300, so 301 (and 0/negative) are rejected.
        for body in ({"device": "h", "seconds": 0},
                     {"device": "h", "seconds": -1},
                     {"device": "h", "seconds": 301},
                     {"device": "h", "seconds": "5"},
                     {"device": "h", "seconds": True},
                     {"device": "", "seconds": 5},
                     {"seconds": 5}):
            resp = await client.post("/api/capture", json=body)
            assert resp.status == 400, body
        assert called == []
    finally:
        await client.close()


async def test_capture_max_seconds_boundary(tmp_path):
    # The upper bound is 300 (CAPTURE_MAX_SECONDS): exactly 300 is accepted (202)
    # and passed through to the capture callable; 301 is rejected (400).
    calls = []

    async def cap(device, seconds):
        calls.append((device, seconds))
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        ok = await client.post("/api/capture", json={"device": "hall", "seconds": 300})
        assert ok.status == 202
        await _poll_state(client, "hall", "done")
        assert calls == [("hall", 300)]

        bad = await client.post("/api/capture", json={"device": "hall", "seconds": 301})
        assert bad.status == 400
    finally:
        await client.close()


async def test_capture_endpoints_without_manager_return_503(tmp_path):
    # No device_capture wired (e.g. tests / API-only boot) -> every capture endpoint
    # returns 503.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        assert (await client.post("/api/capture", json={"device": "h", "seconds": 5})).status == 503
        assert (await client.get("/api/capture", params={"device": "h"})).status == 503
        assert (await client.post("/api/capture/cancel", json={"device": "h"})).status == 503
        assert (await client.get("/api/capture/result", params={"device": "h"})).status == 503
    finally:
        await client.close()


async def test_capture_result_404_when_nothing_recorded(tmp_path):
    # No capture has run for the device -> result endpoint returns 404.
    async def cap(device, seconds):
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        resp = await client.get("/api/capture/result", params={"device": "hall"})
        assert resp.status == 404
        assert "no capture result" in (await resp.json())["error"]
    finally:
        await client.close()


async def test_capture_cancel_discards_result_and_frees_device(tmp_path):
    # Cancel flags an in-flight capture cancelled (the device self-times and keeps
    # draining); a new start stays blocked (409) until the task completes, then the
    # result is discarded and status returns to idle.
    release = asyncio.Event()

    async def cap(device, seconds):
        await release.wait()
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        started = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert started.status == 202
        await _poll_state(client, "hall", "recording")

        # Cancel -> state "cancelled" while the device is still draining.
        cancelled = await client.post("/api/capture/cancel", json={"device": "hall"})
        assert cancelled.status == 200
        assert (await cancelled.json())["state"] == "cancelled"

        # The device window has not elapsed -> a new start is still blocked.
        busy = await client.post("/api/capture", json={"device": "hall", "seconds": 5})
        assert busy.status == 409

        # Let the device finish; the arrived result is discarded and status goes idle.
        release.set()
        await _poll_state(client, "hall", "idle")

        # Nothing to download for a cancelled capture.
        res = await client.get("/api/capture/result", params={"device": "hall"})
        assert res.status == 404
    finally:
        release.set()
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
        assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await client.close()


async def test_cors_allowlist_is_hardcoded_empty_never_reflects_origin(tmp_path):
    # The CORS allowlist is hardcoded empty in panel_api (_ALLOWED_ORIGINS), so NO
    # cross-origin request is ever reflected — not even one that used to be allowlisted
    # via the removed core.panel.allowed_origins knob. Preflight still returns 204.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        # A cross-origin GET carrying an Origin header gets NO ACAO header reflected.
        resp = await client.get("/api/config", headers={"Origin": "http://localhost:5173"})
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers

        # An OPTIONS preflight (even with an Origin) still returns 204 and reflects nothing.
        pre = await client.options("/api/config", headers={"Origin": "http://localhost:5173"})
        assert pre.status == 204
        assert "Access-Control-Allow-Origin" not in pre.headers
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


async def test_get_run_audio_returns_wav_inline(tmp_path):
    # A run with stored audio is served as audio/wav with an inline disposition and
    # the exact stored bytes in the body.
    store = _seed_runs(tmp_path)
    wav = _wav_bytes()
    store.put_audio(1, wav, keep=100)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs/1/audio")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert resp.headers["Content-Disposition"] == \
            'inline; filename="zakhar_run_1.wav"'
        body = await resp.read()
        assert body == wav
    finally:
        await client.close()
        store.close()


async def test_get_run_audio_404_when_no_audio_and_400_on_bad_id(tmp_path):
    # A run without stored audio -> 404; a non-integer id -> 400.
    store = _seed_runs(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        missing = await client.get("/api/runs/1/audio")
        assert missing.status == 404

        bad = await client.get("/api/runs/abc/audio")
        assert bad.status == 400
    finally:
        await client.close()
        store.close()


async def test_get_run_audio_404_without_store(tmp_path):
    # No runs_store wired -> audio endpoint returns 404.
    client, _svc_, _ev = await _client(tmp_path)
    try:
        resp = await client.get("/api/runs/1/audio")
        assert resp.status == 404
    finally:
        await client.close()


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


# --- _patch_config plain-ValueError branch (distinct from ValidationError) ----

async def test_patch_config_value_error_returns_422_with_empty_detail(tmp_path):
    # apply() raises a *plain* ValueError (not a pydantic ValidationError) when a patch
    # selects an unknown provider — get_provider() raises ValueError("Unknown ..."). That
    # branch returns 422 with detail=[] and an "error" string, and must NOT persist.
    client, svc, _ev = await _client(tmp_path)
    try:
        before = svc.document()
        # A real patch whose merged doc fails inside apply() with a bare ValueError.
        resp = await client.patch("/api/config", json={"stt": {"selected": "nope"}})
        assert resp.status == 422
        body = await resp.json()
        # The plain-ValueError branch yields detail == [] (no pydantic error list).
        assert body["detail"] == []
        assert isinstance(body["error"], str) and body["error"]
        # Nothing was applied: the in-memory/on-disk document is unchanged.
        assert svc.document() == before
        assert svc.document()["stt"]["selected"] == "groq"
    finally:
        await client.close()


async def test_patch_config_value_error_via_monkeypatched_apply(tmp_path):
    # Deterministic variant: replace the bound svc.apply with one that raises a bare
    # ValueError, independent of any specific provider. Confirms the handler maps a plain
    # ValueError to 422/detail=[] and leaves the document untouched.
    client, svc, _ev = await _client(tmp_path)
    try:
        before = svc.document()

        def _boom(_patch):
            raise ValueError("bad provider")

        # Monkeypatch the bound method on the real svc instance the server holds.
        svc.apply = _boom

        resp = await client.patch("/api/config", json={"core": {"context": {"max_turns": 7}}})
        assert resp.status == 422
        body = await resp.json()
        assert body["detail"] == []
        assert body["error"] == "bad provider"
        # apply() raised before persisting -> document is unchanged.
        assert svc.document() == before
    finally:
        await client.close()


# --- _post_capture non-dict body guard ---------------------------------------

async def test_capture_non_object_body_returns_400_without_calling_capture(tmp_path):
    # A JSON array body (not an object) is rejected with 400 before the capture
    # callable is ever invoked.
    called = []

    async def cap(device, seconds):
        called.append((device, seconds))
        return _wav_bytes()

    client, _svc_, _ev = await _client(tmp_path, device_capture=cap)
    try:
        resp = await client.post("/api/capture", json=[1, 2])
        assert resp.status == 400
        body = await resp.json()
        assert "object" in body["error"]
        # The capture coroutine was never awaited.
        assert called == []
    finally:
        await client.close()


# --- _get_runs limit clamp + bad-limit fallback ------------------------------

class _LimitSpyStore:
    """Thin RunsStore wrapper that records the `limit` kwarg passed to .list()."""

    def __init__(self, inner):
        self._inner = inner
        self.limits = []

    def list(self, *, device=None, result=None, search=None, limit=100):
        self.limits.append(limit)
        return self._inner.list(device=device, result=result, search=search, limit=limit)

    # Delegate anything else the server might touch to the real store.
    def __getattr__(self, name):
        return getattr(self._inner, name)


async def test_get_runs_limit_is_clamped_and_falls_back(tmp_path):
    # The handler clamps limit into [1, 500] and falls back to 100 on a non-integer.
    # Assert the VALUE that reached the store, not merely the status code.
    inner = _seed_runs(tmp_path)
    spy = _LimitSpyStore(inner)
    client, _svc_, _ev = await _client(tmp_path, runs_store=spy)
    try:
        # Non-integer limit -> fallback 100.
        r1 = await client.get("/api/runs", params={"limit": "abc"})
        assert r1.status == 200
        assert spy.limits[-1] == 100

        # Below the floor -> clamped up to 1.
        r2 = await client.get("/api/runs", params={"limit": "0"})
        assert r2.status == 200
        assert spy.limits[-1] == 1

        # Above the ceiling -> clamped down to 500.
        r3 = await client.get("/api/runs", params={"limit": "99999"})
        assert r3.status == 200
        assert spy.limits[-1] == 500

        assert spy.limits == [100, 1, 500]
    finally:
        await client.close()
        inner.close()


# --- _get_run non-integer id on the plain route ------------------------------

async def test_get_run_bad_id_returns_400_invalid_id(tmp_path):
    # The plain /api/runs/{id} route (NOT the /audio variant) rejects a non-integer id.
    store = _seed_runs(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs/abc")
        assert resp.status == 400
        assert (await resp.json()) == {"error": "invalid id"}
    finally:
        await client.close()
        store.close()


# --- SPA static serving + (?!api/) catch-all guard ---------------------------

def _build_static_dir(tmp_path):
    """Create a minimal built-frontend dir: index.html + assets/<file>."""
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html>SPA INDEX</html>", encoding="utf-8")
    assets = static_dir / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('asset');", encoding="utf-8")
    return static_dir


async def test_spa_static_and_catch_all_does_not_swallow_api(tmp_path):
    # With a static_dir wired, the SPA index is served for "/" and deep links, real
    # assets are served from /assets/, and the (?!api/) negative lookahead must keep
    # /api/* OUT of the SPA index so unknown API paths still 404 (JSON router 404).
    static_dir = _build_static_dir(tmp_path)
    client, _svc_, _ev = await _client(tmp_path, static_dir=str(static_dir))
    try:
        # (a) Root serves index.html.
        root = await client.get("/")
        assert root.status == 200
        assert "SPA INDEX" in await root.text()

        # (b) A deep client-side route also serves index.html (client router takes over).
        deep = await client.get("/settings/x")
        assert deep.status == 200
        assert "SPA INDEX" in await deep.text()

        # (c) A real static asset is served from /assets/ (NOT the SPA index).
        asset = await client.get("/assets/app.js")
        assert asset.status == 200
        asset_body = await asset.text()
        assert "console.log('asset');" in asset_body
        assert "SPA INDEX" not in asset_body

        # (d) Load-bearing: the negative lookahead must NOT route /api/* into the SPA
        # index. An unknown API path 404s as JSON, never falling through to index.html.
        api = await client.get("/api/nope")
        assert api.status == 404
        api_body = await api.text()
        assert "SPA INDEX" not in api_body
    finally:
        await client.close()
