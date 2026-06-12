"""End-to-end tests for the admin-panel HTTP API.

Uses aiohttp's built-in TestServer/TestClient (no extra dependency). asyncio_mode=auto
is on, so async test functions run directly. Each test builds a ConfigService over a
tmp config file and drives a fresh PanelServer app.
"""

import asyncio
import json
import re
import time

import aiohttp
import httpx
import pytest
import respx

import src.plugins  # noqa: F401  register all providers
from aiohttp.test_utils import TestClient, TestServer
from src.config_service import ConfigService
from src.panel_api import PanelServer
from src.plugins.base import Deps
from src.prompt_store import PromptStore
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
    srv = PanelServer(svc, "127.0.0.1", 0, version="9.9", started_at=time.time(), **kw)
    client = TestClient(TestServer(srv.build_app()))
    await client.start_server()
    return client, svc


def _prompt_store(tmp_path, text="seed prompt"):
    """A PromptStore on tmp_path seeded from a legacy file holding `text`, so the
    'default' profile's text matches what _svc's config used to point at."""
    seed = tmp_path / "system_prompt.md"
    seed.write_text(text, encoding="utf-8")
    return PromptStore(str(tmp_path / "prompts.db"), seed_path=str(seed))


async def test_get_catalog(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/catalog")
        assert resp.status == 200
        body = await resp.json()
        assert "categories" in body
    finally:
        await client.close()


async def test_get_config(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/config")
        assert resp.status == 200
        body = await resp.json()
        assert all(k in body for k in ("stt", "llm", "tts", "core"))
    finally:
        await client.close()


async def test_patch_config_valid(tmp_path):
    client, svc = await _client(tmp_path)
    try:
        resp = await client.patch("/api/config",
                                  json={"core": {"context": {"max_turns": 9}}})
        assert resp.status == 200
        # GET reflects the applied change.
        got = await (await client.get("/api/config")).json()
        assert got["core"]["context"]["max_turns"] == 9
    finally:
        await client.close()


async def test_patch_config_invalid_returns_422(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        # vad.trim_start_ms has ge=0; -1 fails core validation -> ValidationError -> 422.
        resp = await client.patch("/api/config",
                                  json={"core": {"vad": {"trim_start_ms": -1}}})
        assert resp.status == 422
        body = await resp.json()
        assert "error" in body and "detail" in body
        assert isinstance(body["detail"], list)
    finally:
        await client.close()


async def test_patch_config_bad_json_returns_400(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.patch("/api/config", data="not json",
                                  headers={"Content-Type": "application/json"})
        assert resp.status == 400
    finally:
        await client.close()


async def test_patch_config_invalid_leaves_config_unchanged(tmp_path):
    client, svc = await _client(tmp_path)
    try:
        # trim_start_ms defaults to 200; an invalid patch must NOT persist anything.
        assert svc.document()["core"]["vad"]["trim_start_ms"] == 200
        resp = await client.patch("/api/config",
                                  json={"core": {"vad": {"trim_start_ms": -1}}})
        assert resp.status == 422
        # On-disk/in-memory config kept the old value.
        assert svc.document()["core"]["vad"]["trim_start_ms"] == 200
    finally:
        await client.close()


async def test_patch_config_non_object_body_returns_400(tmp_path):
    client, svc = await _client(tmp_path)
    try:
        for bad in ([1, 2, 3], "just a string"):
            resp = await client.patch("/api/config", json=bad)
            assert resp.status == 400
            assert "error" in await resp.json()
        # Nothing was applied: config document is untouched.
        assert svc.document()["core"]["vad"]["trim_start_ms"] == 200
    finally:
        await client.close()


async def test_get_options(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "yandex", "field": "voice"})
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body["options"], list)
        assert "zahar" in body["options"]
    finally:
        await client.close()


async def test_get_chimes_lists_bundled_files(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/chimes")
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body["options"], list)
        # The repo ships chime clips under assets/chimes; each option is a repo-root
        # relative path the pipeline opens verbatim.
        assert any(o.endswith(".wav") for o in body["options"])
        assert all(o.startswith("assets/chimes/") for o in body["options"])
    finally:
        await client.close()


@pytest.fixture
def _reset_openrouter_models_cache():
    """The OpenRouter model list is TTL-cached at module level; reset around each
    test so test order never matters."""
    from src.plugins.llm import openrouter as openrouter_mod
    openrouter_mod._models_cache.update({"at": 0.0, "data": None})
    yield
    openrouter_mod._models_cache.update({"at": 0.0, "data": None})


@respx.mock
async def test_get_options_async_provider_returns_fetched_list(
    tmp_path, _reset_openrouter_models_cache
):
    # llm/openrouter "model" options are network-backed: the provider returns a
    # coroutine that the handler must await before responding.
    from src.plugins.llm.openrouter import OPENROUTER_MODELS_URL
    respx.get(OPENROUTER_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a/x", "name": "X"}]})
    )
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "llm", "plugin": "openrouter", "field": "model"})
        assert resp.status == 200
        body = await resp.json()
        assert body["options"] == [{"value": "a/x", "label": "X"}]
    finally:
        await client.close()


@respx.mock
async def test_get_options_upstream_http_error_returns_502(
    tmp_path, _reset_openrouter_models_cache
):
    from src.plugins.llm.openrouter import OPENROUTER_MODELS_URL
    respx.get(OPENROUTER_MODELS_URL).mock(return_value=httpx.Response(500))
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "llm", "plugin": "openrouter", "field": "model"})
        assert resp.status == 502
        assert "upstream fetch failed" in (await resp.json())["error"]
    finally:
        await client.close()


@respx.mock
async def test_get_options_passes_search_query_to_provider(tmp_path):
    # The `q` query param must reach the provider as the server-side search
    # string: fishaudio's remote-searchable reference_id turns it into a single
    # title-filtered catalog GET.
    from src.plugins.tts.fishaudio import FISH_MODELS_URL
    route = respx.get(FISH_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"total": 1, "items": [
            {"_id": "v1", "title": "Anna", "languages": ["ru"]},
        ]}))
    client, _svc_ = await _client(tmp_path)
    try:
        # Give fishaudio an api_key first — without it the catalog is never hit.
        resp = await client.patch(
            "/api/config",
            json={"tts": {"instances": {"fishaudio": {"api_key": "fk-1"}}}})
        assert resp.status == 200
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "fishaudio",
                                        "field": "reference_id", "q": "anna"})
        assert resp.status == 200
        body = await resp.json()
        assert body["options"] == [{"value": "v1", "label": "Anna [ru]"}]
        assert route.call_count == 1
        assert route.calls.last.request.url.params["title"] == "anna"
    finally:
        await client.close()


async def test_options_unknown_field_returns_empty_list(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "yandex", "field": "nope"})
        assert resp.status == 200
        assert (await resp.json())["options"] == []
    finally:
        await client.close()


async def test_options_missing_params_returns_400(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options")
        assert resp.status == 400
        assert "error" in await resp.json()
    finally:
        await client.close()


async def test_options_unknown_plugin_returns_404(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/options",
                                params={"category": "tts", "plugin": "nope", "field": "voice"})
        assert resp.status == 404
        assert "error" in await resp.json()
    finally:
        await client.close()


# --- POST /api/tts/test (voice test with unsaved draft settings) --------------

@respx.mock
async def test_tts_test_synthesizes_with_draft_settings(tmp_path):
    # The endpoint builds an AD-HOC teratts backend from the request's settings
    # (NOT the stored config, which selects yandex) and streams back the upstream
    # audio bytes in the provider's native format.
    route = respx.get(url__regex=r"http://tts\.test/synthesize/.*").mock(
        return_value=httpx.Response(
            200, content=b"MP3-BYTES", headers={"Content-Type": "audio/mpeg"}
        )
    )
    client, svc = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "teratts",
            "settings": {"base_url": "http://tts.test"},
            "text": "Привет, проверка голоса",
        })
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/mpeg"
        assert await resp.read() == b"MP3-BYTES"
        assert route.call_count == 1
        # The stored config was never touched by the ad-hoc build.
        assert svc.document()["tts"]["selected"] == "yandex"
        assert "teratts" not in svc.document()["tts"]["instances"]
    finally:
        await client.close()


async def test_tts_test_unknown_provider_returns_404(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "nope", "settings": {}, "text": "hi",
        })
        assert resp.status == 404
        assert "error" in await resp.json()
    finally:
        await client.close()


async def test_tts_test_empty_text_returns_422(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        for bad in ("", "   "):
            resp = await client.post("/api/tts/test", json={
                "provider": "teratts", "settings": {"base_url": "http://x"}, "text": bad,
            })
            assert resp.status == 422
            assert "error" in await resp.json()
    finally:
        await client.close()


async def test_tts_test_too_long_text_returns_422(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "teratts", "settings": {"base_url": "http://x"}, "text": "а" * 501,
        })
        assert resp.status == 422
        assert (await resp.json())["error"] == "text too long (max 500 chars)"
    finally:
        await client.close()


async def test_tts_test_invalid_settings_returns_422(tmp_path):
    # The fishaudio backend refuses to build without an api_key (ValueError from
    # backend init) -> 422, not a 500.
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "fishaudio", "settings": {}, "text": "hi",
        })
        assert resp.status == 422
        assert "api_key" in (await resp.json())["error"]
    finally:
        await client.close()


@respx.mock
async def test_tts_test_upstream_failure_returns_502(tmp_path):
    respx.get(url__regex=r"http://tts\.test/synthesize/.*").mock(
        return_value=httpx.Response(500, text="boom")
    )
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "teratts",
            "settings": {"base_url": "http://tts.test"},
            "text": "проверка",
        })
        assert resp.status == 502
        assert "error" in await resp.json()
    finally:
        await client.close()


async def test_tts_test_non_object_body_returns_400(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        for bad in ([1, 2, 3], "just a string"):
            resp = await client.post("/api/tts/test", json=bad)
            assert resp.status == 400
            assert "error" in await resp.json()
    finally:
        await client.close()


async def test_tts_test_piper_missing_voice_path_returns_422(tmp_path):
    # Provider create() may touch the filesystem: the piper backend loads an ONNX
    # voice from voice_path (PiperVoice.load), which raises FileNotFoundError (an
    # OSError) for a nonexistent path. That's a config problem from the panel's
    # draft, so it must map to a 422 JSON error, not leak as a bare 500. No real
    # model is needed: the load fails on the missing file before anything heavy.
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/api/tts/test", json={
            "provider": "piper",
            "settings": {"voice_path": str(tmp_path / "no-such-voice.onnx")},
            "text": "привет",
        })
        assert resp.status == 422
        assert "error" in await resp.json()
    finally:
        await client.close()


@respx.mock
async def test_tts_test_text_at_500_char_cap_returns_200(tmp_path):
    # Boundary: the 500-char cap is inclusive and measured AFTER strip, so both
    # exactly-500 text and 500 significant chars padded with whitespace pass.
    respx.get(url__regex=r"http://tts\.test/synthesize/.*").mock(
        return_value=httpx.Response(
            200, content=b"MP3-BYTES", headers={"Content-Type": "audio/mpeg"}
        )
    )
    client, _svc_ = await _client(tmp_path)
    try:
        for text in ("а" * 500, "  \n" + "а" * 500 + "  \t"):
            resp = await client.post("/api/tts/test", json={
                "provider": "teratts",
                "settings": {"base_url": "http://tts.test"},
                "text": text,
            })
            assert resp.status == 200
            assert await resp.read() == b"MP3-BYTES"
    finally:
        await client.close()


async def test_prompt_round_trip(tmp_path):
    # Back-compat endpoints: GET /api/prompt returns the ACTIVE profile
    # {"id","name","text"} (no "path" anymore); PUT updates its text.
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.get("/api/prompt")
        assert resp.status == 200
        body = await resp.json()
        assert body["text"] == "seed prompt"
        assert body["name"] == "default"
        assert isinstance(body["id"], int)
        assert "path" not in body

        put = await client.put("/api/prompt", json={"text": "updated prompt"})
        assert put.status == 200
        assert (await put.json()) == {"ok": True}

        # Re-read reflects the new text, and the store agrees.
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "updated prompt"
        assert store.active_text() == "updated prompt"
    finally:
        await client.close()
        store.close()


async def test_put_prompt_non_object_returns_400(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.put("/api/prompt", json="x")
        assert resp.status == 400
        assert "error" in await resp.json()
        # The stored prompt was NOT clobbered.
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "seed prompt"
    finally:
        await client.close()
        store.close()


async def test_put_prompt_missing_text_returns_400(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.put("/api/prompt", json={})
        assert resp.status == 400
        assert "error" in await resp.json()
        again = await (await client.get("/api/prompt")).json()
        assert again["text"] == "seed prompt"
    finally:
        await client.close()
        store.close()


# --- prompt profiles (CRUD + activate) -----------------------------------------

async def test_prompt_endpoints_without_store_return_503(tmp_path):
    # No prompt_store wired -> every /api/prompt* endpoint returns 503, like the
    # other optional subsystems.
    client, _svc_ = await _client(tmp_path)
    try:
        assert (await client.get("/api/prompt")).status == 503
        assert (await client.put("/api/prompt", json={"text": "x"})).status == 503
        assert (await client.get("/api/prompt/profiles")).status == 503
        assert (await client.post("/api/prompt/profiles", json={"name": "x"})).status == 503
        assert (await client.get("/api/prompt/profiles/1")).status == 503
        assert (await client.put("/api/prompt/profiles/1", json={"text": "x"})).status == 503
        assert (await client.delete("/api/prompt/profiles/1")).status == 503
        assert (await client.post("/api/prompt/profiles/1/activate")).status == 503
    finally:
        await client.close()


async def test_prompt_profiles_list(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.get("/api/prompt/profiles")
        assert resp.status == 200
        body = await resp.json()
        assert [p["name"] for p in body["profiles"]] == ["default"]
        prof = body["profiles"][0]
        assert prof["is_active"] is True
        assert prof["chars"] == len("seed prompt")
        assert "text" not in prof  # summaries never carry the full text
        assert body["active_id"] == prof["id"]
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_create_with_text_and_get(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.post("/api/prompt/profiles",
                                 json={"name": "work", "text": "work prompt"})
        assert resp.status == 200
        created = await resp.json()
        assert created["name"] == "work"
        assert created["text"] == "work prompt"
        assert created["is_active"] is False

        got = await client.get(f"/api/prompt/profiles/{created['id']}")
        assert got.status == 200
        assert (await got.json()) == created
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_create_without_text_copies_active(tmp_path):
    # Omitting "text" duplicates the CURRENT ACTIVE profile's text.
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        resp = await client.post("/api/prompt/profiles", json={"name": "copy"})
        assert resp.status == 200
        created = await resp.json()
        assert created["text"] == "seed prompt"
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_create_validation_and_duplicate(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        # Bad bodies: missing/empty/non-string name, non-string text, non-object.
        for body in ({}, {"name": ""}, {"name": "  "}, {"name": 5},
                     {"name": "x", "text": 5}):
            resp = await client.post("/api/prompt/profiles", json=body)
            assert resp.status == 400, body
        assert (await client.post("/api/prompt/profiles", json=[1])).status == 400

        # Duplicate name -> 409 with an error message.
        dup = await client.post("/api/prompt/profiles", json={"name": "default"})
        assert dup.status == 409
        assert "error" in await dup.json()
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_update_rename_and_text(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles", json={"name": "work", "text": "v1"})).json()
        pid = created["id"]

        resp = await client.put(f"/api/prompt/profiles/{pid}",
                                json={"name": "job", "text": "v2"})
        assert resp.status == 200
        body = await resp.json()
        assert body["name"] == "job" and body["text"] == "v2"

        # 400: empty body / wrong types; 404 unknown id; 409 duplicate name.
        assert (await client.put(f"/api/prompt/profiles/{pid}", json={})).status == 400
        assert (await client.put(f"/api/prompt/profiles/{pid}",
                                 json={"name": ""})).status == 400
        assert (await client.put(f"/api/prompt/profiles/{pid}",
                                 json={"text": 5})).status == 400
        assert (await client.put("/api/prompt/profiles/9999",
                                 json={"text": "x"})).status == 404
        assert (await client.put(f"/api/prompt/profiles/{pid}",
                                 json={"name": "default"})).status == 409
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_delete(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles", json={"name": "tmp"})).json()
        resp = await client.delete(f"/api/prompt/profiles/{created['id']}")
        assert resp.status == 200
        assert (await resp.json()) == {"ok": True}

        # Already gone -> 404; the ACTIVE profile is refused -> 409.
        assert (await client.delete(f"/api/prompt/profiles/{created['id']}")).status == 404
        active = await (await client.get("/api/prompt")).json()
        refused = await client.delete(f"/api/prompt/profiles/{active['id']}")
        assert refused.status == 409
        assert "active" in (await refused.json())["error"]
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_activate_switches_what_get_prompt_returns(tmp_path):
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles", json={"name": "work", "text": "WORK"})).json()
        pid = created["id"]

        resp = await client.post(f"/api/prompt/profiles/{pid}/activate")
        assert resp.status == 200
        assert (await resp.json()) == {"ok": True, "active_id": pid}

        # The list reflects the switch and GET /api/prompt now serves the new text.
        listing = await (await client.get("/api/prompt/profiles")).json()
        assert listing["active_id"] == pid
        assert sum(1 for p in listing["profiles"] if p["is_active"]) == 1
        active = await (await client.get("/api/prompt")).json()
        assert active["id"] == pid and active["text"] == "WORK"

        # Unknown id -> 404.
        assert (await client.post("/api/prompt/profiles/9999/activate")).status == 404
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_activate_apply_voice_sets_tts(tmp_path):
    # Activating with apply_voice=true reads the profile's preferred-voice marker
    # and applies it to the TTS config. The test ConfigService registers no
    # reconfig callbacks, so apply() only validates+persists — selecting piper does
    # NOT load an ONNX file.
    store = _prompt_store(tmp_path)
    client, svc = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles",
            json={"name": "voiced",
                  "text": "BODY <<<<<VOICE provider=piper voice_path=models/x.onnx>>>>>"},
        )).json()
        pid = created["id"]

        resp = await client.post(f"/api/prompt/profiles/{pid}/activate",
                                 json={"apply_voice": True})
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True and body["active_id"] == pid
        assert body["voice_applied"] == {
            "provider": "piper", "fields": {"voice_path": "models/x.onnx"}
        }

        doc = svc.document()
        assert doc["tts"]["selected"] == "piper"
        assert doc["tts"]["instances"]["piper"]["voice_path"] == "models/x.onnx"
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_activate_apply_voice_no_marker(tmp_path):
    # apply_voice=true but no marker -> voice_applied is null and TTS is unchanged.
    store = _prompt_store(tmp_path)
    client, svc = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles", json={"name": "plain", "text": "no marker here"})).json()
        pid = created["id"]

        resp = await client.post(f"/api/prompt/profiles/{pid}/activate",
                                 json={"apply_voice": True})
        assert resp.status == 200
        body = await resp.json()
        assert body["voice_applied"] is None
        assert svc.document()["tts"]["selected"] == "yandex"
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_activate_apply_voice_unknown_provider(tmp_path):
    # An unknown provider in the marker -> voice_error, but activation still
    # succeeded (active_id switched) and the TTS config is untouched.
    store = _prompt_store(tmp_path)
    client, svc = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles",
            json={"name": "bad", "text": "<<<<<VOICE provider=nope voice=x>>>>>"})).json()
        pid = created["id"]

        resp = await client.post(f"/api/prompt/profiles/{pid}/activate",
                                 json={"apply_voice": True})
        assert resp.status == 200
        body = await resp.json()
        assert body["active_id"] == pid
        assert "voice_error" in body and "nope" in body["voice_error"]
        assert svc.document()["tts"]["selected"] == "yandex"
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_activate_no_body_back_compat(tmp_path):
    # Legacy callers POST with no body: still returns {"ok", "active_id"} and never
    # touches TTS (no voice_applied / voice_error keys).
    store = _prompt_store(tmp_path)
    client, svc = await _client(tmp_path, prompt_store=store)
    try:
        created = await (await client.post(
            "/api/prompt/profiles",
            json={"name": "voiced", "text": "<<<<<VOICE provider=piper voice_path=x.onnx>>>>>"})).json()
        pid = created["id"]

        resp = await client.post(f"/api/prompt/profiles/{pid}/activate")
        assert resp.status == 200
        assert (await resp.json()) == {"ok": True, "active_id": pid}
        assert svc.document()["tts"]["selected"] == "yandex"
    finally:
        await client.close()
        store.close()


async def test_prompt_profile_non_numeric_id_returns_400(tmp_path):
    # Non-numeric {id} segments are rejected with 400 "invalid id" consistently.
    store = _prompt_store(tmp_path)
    client, _svc_ = await _client(tmp_path, prompt_store=store)
    try:
        assert (await client.get("/api/prompt/profiles/abc")).status == 400
        assert (await client.put("/api/prompt/profiles/abc",
                                 json={"text": "x"})).status == 400
        assert (await client.delete("/api/prompt/profiles/abc")).status == 400
        assert (await client.post("/api/prompt/profiles/abc/activate")).status == 400
    finally:
        await client.close()
        store.close()


async def test_cors_header_present_on_error_response(tmp_path):
    client, _svc_ = await _client(tmp_path)
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
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/system")
        assert resp.status == 200
        body = await resp.json()
        assert body["version"] == "9.9"
        assert body["running"] is True
        assert body["log_level"] == "DEBUG"
        assert "started" in body
        assert isinstance(body["uptime_seconds"], int)
        # Without a wired device_status the snapshot carries an empty device list.
        assert body["devices"] == []
    finally:
        await client.close()


async def test_get_system_includes_device_statuses(tmp_path):
    devices = [{"name": "x", "host": "y", "enabled": True, "online": True, "versions": []}]
    client, _svc_ = await _client(tmp_path, device_status=lambda: devices)
    try:
        resp = await client.get("/api/system")
        assert resp.status == 200
        body = await resp.json()
        # The snapshot embeds the live device_status() result verbatim.
        assert body["devices"] == devices
    finally:
        await client.close()


async def test_get_devices(tmp_path):
    devices = [{"name": "x", "host": "y", "online": True}]
    client, _svc_ = await _client(tmp_path, device_status=lambda: devices)
    try:
        resp = await client.get("/api/devices")
        assert resp.status == 200
        assert (await resp.json()) == devices
    finally:
        await client.close()


async def test_get_devices_default_empty(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/devices")
        assert resp.status == 200
        assert (await resp.json()) == []
    finally:
        await client.close()


def _wav_bytes(pcm=b"\x01\x02" * 16, channels=1):
    """Build a tiny valid 16k/16-bit WAV (mono by default) for audio assertions."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_params_and_pcm(body):
    """Parse WAV bytes -> ((nchannels, sampwidth, framerate), raw PCM frames)."""
    import io
    import wave
    with wave.open(io.BytesIO(body)) as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate()), \
            w.readframes(w.getnframes())


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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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
    client, _svc_ = await _client(tmp_path)
    try:
        assert (await client.post("/api/capture", json={"device": "h", "seconds": 5})).status == 503
        assert (await client.get("/api/capture", params={"device": "h"})).status == 503
        assert (await client.get("/api/capture/result", params={"device": "h"})).status == 503
    finally:
        await client.close()


async def test_capture_result_404_when_nothing_recorded(tmp_path):
    # No capture has run for the device -> result endpoint returns 404.
    async def cap(device, seconds):
        return _wav_bytes()

    client, _svc_ = await _client(tmp_path, device_capture=cap)
    try:
        resp = await client.get("/api/capture/result", params={"device": "hall"})
        assert resp.status == 404
        assert "no capture result" in (await resp.json())["error"]
    finally:
        await client.close()


async def test_get_tools(tmp_path):
    # tool_sources is a zero-arg callable returning ToolHub.describe() output.
    sources = [{
        "id": "home", "kind": "http", "online": True,
        "tools": [{"name": "light.set", "description": "x"}],
    }]
    client, _svc_ = await _client(tmp_path, tool_sources=lambda: sources)
    try:
        resp = await client.get("/api/tools")
        assert resp.status == 200
        assert (await resp.json()) == {"sources": sources}
    finally:
        await client.close()


async def test_get_tools_default_empty(tmp_path):
    # No tool_sources -> empty list.
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/tools")
        assert resp.status == 200
        assert (await resp.json()) == {"sources": []}
    finally:
        await client.close()


async def test_cors_preflight(tmp_path):
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.options("/api/config")
        assert resp.status == 204
        assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await client.close()


async def test_cors_allowlist_is_hardcoded_empty_never_reflects_origin(tmp_path):
    # The CORS allowlist is hardcoded empty in panel_api (_ALLOWED_ORIGINS), so NO
    # cross-origin request is ever reflected — it is not a config knob. Preflight
    # still returns 204.
    client, _svc_ = await _client(tmp_path)
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
        "t_vad": 1000, "t_stt": 200, "t_llm": 300, "t_stress": 0, "t_tts": 100,
        "t_total": 1600, "audio_ms": None, "audio_bytes": 500, "audio_fmt": "mp3",
        "error_stage": None, "error_text": None,
        "rounds": [{"round": 1, "note": "final answer", "tokens": 10, "calls": []}],
    })
    store.insert({
        "ts": now, "device": "bedroom", "result": "error", "reason": "endpoint",
        # error_text carries the RAW stage error (StageError message), llm_text the
        # spoken fallback phrase — mirrors what pipeline records since the R1 contract.
        "stt_text": "сломайся", "llm_text": "Что-то сломалось, попробуй ещё раз попозже.",
        "model": None, "tokens": None,
        "t_vad": 1000, "t_stt": 200, "t_llm": 300, "t_stress": 0, "t_tts": 0,
        "t_total": 1500, "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
        "error_stage": "LLM", "error_text": "boom", "rounds": [],
    })
    return store


async def test_get_runs_list_and_filters(tmp_path):
    store = _seed_runs(tmp_path)
    client, _svc_ = await _client(tmp_path, runs_store=store)
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
    client, _svc_ = await _client(tmp_path, runs_store=store)
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
    client, _svc_ = await _client(tmp_path, runs_store=store)
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


async def test_get_run_audio_channel_split_on_stereo(tmp_path):
    # A stored stereo WAV: ?channel=stt serves channel 0 (left) and ?channel=raw
    # channel 1 (right), each as a standalone mono WAV; no param stays the
    # byte-identical full stereo file (the Download path).
    store = _seed_runs(tmp_path)
    left = b"\x01\x00" * 16   # int16 samples 1
    right = b"\x02\x00" * 16  # int16 samples 2
    interleaved = b"".join(left[i:i + 2] + right[i:i + 2] for i in range(0, len(left), 2))
    stereo = _wav_bytes(pcm=interleaved, channels=2)
    store.put_audio(1, stereo, keep=100)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        # Left = STT channel.
        resp = await client.get("/api/runs/1/audio", params={"channel": "stt"})
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert resp.headers["Content-Disposition"] == \
            'inline; filename="zakhar_run_1_stt.wav"'
        params, pcm = _wav_params_and_pcm(await resp.read())
        assert params == (1, 2, 16000)
        assert pcm == left

        # Right = raw channel.
        resp = await client.get("/api/runs/1/audio", params={"channel": "raw"})
        assert resp.status == 200
        assert resp.headers["Content-Disposition"] == \
            'inline; filename="zakhar_run_1_raw.wav"'
        params, pcm = _wav_params_and_pcm(await resp.read())
        assert params == (1, 2, 16000)
        assert pcm == right

        # No param -> the exact stored stereo bytes.
        resp = await client.get("/api/runs/1/audio")
        assert resp.status == 200
        assert await resp.read() == stereo

        # Run detail exposes the channel count for the panel.
        detail = await (await client.get("/api/runs/1")).json()
        assert detail["audio_channels"] == 2
    finally:
        await client.close()
        store.close()


async def test_get_run_audio_channel_on_mono_and_bad_channel(tmp_path):
    # On a MONO stored WAV ?channel=raw is a 404 (no such channel) while the
    # detail payload reports audio_channels == 1; an unknown channel name is a
    # 400; a run without audio reports audio_channels None.
    store = _seed_runs(tmp_path)
    mono = _wav_bytes()
    store.put_audio(1, mono, keep=100)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs/1/audio", params={"channel": "raw"})
        assert resp.status == 404
        assert (await resp.json()) == {"error": "channel not available"}

        # channel=stt on mono serves the original mono bytes.
        resp = await client.get("/api/runs/1/audio", params={"channel": "stt"})
        assert resp.status == 200
        assert await resp.read() == mono

        resp = await client.get("/api/runs/1/audio", params={"channel": "bogus"})
        assert resp.status == 400
        assert (await resp.json()) == {"error": 'channel must be "stt" or "raw"'}

        detail = await (await client.get("/api/runs/1")).json()
        assert detail["audio_channels"] == 1
        # Run 2 has no stored audio -> null channel count.
        detail2 = await (await client.get("/api/runs/2")).json()
        assert detail2["audio_channels"] is None
    finally:
        await client.close()
        store.close()


async def test_get_run_audio_404_when_no_audio_and_400_on_bad_id(tmp_path):
    # A run without stored audio -> 404; a non-integer id -> 400.
    store = _seed_runs(tmp_path)
    client, _svc_ = await _client(tmp_path, runs_store=store)
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
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/runs/1/audio")
        assert resp.status == 404
    finally:
        await client.close()


async def test_get_run_tts_audio_returns_native_inline(tmp_path):
    # A run with stored TTS audio is served with its native mime as Content-Type,
    # an inline disposition (extension derived from the mime), and the exact bytes.
    store = _seed_runs(tmp_path)
    store.put_tts_audio(1, b"MP3", "audio/mpeg", keep=100)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs/1/tts-audio")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/mpeg"
        assert resp.headers["Content-Disposition"] == \
            'inline; filename="zakhar_run_1_tts.mp3"'
        body = await resp.read()
        assert body == b"MP3"
    finally:
        await client.close()
        store.close()


async def test_get_run_tts_audio_wav_mime_extension(tmp_path):
    # An audio/wav blob picks the .wav download extension from the stored mime.
    store = _seed_runs(tmp_path)
    store.put_tts_audio(1, b"WAVE", "audio/wav", keep=100)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/runs/1/tts-audio")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "audio/wav"
        assert resp.headers["Content-Disposition"] == \
            'inline; filename="zakhar_run_1_tts.wav"'
        body = await resp.read()
        assert body == b"WAVE"
    finally:
        await client.close()
        store.close()


async def test_get_run_tts_audio_404_when_none_and_400_on_bad_id(tmp_path):
    # A run without stored TTS audio -> 404; a non-integer id -> 400.
    store = _seed_runs(tmp_path)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        missing = await client.get("/api/runs/1/tts-audio")
        assert missing.status == 404

        bad = await client.get("/api/runs/abc/tts-audio")
        assert bad.status == 400
    finally:
        await client.close()
        store.close()


async def test_get_run_tts_audio_404_without_store(tmp_path):
    # No runs_store wired -> tts-audio endpoint returns 404.
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.get("/api/runs/1/tts-audio")
        assert resp.status == 404
    finally:
        await client.close()


async def test_get_metrics(tmp_path):
    store = _seed_runs(tmp_path)
    client, _svc_ = await _client(tmp_path, runs_store=store)
    try:
        resp = await client.get("/api/metrics")
        assert resp.status == 200
        body = await resp.json()
        assert body["requests_24h"] == 2
        assert body["error_rate"] == 0.5
        assert "p50_ms" in body and "p95_ms" in body
        assert set(body["per_stage_avg_ms"]) == {"vad", "stt", "llm", "stress", "tts"}
    finally:
        await client.close()
        store.close()


async def test_runs_endpoints_empty_without_store(tmp_path):
    # No runs_store -> list returns empty, metrics returns zeros, get id is 404.
    client, _svc_ = await _client(tmp_path)
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
    client, _svc_ = await _client(tmp_path, run_events=hub)
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
        # The stream also carries periodic {"type":"system",...} heartbeats; skip them.
        while True:
            msg = await asyncio.wait_for(ws.receive_json(), 2)
            if msg.get("type") == "run":
                break
        assert msg == payload

        await ws.close()
    finally:
        await client.close()


async def test_runs_stream_pushes_system_heartbeat(tmp_path):
    hub = RunEventsHub()
    client, _svc_ = await _client(tmp_path, run_events=hub, heartbeat_interval=0.05)
    try:
        ws = await client.ws_connect("/api/runs/stream")
        while True:
            msg = await asyncio.wait_for(ws.receive_json(), 2)
            if msg.get("type") == "system":
                break
        assert msg["running"] is True
        assert isinstance(msg["uptime_seconds"], int)
        assert "db_size_bytes" not in msg  # heartbeat stays lightweight
        # No device_status wired -> the heartbeat still carries an (empty) device list.
        assert msg["devices"] == []
        await ws.close()
    finally:
        await client.close()


async def test_runs_stream_heartbeat_carries_device_statuses(tmp_path):
    # The 1 s heartbeat is the panel's live device-status feed: it must embed the
    # device_status() snapshot so the Devices page updates without polling.
    hub = RunEventsHub()
    devices = [{"name": "x", "host": "y", "enabled": True, "online": False, "versions": []}]
    client, _svc_ = await _client(
        tmp_path, run_events=hub, heartbeat_interval=0.05,
        device_status=lambda: devices,
    )
    try:
        ws = await client.ws_connect("/api/runs/stream")
        while True:
            msg = await asyncio.wait_for(ws.receive_json(), 2)
            if msg.get("type") == "system":
                break
        assert msg["devices"] == devices
        await ws.close()
    finally:
        await client.close()


async def test_runs_stream_without_hub_closes_promptly(tmp_path):
    # No run_events hub on the server -> the handler closes the socket immediately.
    client, _svc_ = await _client(tmp_path)
    try:
        ws = await client.ws_connect("/api/runs/stream")
        # The first receive should observe the close from the server side.
        msg = await asyncio.wait_for(ws.receive(), 2)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING)
        await ws.close()
    finally:
        await client.close()


async def test_runs_stream_handshake_disconnect_is_quiet(tmp_path, monkeypatch):
    # A client that drops DURING the WS handshake makes ws.prepare() raise
    # ClientConnectionResetError. The handler must swallow it (debug log, no
    # traceback), never register the socket, and leave the response in a state
    # where aiohttp's real finish_response (prepare() + write_eof()) is a clean
    # no-op instead of raising a second time ("Call .prepare() first").
    from aiohttp.client_exceptions import ClientConnectionResetError
    from src import panel_api

    hub = RunEventsHub()
    srv = PanelServer(_svc(tmp_path), "127.0.0.1", 0, version="9.9",
                      started_at=time.time(), run_events=hub)

    # Subclass the REAL WebSocketResponse so write_eof()/close() are genuine aiohttp
    # code, only overriding prepare() to reproduce the failure: aiohttp's
    # StreamResponse._start assigns _payload_writer BEFORE writing the 101 headers,
    # so when the transport write fails _payload_writer is already set while _writer
    # (assigned later in _post_start) is not.
    class _HandshakeBoomWS(aiohttp.web.WebSocketResponse):
        async def prepare(self, request):
            self._payload_writer = object()
            raise ClientConnectionResetError("Cannot write to closing transport")

    monkeypatch.setattr(panel_api.web, "WebSocketResponse", _HandshakeBoomWS)

    # request is unused on the handshake-failure path; a bare object() is enough.
    ws = await srv._runs_stream(object())

    assert hub.count() == 0        # the socket was never registered
    assert ws._eof_sent is True    # response marked finalized

    # The crux of the regression: aiohttp's real write_eof() must now be a clean
    # no-op. Without the fix (ws._eof_sent left False) this call raises
    # RuntimeError("Call .prepare() first").
    await ws.write_eof()


# --- _patch_config plain-ValueError branch (distinct from ValidationError) ----

async def test_patch_config_value_error_returns_422_with_empty_detail(tmp_path):
    # apply() raises a *plain* ValueError (not a pydantic ValidationError) when a patch
    # selects an unknown provider — get_provider() raises ValueError("Unknown ..."). That
    # branch returns 422 with detail=[] and an "error" string, and must NOT persist.
    client, svc = await _client(tmp_path)
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
    client, svc = await _client(tmp_path)
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

    client, _svc_ = await _client(tmp_path, device_capture=cap)
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
    client, _svc_ = await _client(tmp_path, runs_store=spy)
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
    client, _svc_ = await _client(tmp_path, runs_store=store)
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
    client, _svc_ = await _client(tmp_path, static_dir=str(static_dir))
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


# --- /mcp (agent MCP endpoint bridged into the panel) -------------------------

# MCP streamable-HTTP requests must accept both JSON and SSE; in stateless+json
# mode the server answers with plain JSON.
_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _mcp_initialize():
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26", "capabilities": {},
            "clientInfo": {"name": "test-agent", "version": "0"},
        },
    }


async def _mcp_client(tmp_path, **kw):
    """Panel test client with a STARTED AgentMcpEndpoint bridged at /mcp.

    The endpoint's Runtime stub shares the panel's ConfigService, so the
    get_config tool returns the same document the panel serves."""
    from types import SimpleNamespace

    from src.agent_mcp import AgentMcpEndpoint

    svc = _svc(tmp_path)
    rt = SimpleNamespace(
        svc=svc, runs_store=None,
        manager=SimpleNamespace(clients=[], statuses=lambda: []),
    )
    endpoint = AgentMcpEndpoint(rt)
    await endpoint.start()
    srv = PanelServer(svc, "127.0.0.1", 0, version="9.9", started_at=time.time(),
                      agent_mcp=endpoint, **kw)
    client = TestClient(TestServer(srv.build_app()))
    await client.start_server()
    return client, svc, endpoint


async def test_mcp_post_initialize_round_trip(tmp_path):
    # A JSON-RPC initialize POST through the aiohttp bridge answers 200 with a
    # plain JSON body (stateless+json mode) carrying serverInfo.
    client, _svc_, endpoint = await _mcp_client(tmp_path)
    try:
        resp = await client.post("/mcp", json=_mcp_initialize(), headers=_MCP_HEADERS)
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("application/json")
        body = await resp.json()
        assert body["result"]["serverInfo"]["name"] == "zakhar-voice-assistant"
    finally:
        await client.close()
        await endpoint.stop()


async def test_mcp_disabled_via_config_returns_403(tmp_path):
    # core.agent_mcp.enabled is read LIVE per request: flipping it off makes the
    # very next /mcp request 403 — no rebuild, no restart.
    client, svc, endpoint = await _mcp_client(tmp_path)
    try:
        svc.apply({"core": {"agent_mcp": {"enabled": False}}})
        resp = await client.post("/mcp", json=_mcp_initialize(), headers=_MCP_HEADERS)
        assert resp.status == 403
        assert "disabled" in (await resp.json())["error"]

        # Flipping it back on restores service on the next request.
        svc.apply({"core": {"agent_mcp": {"enabled": True}}})
        resp = await client.post("/mcp", json=_mcp_initialize(), headers=_MCP_HEADERS)
        assert resp.status == 200
    finally:
        await client.close()
        await endpoint.stop()


async def test_mcp_unavailable_returns_503(tmp_path):
    # No AgentMcpEndpoint wired (boot failure / API-only) -> 503, like runs_store=None.
    client, _svc_ = await _client(tmp_path)
    try:
        resp = await client.post("/mcp", json=_mcp_initialize(), headers=_MCP_HEADERS)
        assert resp.status == 503
        assert (await resp.json()) == {"error": "agent MCP endpoint not available"}
    finally:
        await client.close()


async def test_mcp_get_is_not_swallowed_by_spa(tmp_path):
    # With the SPA catch-all mounted, /mcp must always reach the MCP bridge (or a
    # plain router 405), never fall through to index.html.
    static_dir = _build_static_dir(tmp_path)
    client, _svc_, endpoint = await _mcp_client(tmp_path, static_dir=str(static_dir))
    try:
        # GET without an SSE accept reaches the MCP server, which answers a
        # JSON-RPC "Not Acceptable" error — proving the bridge handled it.
        resp = await client.get("/mcp")
        body = await resp.text()
        assert resp.status == 406
        assert resp.headers["Content-Type"].startswith("application/json")
        assert "SPA INDEX" not in body

        # An unregistered method on /mcp (PUT) must also stay out of the SPA:
        # the (?!api/|mcp$) lookahead plus route order give a plain 405.
        resp = await client.put("/mcp")
        body = await resp.text()
        assert resp.status == 405
        assert "SPA INDEX" not in body
    finally:
        await client.close()
        await endpoint.stop()


async def test_mcp_tools_call_round_trip(tmp_path):
    # Full stateless round-trip: initialize -> notifications/initialized (202) ->
    # tools/call get_config returns the panel's own config document.
    client, svc, endpoint = await _mcp_client(tmp_path)
    try:
        resp = await client.post("/mcp", json=_mcp_initialize(), headers=_MCP_HEADERS)
        assert resp.status == 200

        resp = await client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_MCP_HEADERS,
        )
        assert resp.status == 202

        resp = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "get_config", "arguments": {}}},
            headers=_MCP_HEADERS,
        )
        assert resp.status == 200
        body = await resp.json()
        result = body["result"]
        assert result.get("isError") in (False, None)
        # The tool's dict payload comes back as JSON text content; it must equal
        # the very document the panel itself serves.
        assert json.loads(result["content"][0]["text"]) == svc.document()
    finally:
        await client.close()
        await endpoint.stop()
