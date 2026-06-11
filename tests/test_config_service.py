import httpx
import pytest
from pydantic import ValidationError

import src.plugins  # noqa: F401  register all providers
from src import config_store
from src.config_service import ConfigService
from src.core_config import CoreConfig
from src.plugins.base import Deps


def _deps():
    return Deps(
        http_cloud=httpx.AsyncClient(),
        http_local=httpx.AsyncClient(),
        tts_timeout=30,
    )


def _doc():
    return {
        "version": 1,
        "stt": {"selected": "groq", "instances": {"groq": {"api_key": "gsk-secret"}}},
        "llm": {
            "selected": "openrouter",
            "instances": {"openrouter": {"api_key": "sk-or-secret", "model": "m1"}},
        },
        "tts": {
            "selected": "yandex",
            "instances": {"yandex": {"api_key": "AQVN-secret", "voice": "jane", "speed": 1.2}},
        },
        "core": {"openweathermap": {"api_key": "weather-secret", "city": "Москва"}},
    }


def _service(tmp_path):
    return ConfigService(_doc(), _deps(), path=str(tmp_path / "config.json"))


def test_get_returns_validated_selected_config(tmp_path):
    svc = _service(tmp_path)
    tts_cfg = svc.get("tts")
    assert tts_cfg.__class__.__name__ == "YandexTtsConfig"
    assert tts_cfg.voice == "jane"
    assert tts_cfg.speed == 1.2


def test_create_builds_backend(tmp_path):
    svc = _service(tmp_path)
    backend = svc.create("tts")
    assert backend.__class__.__name__ == "YandexTtsBackend"


def test_core_property_is_core_config(tmp_path):
    svc = _service(tmp_path)
    assert isinstance(svc.core, CoreConfig)
    assert svc.core.openweathermap.city == "Москва"


def test_constructor_validates_selected_instances(tmp_path):
    bad = _doc()
    bad["tts"]["instances"]["yandex"]["speed"] = 9.9  # out of range
    with pytest.raises(ValidationError):
        ConfigService(bad, _deps(), path=str(tmp_path / "config.json"))


def test_catalog_lists_all_categories_with_schemas(tmp_path):
    cat = _service(tmp_path).catalog()
    ids = {c["id"] for c in cat["categories"]}
    assert ids == {"vad", "stt", "llm", "tts"}
    for c in cat["categories"]:
        for prov in c["providers"]:
            assert "schema" in prov and "properties" in prov["schema"]
            assert "values" in prov
    assert "schema" in cat["core"] and "values" in cat["core"]


def test_old_doc_without_vad_slot_gets_webrtc_defaults(tmp_path):
    # Backward compat: a config written before the VAD stage existed (no "vad" key —
    # see _doc()) must still parse, defaulting the slot to the webrtc provider with
    # its own field defaults. Unlike stt/llm/tts, the vad slot has a default.
    svc = _service(tmp_path)
    vad_cfg = svc.get("vad")
    assert vad_cfg.__class__.__name__ == "WebRtcVadConfig"
    assert vad_cfg.aggressiveness == 2
    assert vad_cfg.auto_gain is False
    backend = svc.create("vad")
    assert backend.__class__.__name__ == "WebRtcVadBackend"
    # The catalog exposes the vad category with the webrtc provider selected.
    cat = svc.catalog()
    vad = next(c for c in cat["categories"] if c["id"] == "vad")
    assert vad["selected"] == "webrtc"
    webrtc = next(p for p in vad["providers"] if p["id"] == "webrtc")
    # vad provider fields are stage-instance paths -> rebuild_backends apply class.
    assert webrtc["schema"]["properties"]["aggressiveness"]["apply"] == "rebuild_backends"
    assert webrtc["schema"]["properties"]["auto_gain"]["apply"] == "rebuild_backends"


def test_catalog_exposes_values_plainly(tmp_path):
    # Secrets are plain config values now: catalog() emits them verbatim (no masking).
    doc = _doc()
    doc["core"]["openweathermap"]["api_key"] = "wkey"
    doc["core"]["devices"] = [
        {"name": "kitchen", "host": "10.0.0.5", "psk": "psk-plaintext-1"},
    ]
    cat = ConfigService(doc, _deps(), path=str(tmp_path / "config.json")).catalog()

    # Provider api_key comes through as plaintext.
    tts = next(c for c in cat["categories"] if c["id"] == "tts")
    yandex = next(p for p in tts["providers"] if p["id"] == "yandex")
    assert yandex["values"]["api_key"] == "AQVN-secret"
    assert yandex["values"]["voice"] == "jane"

    # Core openweathermap api_key and device psk are plaintext too.
    assert cat["core"]["values"]["openweathermap"]["api_key"] == "wkey"
    assert cat["core"]["values"]["devices"][0]["psk"] == "psk-plaintext-1"

    # JSON Schema is still emitted per provider and for core.
    assert "properties" in yandex["schema"]
    assert "properties" in cat["core"]["schema"]


def test_catalog_injects_apply_class_per_field(tmp_path):
    # catalog() injects a backend-computed `apply` class (reconfig.action_for) into EVERY
    # field's schema — provider fields (flat) and core fields (nested $defs) alike. This is
    # the single source of truth for the apply class and doubles as the consistency guard
    # against action_for. The ONLY value meaning "restart required" is "restart".
    from src.reconfig import action_for

    cat = _service(tmp_path).catalog()

    # Stage provider (flat) schemas carry per-field apply.
    llm = next(c for c in cat["categories"] if c["id"] == "llm")
    openrouter = next(p for p in llm["providers"] if p["id"] == "openrouter")
    lp = openrouter["schema"]["properties"]
    assert lp["api_key"]["apply"] == "rebuild_backends"     # NOT "restart"
    assert lp["max_tool_rounds"]["apply"] == "live"
    assert lp["reply_empty"]["apply"] == "live"
    assert lp["timeout"]["apply"] == "rebuild_backends"
    # Every provider field carries an apply consistent with action_for.
    for field, node in lp.items():
        assert node["apply"] == action_for(f"llm.instances.openrouter.{field}")

    tts = next(c for c in cat["categories"] if c["id"] == "tts")
    yandex = next(p for p in tts["providers"] if p["id"] == "yandex")
    yp = yandex["schema"]["properties"]
    assert yp["api_key"]["apply"] == "rebuild_backends"
    assert yp["voice"]["apply"] == "rebuild_backends"
    # The widget/options annotations on yandex voice survive (not clobbered by apply).
    assert yp["voice"]["widget"] == "select"
    assert yp["voice"]["options"] == "dynamic"

    # Core (nested $defs) schemas carry per-field apply on each sub-model's properties.
    core_defs = cat["core"]["schema"]["$defs"]
    assert core_defs["ContextConfig"]["properties"]["max_turns"]["apply"] == "live"
    assert core_defs["AudioConfig"]["properties"]["public_base_url"]["apply"] == "live"
    assert core_defs["AudioConfig"]["properties"]["host"]["apply"] == "rebuild_audio"
    # Scalar / list core properties are annotated on the property itself.
    core_props = cat["core"]["schema"]["properties"]
    assert core_props["tts_timeout"]["apply"] == "rebuild_backends"
    assert core_props["log_level"]["apply"] == "logging"
    assert core_props["mcp_servers"]["apply"] == "rebuild_tools"
    assert core_props["devices"]["apply"] == "rebuild_devices"
    # "restart" is the ONLY value meaning a real restart is required.
    assert core_props["log_level"]["apply"] != "restart"


def test_options_proxies_provider(tmp_path):
    svc = _service(tmp_path)
    assert "zahar" in svc.options("tts", "yandex", "voice")
    assert svc.options("tts", "yandex", "nope") is None


def test_apply_persists_and_updates_get(tmp_path):
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    svc.apply({"tts": {"instances": {"yandex": {"voice": "filipp"}}}})

    assert svc.get("tts").voice == "filipp"
    # Persisted to disk, with the secret intact (not corrupted to "**********").
    saved = config_store.load(path)
    assert saved["tts"]["instances"]["yandex"]["voice"] == "filipp"
    assert saved["core"]["openweathermap"]["api_key"] == "weather-secret"


def test_apply_invalid_patch_raises_and_does_not_persist(tmp_path):
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    with pytest.raises(ValidationError):
        svc.apply({"tts": {"instances": {"yandex": {"speed": 9.9}}}})

    import os
    assert not os.path.exists(path)        # nothing written
    assert svc.get("tts").speed == 1.2     # in-memory state unchanged


def test_on_change_fires_after_successful_apply(tmp_path):
    svc = _service(tmp_path)
    fired = []
    svc.on_change(lambda paths: fired.append(paths))
    svc.apply({"tts": {"instances": {"yandex": {"voice": "ermil"}}}})
    # Fired exactly once with a non-empty change set carrying the touched leaf.
    assert len(fired) == 1
    assert "tts.instances.yandex.voice" in fired[0]


def test_on_change_does_not_fire_on_failed_apply(tmp_path):
    svc = _service(tmp_path)
    fired = []
    svc.on_change(lambda paths: fired.append(paths))
    with pytest.raises(ValidationError):
        svc.apply({"tts": {"instances": {"yandex": {"speed": 9.9}}}})
    assert fired == []


def test_apply_deep_merge_preserves_untouched_siblings(tmp_path):
    # Data-loss regression guard: a partial patch touching ONE nested leaf must not
    # wipe sibling leaves in other categories or core. Assert both the in-memory state
    # and the persisted file keep the untouched values verbatim.
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    svc.apply({"tts": {"instances": {"yandex": {"voice": "zahar"}}}})

    # The targeted leaf changed.
    assert svc.get("tts").voice == "zahar"
    # A sibling in a DIFFERENT category survives in memory.
    assert svc.get("stt").api_key == "gsk-secret"
    # A core value survives in memory.
    assert svc.core.openweathermap.city == "Москва"

    # All three survive on disk too (no silent data loss on persist).
    saved = config_store.load(path)
    assert saved["tts"]["instances"]["yandex"]["voice"] == "zahar"
    assert saved["stt"]["instances"]["groq"]["api_key"] == "gsk-secret"
    assert saved["core"]["openweathermap"]["city"] == "Москва"


def test_provider_returns_selected_provider_object(tmp_path):
    # provider() must hand back the SELECTED provider singleton (same identity as the
    # service's own get_provider path) — not a freshly built backend and not a ConfigModel.
    from src.plugins.base import get_provider
    from src.plugins.tts.yandex import YandexTtsConfig, YandexTtsProvider
    from src.tts import YandexTtsBackend

    svc = _service(tmp_path)
    prov = svc.provider("tts")

    # Exact type and identity: the registry singleton for the selected provider.
    assert type(prov) is YandexTtsProvider
    assert prov is get_provider("tts", "yandex")
    # Not the config model and not a backend instance.
    assert not isinstance(prov, YandexTtsConfig)
    assert not isinstance(prov, YandexTtsBackend)
