import httpx
import pytest
from pydantic import ValidationError

import src.plugins  # noqa: F401  register all providers
from src import config_store
from src.config_service import ConfigService, mask_secrets
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
        "core": {"weather": {"api_key": "weather-secret", "city": "Москва"}},
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
    assert svc.core.weather.city == "Москва"


def test_constructor_validates_selected_instances(tmp_path):
    bad = _doc()
    bad["tts"]["instances"]["yandex"]["speed"] = 9.9  # out of range
    with pytest.raises(ValidationError):
        ConfigService(bad, _deps(), path=str(tmp_path / "config.json"))


def test_catalog_lists_all_categories_with_schemas(tmp_path):
    cat = _service(tmp_path).catalog()
    ids = {c["id"] for c in cat["categories"]}
    assert ids == {"stt", "llm", "tts"}
    for c in cat["categories"]:
        for prov in c["providers"]:
            assert "schema" in prov and "properties" in prov["schema"]
            assert "values" in prov
    assert "schema" in cat["core"] and "values" in cat["core"]


def test_catalog_masks_secrets_and_never_leaks_plaintext(tmp_path):
    cat = _service(tmp_path).catalog()
    blob = repr(cat)
    # No plaintext secret anywhere in the emitted catalog.
    for secret in ("gsk-secret", "sk-or-secret", "AQVN-secret", "weather-secret"):
        assert secret not in blob

    # The yandex api_key field is masked to {"is_set": True}.
    tts = next(c for c in cat["categories"] if c["id"] == "tts")
    yandex = next(p for p in tts["providers"] if p["id"] == "yandex")
    assert yandex["values"]["api_key"] == {"is_set": True}
    assert yandex["values"]["voice"] == "jane"  # non-secret passes through

    # Core weather secret masked; city stays readable.
    assert cat["core"]["values"]["weather"]["api_key"] == {"is_set": True}
    assert cat["core"]["values"]["weather"]["city"] == "Москва"


def test_mask_secrets_unset_secret_is_false():
    masked = mask_secrets(CoreConfig())
    assert masked["weather"]["api_key"] == {"is_set": False}
    assert masked["mcp"]["token"] == {"is_set": False}


def test_catalog_masks_psk_in_device_list(tmp_path):
    # devices is a list of models, each with a secret `psk` field — masking must
    # recurse into the list and never leak the plaintext psk.
    doc = _doc()
    doc["core"]["devices"] = [
        {"name": "kitchen", "host": "10.0.0.5", "psk": "psk-plaintext-1"},
        {"name": "bedroom", "host": "10.0.0.6", "psk": "psk-plaintext-2"},
    ]
    cat = ConfigService(doc, _deps(), path=str(tmp_path / "config.json")).catalog()

    blob = repr(cat)
    assert "psk-plaintext-1" not in blob
    assert "psk-plaintext-2" not in blob

    devices = cat["core"]["values"]["devices"]
    assert len(devices) == 2
    for dev in devices:
        assert dev["psk"] == {"is_set": True}
    assert {d["name"] for d in devices} == {"kitchen", "bedroom"}


def test_apply_with_masked_secret_placeholder_raises_and_does_not_persist(tmp_path):
    # Documented contract: a masked placeholder ({"is_set": ...}) sent back for a
    # secret field fails validation and the file is not written.
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    with pytest.raises(ValidationError):
        svc.apply({"core": {"weather": {"api_key": {"is_set": True}}}})

    import os
    assert not os.path.exists(path)  # nothing persisted
    # In-memory secret is untouched.
    assert svc.core.weather.api_key.get_secret_value() == "weather-secret"


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
    assert saved["core"]["weather"]["api_key"] == "weather-secret"


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
    svc.on_change(lambda: fired.append(True))
    svc.apply({"tts": {"instances": {"yandex": {"voice": "ermil"}}}})
    assert fired == [True]


def test_on_change_does_not_fire_on_failed_apply(tmp_path):
    svc = _service(tmp_path)
    fired = []
    svc.on_change(lambda: fired.append(True))
    with pytest.raises(ValidationError):
        svc.apply({"tts": {"instances": {"yandex": {"speed": 9.9}}}})
    assert fired == []
