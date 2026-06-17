import pytest

import src.plugins  # noqa: F401  triggers @register on all providers
from src.plugins.base import REGISTRY, Provider, get_provider, providers, register


def test_tts_ids_registered():
    assert set(providers("tts")) == {"piper", "yandex", "fishaudio"}


def test_stt_ids_registered():
    assert set(providers("stt")) == {"groq", "vosk", "openrouter", "yandex"}


def test_llm_ids_registered():
    assert set(providers("llm")) == {"openrouter", "groq", "simple-nlu"}


def test_registry_holds_provider_instances():
    prov = get_provider("tts", "yandex")
    assert prov.category == "tts"
    assert prov.id == "yandex"
    assert prov.label == "Yandex SpeechKit"


def test_get_provider_unknown_category_raises():
    with pytest.raises(ValueError):
        get_provider("nope", "x")


def test_get_provider_unknown_id_raises():
    with pytest.raises(ValueError):
        get_provider("tts", "does-not-exist")


def test_providers_returns_copy():
    snap = providers("tts")
    snap["fake"] = object()
    assert "fake" not in REGISTRY["tts"]


def test_duplicate_registration_raises():
    # Register under a fresh fake category to avoid polluting the real REGISTRY;
    # a second register of the same id must raise instead of silently shadowing.
    class _Fake(Provider):
        category = "fake-cat"
        id = "dup"

    try:
        register(_Fake)  # first registration succeeds
        with pytest.raises(ValueError):
            register(_Fake)  # duplicate category/id -> guard fires
    finally:
        REGISTRY.pop("fake-cat", None)  # keep the real REGISTRY clean
