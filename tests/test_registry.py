import pytest

import src.plugins  # noqa: F401  triggers @register on all providers
from src.plugins.base import REGISTRY, get_provider, providers


def test_tts_ids_registered():
    assert set(providers("tts")) == {"teratts", "piper", "yandex"}


def test_stt_ids_registered():
    assert set(providers("stt")) == {"groq", "vosk"}


def test_llm_ids_registered():
    assert set(providers("llm")) == {"openrouter", "groq"}


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
