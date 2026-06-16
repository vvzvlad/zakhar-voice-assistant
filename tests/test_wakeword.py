"""Unit tests for the Vosk grammar-restricted wakeword verification stage.

The real `vosk` package and its models are NEVER imported here: the backend takes
an injected fake model (and a stubbed recognizer factory), the provider is
exercised on the disabled (passthrough) path, and registration/config wiring is
checked via the REGISTRY / ConfigService. So the suite stays hermetic and the
heavy dependency is not needed to run it.
"""

import json

import httpx
import pytest

import src.plugins  # noqa: F401  register all providers
from src.config_service import ConfigService
from src.plugins.base import REGISTRY, Deps
from src.plugins.wakeword.vosk import (
    VoskWakewordConfig,
    VoskWakewordProvider,
    VoskWakewordVerifier,
)
from src.reconfig import action_for, backend_categories
from src.wakeword import PassthroughVerifier, WakewordVerdict, WakewordVerifier


def _deps():
    return Deps(
        http_cloud=httpx.AsyncClient(),
        http_local=httpx.AsyncClient(),
        tts_timeout=30,
    )


class _FakeRecognizer:
    """KaldiRecognizer double: returns a fixed FinalResult JSON; records grammar."""

    def __init__(self, result: dict):
        self._result = result
        self.words = None
        self.accepted = None

    def SetWords(self, on):
        self.words = on

    def AcceptWaveform(self, pcm):
        self.accepted = pcm
        return True

    def FinalResult(self):
        return json.dumps(self._result, ensure_ascii=False)


def _verifier(result: dict, *, keywords=None, on_error="open", timeout_ms=300):
    """Build a VoskWakewordVerifier with an injected model and a stubbed recognizer
    that always returns `result`, so no real Vosk model is ever loaded."""
    v = VoskWakewordVerifier(
        model_path="unused",
        keywords=keywords or ["захар"],
        window_ms=1500,
        timeout_ms=timeout_ms,
        on_error=on_error,
        model=object(),  # injected sentinel -> no real model load
    )
    v._make_recognizer = lambda: _FakeRecognizer(result)
    return v


# --- contract: PassthroughVerifier -------------------------------------------

async def test_passthrough_always_accepts():
    p = PassthroughVerifier()
    assert isinstance(p, WakewordVerifier)
    v = await p.verify(b"\x00\x01anything")
    assert v == WakewordVerdict(accepted=True, score=None)
    # Empty PCM is accepted too (the disabled stage never gates).
    assert (await p.verify(b"")) == WakewordVerdict(accepted=True, score=None)


def test_passthrough_policy_attrs():
    # The pipeline reads these to apply the timeout + fail-open policy.
    p = PassthroughVerifier()
    assert p.fail_open is True
    assert p.timeout_ms == 1000


# --- backend: VoskWakewordVerifier (injected model, stubbed recognizer) -------

async def test_verify_accepts_when_keyword_in_text_with_conf_score():
    # A matched keyword -> accepted; the score is the max per-word conf among matches.
    result = {"text": "захар", "result": [{"word": "захар", "conf": 0.83}]}
    v = _verifier(result)
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is True
    assert verdict.score == pytest.approx(0.83)


async def test_verify_matches_case_insensitively():
    # The Russian Vosk model emits lowercase text. A capitalized operator-configured
    # keyword ("Захар") is normalized to lowercase so it still matches the lowercase
    # decoded token. Also asserts the grammar JSON is built from lowercased keywords.
    result = {"text": "захар", "result": [{"word": "захар", "conf": 0.77}]}
    v = _verifier(result, keywords=["Захар"])
    # The grammar (and accept-set) use the lowercased keyword.
    grammar = json.loads(json.dumps(v._keywords + ["[unk]"], ensure_ascii=False))
    assert grammar == ["захар", "[unk]"]
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is True
    assert verdict.score == pytest.approx(0.77)


async def test_verify_rejects_when_no_keyword_present():
    # No keyword token in the recognized text -> rejected, score 0.0.
    result = {"text": "[unk]", "result": [{"word": "[unk]", "conf": 0.5}]}
    v = _verifier(result)
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is False
    assert verdict.score == 0.0


async def test_verify_accepts_without_word_confs_falls_back_to_binary_score():
    # text matches but no per-word "result" array -> score 1.0 when accepted.
    v = _verifier({"text": "захар"})
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is True
    assert verdict.score == 1.0


async def test_verify_grammar_recognizer_gets_words_and_waveform():
    # SetWords(True) and AcceptWaveform(pcm) are invoked on the recognizer.
    rec = _FakeRecognizer({"text": "захар"})
    v = VoskWakewordVerifier(
        model_path="unused", keywords=["захар"], window_ms=1500,
        timeout_ms=300, on_error="open", model=object(),
    )
    v._make_recognizer = lambda: rec
    await v.verify(b"pcmbytes")
    assert rec.words is True
    assert rec.accepted == b"pcmbytes"


async def test_verify_any_of_several_single_word_keywords_matches():
    # Several single-word keywords configured (OR): the recognized token matching
    # ANY of them accepts.
    result = {"text": "алиса", "result": [{"word": "алиса", "conf": 0.9}]}
    v = _verifier(result, keywords=["захар", "алиса", "окей"])
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is True
    assert verdict.score == pytest.approx(0.9)


async def test_verify_multiword_keyword_phrase_matches_when_contiguous():
    # A multi-word keyword ("окей захар") matches when the recognizer returns that
    # phrase as contiguous tokens. The score is the max per-word conf of the run.
    result = {
        "text": "окей захар",
        "result": [{"word": "окей", "conf": 0.7}, {"word": "захар", "conf": 0.82}],
    }
    v = _verifier(result, keywords=["окей захар"])
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is True
    assert verdict.score == pytest.approx(0.82)


async def test_verify_multiword_keyword_does_not_match_on_single_word():
    # Only ONE word of a multi-word phrase appears -> the contiguous run is absent,
    # so the phrase keyword does NOT match.
    result = {"text": "захар", "result": [{"word": "захар", "conf": 0.9}]}
    v = _verifier(result, keywords=["окей захар"])
    verdict = await v.verify(b"pcmbytes")
    assert verdict.accepted is False
    assert verdict.score == 0.0


async def test_verify_empty_keywords_accepts():
    # An operator who cleared the keyword list disables the gate: verify accepts
    # (score None) rather than rejecting every wake. Built directly (not via the
    # _verifier helper, which would substitute the default keyword for an empty list).
    v = VoskWakewordVerifier(
        model_path="unused", keywords=[], window_ms=1500,
        timeout_ms=300, on_error="closed", model=object(),
    )
    # The guard must hold even under the strict fail-closed policy.
    assert v._keywords == []
    verdict = await v.verify(b"pcmbytes")
    assert verdict == WakewordVerdict(accepted=True, score=None)


async def test_verify_empty_pcm_applies_fail_policy():
    # No audio -> fail-open accepts, fail-closed rejects; score is None either way.
    open_v = _verifier({"text": "захар"}, on_error="open")
    closed_v = _verifier({"text": "захар"}, on_error="closed")
    assert (await open_v.verify(b"")) == WakewordVerdict(accepted=True, score=None)
    assert (await closed_v.verify(b"")) == WakewordVerdict(accepted=False, score=None)


async def test_verify_never_raises_on_decode_error_uses_fail_policy():
    # A decode that raises must NOT propagate: fall back to the fail policy.
    def _boom():
        raise RuntimeError("native vosk blew up")

    v = VoskWakewordVerifier(
        model_path="unused", keywords=["захар"], window_ms=1500,
        timeout_ms=300, on_error="open", model=object(),
    )
    v._make_recognizer = _boom
    verdict = await v.verify(b"pcmbytes")
    assert verdict == WakewordVerdict(accepted=True, score=None)

    v._make_recognizer = _boom
    v.fail_open = False
    assert (await v.verify(b"pcmbytes")) == WakewordVerdict(accepted=False, score=None)


def test_verifier_reads_policy_attrs_from_config():
    # timeout_ms and fail_open (= on_error=='open') are taken from config.
    v = _verifier({"text": "захар"}, on_error="closed", timeout_ms=250)
    assert v.timeout_ms == 250
    assert v.fail_open is False


# --- provider wiring ---------------------------------------------------------

def test_provider_disabled_returns_passthrough_without_model_load():
    # enabled=False must return a PassthroughVerifier and never touch the model.
    backend = VoskWakewordProvider().create(VoskWakewordConfig(enabled=False), _deps())
    assert isinstance(backend, PassthroughVerifier)


def test_provider_describe():
    prov = VoskWakewordProvider()
    enabled = VoskWakewordConfig(enabled=True, model_path="models/vosk-model-small-ru-0.22")
    assert prov.describe(enabled) == "vosk/vosk-model-small-ru-0.22"
    assert prov.describe(VoskWakewordConfig(enabled=False)) == "vosk/off"


def test_provider_is_registered():
    assert "vosk" in REGISTRY["wakeword"]


def test_config_defaults():
    cfg = VoskWakewordConfig()
    assert cfg.enabled is True
    assert cfg.keywords == ["захар"]
    assert cfg.window_ms == 1500
    assert cfg.timeout_ms == 300
    assert cfg.on_error == "open"


# --- ConfigService round-trip ------------------------------------------------

def _doc():
    return {
        "version": 1,
        "stt": {"selected": "groq", "instances": {"groq": {"api_key": "gsk-secret"}}},
        "llm": {"selected": "openrouter", "instances": {"openrouter": {"api_key": "k", "model": "m"}}},
        "tts": {"selected": "teratts", "instances": {"teratts": {"base_url": ""}}},
        "core": {},
    }


def test_old_doc_without_wakeword_slot_gets_vosk_defaults(tmp_path):
    # Backward compat: a config written before the wakeword stage existed (no
    # "wakeword" key) must still parse, defaulting the slot to the vosk provider.
    svc = ConfigService(_doc(), _deps(), path=str(tmp_path / "config.json"))
    cfg = svc.get("wakeword")
    assert cfg.__class__.__name__ == "VoskWakewordConfig"
    assert cfg.enabled is True
    assert cfg.keywords == ["захар"]


def test_catalog_includes_wakeword_category(tmp_path):
    svc = ConfigService(_doc(), _deps(), path=str(tmp_path / "config.json"))
    cat = svc.catalog()
    ids = {c["id"] for c in cat["categories"]}
    assert "wakeword" in ids
    ww = next(c for c in cat["categories"] if c["id"] == "wakeword")
    assert ww["selected"] == "vosk"
    vosk = next(p for p in ww["providers"] if p["id"] == "vosk")
    # Stage-instance fields classify as rebuild_backends.
    assert vosk["schema"]["properties"]["enabled"]["apply"] == "rebuild_backends"


def test_config_create_disabled_round_trips_to_passthrough(tmp_path):
    # A stored disabled wakeword slot validates and create() yields a Passthrough
    # (no real model load), so the whole slot round-trips through ConfigService.
    doc = _doc()
    doc["wakeword"] = {"selected": "vosk", "instances": {"vosk": {"enabled": False}}}
    svc = ConfigService(doc, _deps(), path=str(tmp_path / "config.json"))
    backend = svc.create("wakeword")
    assert isinstance(backend, PassthroughVerifier)


# --- reconfig classification -------------------------------------------------

def test_action_for_wakeword_is_rebuild_backends():
    assert action_for("wakeword.instances.vosk.enabled") == "rebuild_backends"
    assert action_for("wakeword.selected") == "rebuild_backends"


def test_backend_categories_includes_wakeword():
    assert "wakeword" in backend_categories({"wakeword.instances.vosk.enabled"})
    # The synthetic warm-up path reduces to the {wakeword} category too.
    assert backend_categories({"wakeword.__warmup__"}) == {"wakeword"}
