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
        "stt": {"instances": {"groq": {"api_key": "gsk-secret"}}},
        "llm": {"instances": {"openrouter": {"api_key": "sk-or-secret", "model": "m1"}}},
        "tts": {
            "variants": [{"provider": "yandex", "enabled": True}],
            "instances": {"yandex": {"api_key": "AQVN-secret", "voice": "jane", "speed": 1.2}},
        },
        "core": {
            "openweathermap": {"api_key": "weather-secret", "city": "Москва"},
            "chains": [{"name": "main", "stt": "groq", "handler": "openrouter"}],
        },
    }


def _service(tmp_path):
    return ConfigService(_doc(), _deps(), path=str(tmp_path / "config.json"))


def test_instance_config_returns_validated_config(tmp_path):
    svc = _service(tmp_path)
    tts_cfg = svc.instance_config("tts", "yandex")
    assert tts_cfg.__class__.__name__ == "YandexTtsConfig"
    assert tts_cfg.voice == "jane"
    assert tts_cfg.speed == 1.2


def test_create_builds_backend(tmp_path):
    svc = _service(tmp_path)
    backend = svc.create_instance("tts", "yandex")
    assert backend.__class__.__name__ == "YandexTtsBackend"


def test_create_adhoc_builds_backend_from_overrides(tmp_path):
    # create_adhoc builds a transient backend from caller-supplied settings (the
    # panel's unsaved draft) for ANY plugin — not just the selected one — without
    # touching the stored document.
    svc = _service(tmp_path)
    backend = svc.create_adhoc("tts", "fishaudio", {"api_key": "k"})
    assert backend.__class__.__name__ == "FishAudioTtsBackend"
    assert backend.api_key == "k"
    # The stored doc is untouched: variants unchanged, no fishaudio instance created.
    doc = svc.document()
    assert doc["tts"]["variants"] == [{"provider": "yandex", "enabled": True}]
    assert "fishaudio" not in doc["tts"]["instances"]


def test_create_adhoc_invalid_settings_raise_validation_error(tmp_path):
    svc = _service(tmp_path)
    with pytest.raises(ValidationError):
        svc.create_adhoc("tts", "yandex", {"speed": 9.9})  # out of range


def test_create_adhoc_unknown_plugin_raises_value_error(tmp_path):
    svc = _service(tmp_path)
    with pytest.raises(ValueError):
        svc.create_adhoc("tts", "nope", {})


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
    assert ids == {"vad", "wakeword", "stt", "llm", "stress", "tts", "followup"}
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
    backend = svc.create("vad")
    assert backend.__class__.__name__ == "WebRtcVadBackend"
    # The catalog exposes the vad category with the webrtc provider selected.
    cat = svc.catalog()
    vad = next(c for c in cat["categories"] if c["id"] == "vad")
    assert vad["selected"] == "webrtc"
    webrtc = next(p for p in vad["providers"] if p["id"] == "webrtc")
    # vad provider fields are stage-instance paths -> rebuild_backends apply class.
    assert webrtc["schema"]["properties"]["aggressiveness"]["apply"] == "rebuild_backends"
    # The decision-only boost is a core conditioning toggle now (pipeline-applied,
    # read live per chunk) -> annotated with the "live" apply class.
    mic_auto_gain = cat["core"]["schema"]["$defs"]["VadConfig"]["properties"]["mic_auto_gain"]
    assert mic_auto_gain["apply"] == "live"


def test_catalog_exposes_values_plainly(tmp_path):
    # Secrets are plain config values now: catalog() emits them verbatim (no masking).
    doc = _doc()
    doc["core"]["openweathermap"]["api_key"] = "wkey"
    cat = ConfigService(doc, _deps(), path=str(tmp_path / "config.json")).catalog()

    # Provider api_key comes through as plaintext.
    tts = next(c for c in cat["categories"] if c["id"] == "tts")
    yandex = next(p for p in tts["providers"] if p["id"] == "yandex")
    assert yandex["values"]["api_key"] == "AQVN-secret"
    assert yandex["values"]["voice"] == "jane"

    # Core openweathermap api_key is plaintext too.
    assert cat["core"]["values"]["openweathermap"]["api_key"] == "wkey"

    # JSON Schema is still emitted per provider and for core.
    assert "properties" in yandex["schema"]
    assert "properties" in cat["core"]["schema"]


def test_catalog_injects_apply_class_per_field(tmp_path):
    # catalog() injects a backend-computed `apply` class (reconfig.action_for) into EVERY
    # field's schema — provider fields (flat) and core fields (nested $defs) alike. This is
    # the single source of truth for the apply class and doubles as the consistency guard
    # against action_for.
    from src.reconfig import action_for

    cat = _service(tmp_path).catalog()

    # Stage provider (flat) schemas carry per-field apply.
    llm = next(c for c in cat["categories"] if c["id"] == "llm")
    openrouter = next(p for p in llm["providers"] if p["id"] == "openrouter")
    lp = openrouter["schema"]["properties"]
    assert lp["api_key"]["apply"] == "rebuild_backends"
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
    assert core_defs["AudioConfig"]["properties"]["host"]["apply"] == "rebuild_audio"
    # Scalar / list core properties are annotated on the property itself.
    core_props = cat["core"]["schema"]["properties"]
    assert core_props["tts_timeout"]["apply"] == "rebuild_backends"
    assert core_props["log_level"]["apply"] == "logging"
    assert core_props["mcp_servers"]["apply"] == "rebuild_tools"
    assert core_props["devices"]["apply"] == "rebuild_devices"


def test_every_config_field_has_an_apply_rule(tmp_path):
    # Every field of every provider and of the core gets its apply class from
    # reconfig.action_for; a field no rule covers comes out "unknown" and would only
    # take effect after a container restart, so a new knob without a rule fails here.
    cat = _service(tmp_path).catalog()
    unruled = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("apply") == "unknown":
                unruled.append(path)
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(cat, "catalog")
    assert unruled == []


def test_options_forwards_search_query_to_provider(tmp_path, monkeypatch):
    # svc.options(cat, plugin, field, query) hands the user-typed search string
    # through to the provider's options() unchanged.
    from src.plugins.base import get_provider
    svc = _service(tmp_path)
    seen = {}
    prov = get_provider("tts", "yandex")

    def fake_options(field, cfg, deps, query=""):
        seen["field"], seen["query"] = field, query
        return ["ok"]

    monkeypatch.setattr(prov, "options", fake_options)
    assert svc.options("tts", "yandex", "voice", "zzz") == ["ok"]
    assert seen == {"field": "voice", "query": "zzz"}


def test_options_proxies_provider(tmp_path):
    svc = _service(tmp_path)
    assert "zahar" in svc.options("tts", "yandex", "voice")
    assert svc.options("tts", "yandex", "nope") is None


def test_apply_persists_and_updates_get(tmp_path):
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    svc.apply({"tts": {"instances": {"yandex": {"voice": "filipp"}}}})

    assert svc.instance_config("tts", "yandex").voice == "filipp"
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
    assert svc.instance_config("tts", "yandex").speed == 1.2     # in-memory state unchanged


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
    # A partial patch touching ONE nested leaf must not
    # wipe sibling leaves in other categories or core. Assert both the in-memory state
    # and the persisted file keep the untouched values verbatim.
    path = str(tmp_path / "config.json")
    svc = ConfigService(_doc(), _deps(), path=path)
    svc.apply({"tts": {"instances": {"yandex": {"voice": "zahar"}}}})

    # The targeted leaf changed.
    assert svc.instance_config("tts", "yandex").voice == "zahar"
    # A sibling in a DIFFERENT category survives in memory.
    assert svc.instance_config("stt", "groq").api_key == "gsk-secret"
    # A core value survives in memory.
    assert svc.core.openweathermap.city == "Москва"

    # All three survive on disk too (no silent data loss on persist).
    saved = config_store.load(path)
    assert saved["tts"]["instances"]["yandex"]["voice"] == "zahar"
    assert saved["stt"]["instances"]["groq"]["api_key"] == "gsk-secret"
    assert saved["core"]["openweathermap"]["city"] == "Москва"


def test_vocabulary_empty_for_cloud_llm(tmp_path):
    # A cloud LLM handler (openrouter) offers no closed vocabulary -> [] (a free-form
    # LLM imposes no grammar on the chain's STT).
    svc = _service(tmp_path)
    assert svc.vocabulary("openrouter") == []


def test_vocabulary_uses_simple_nlu_words(tmp_path):
    # vocabulary(id) returns the handler's alias/verb/number words of the command
    # catalog (the provider-agnostic vocabulary() hook) — the words a narrowing chain
    # restricts to.
    from src.commands import nlu_vocabulary
    from src.core_config import CommandsConfig

    doc = _doc()
    aliases = "свет в зале, люстра = bright_room_light"
    actions = "on = включи\noff = выключи"
    doc["core"]["commands"] = {"aliases": aliases, "actions": actions}
    svc = ConfigService(doc, _deps(), path=str(tmp_path / "config.json"))

    vocab = svc.vocabulary("simple-nlu")
    assert vocab == nlu_vocabulary(aliases, actions, CommandsConfig().filler_words)
    # Non-empty and contains the spoken phrase/verb words.
    assert "свет" in vocab and "включи" in vocab


def test_base_provider_vocabulary_returns_empty(tmp_path):
    # The base hook (used by every cloud LLM provider) returns [] — the contract
    # that keeps vocabulary() provider-agnostic.
    from src.plugins.base import get_provider

    svc = _service(tmp_path)
    prov = get_provider("llm", "openrouter")
    assert prov.vocabulary(svc.instance_config("llm", "openrouter"), svc.core.commands) == []


def test_provider_returns_selected_provider_object(tmp_path):
    # provider() must hand back the SELECTED provider singleton of a slot category
    # (same identity as the service's own get_provider path) — not a backend.
    from src.plugins.base import get_provider
    from src.plugins.vad.webrtc import WebRtcVadBackend

    svc = _service(tmp_path)
    prov = svc.provider("vad")

    assert prov is get_provider("vad", "webrtc")
    assert not isinstance(prov, WebRtcVadBackend)


# --- recognition chains: pools, validation, catalog flags ---------------------------

def test_create_instance_builds_any_pool_provider(tmp_path):
    # create_instance builds one provider instance of a pool category from its stored
    # settings; instance_config validates it (defaults for an absent instance).
    svc = _service(tmp_path)
    assert svc.create_instance("stt", "groq").__class__.__name__ == "GroqSttBackend"
    assert svc.instance_config("llm", "openrouter").model == "m1"
    assert svc.instance_config("llm", "groq").__class__.__name__ == "GroqLlmConfig"


@pytest.mark.parametrize("chains, message", [
    ([], "at least one enabled chain"),
    ([{"name": "a", "enabled": False, "stt": "groq", "handler": "openrouter"}],
     "at least one enabled chain"),
    ([{"name": " ", "stt": "groq", "handler": "openrouter"}], "needs a name"),
    ([{"name": "a", "stt": "groq", "handler": "openrouter"},
      {"name": "a", "stt": "vosk", "handler": "simple-nlu"}], "duplicate"),
    ([{"name": "a", "stt": "nope", "handler": "openrouter"}], "Unknown stt provider"),
    # A disabled chain is validated too: enabling it later must not break the runtime.
    ([{"name": "a", "stt": "groq", "handler": "openrouter"},
      {"name": "b", "enabled": False, "stt": "groq", "handler": "nope"}],
     "Unknown llm provider"),
])
def test_chains_validated_on_init_and_apply(tmp_path, chains, message):
    bad = _doc()
    bad["core"]["chains"] = chains
    with pytest.raises(ValueError, match=message):
        ConfigService(bad, _deps(), path=str(tmp_path / "config.json"))
    svc = _service(tmp_path)
    with pytest.raises(ValueError, match=message):
        svc.apply({"core": {"chains": chains}})
    assert svc.core.chains[0].name == "main"   # nothing applied


def test_chain_referenced_instance_config_is_validated(tmp_path):
    # Every instance a chain references must validate; an unreferenced broken one is
    # not the chains' business.
    svc = _service(tmp_path)
    with pytest.raises(ValidationError):
        svc.apply({"stt": {"instances": {"groq": {"temperature": "hot"}}}})


def test_catalog_flags_handler_limits_and_fallback_phrases_advanced(tmp_path):
    # The panel folds `advanced` fields under the form's "Advanced" section.
    advanced = {"max_tokens", "max_tool_rounds", "timeout", "reply_rate_limit",
                "reply_empty_after_tools", "reply_empty", "reply_error", "reply_error_after_tools"}
    llm = next(c for c in _service(tmp_path).catalog()["categories"] if c["id"] == "llm")
    for p in llm["providers"]:
        props = p["schema"]["properties"]
        assert {f for f, n in props.items() if n.get("advanced")} == advanced & props.keys(), p["id"]
        if "timeout" in props:
            assert props["timeout"]["unit"] == "s"   # merged with the field's other extra


def test_every_panel_form_field_has_a_title():
    # Without one pydantic titles the field from its name ("Tiny Mode", "Api Key").
    from src.plugins.base import REGISTRY
    models = [p.ConfigModel for provs in REGISTRY.values() for p in provs.values()]
    models += [CoreConfig.model_fields[s].annotation for s in (
        "context", "audio", "vad", "ack", "network", "openweathermap", "calendar",
        "reminders", "agent_mcp", "runs", "commands")]
    untitled = [f"{m.__name__}.{name}" for m in models for name, f in m.model_fields.items()
                if f.title is None and not (f.json_schema_extra or {}).get("hidden")]
    assert untitled == []


def test_catalog_pool_categories_have_no_selected_and_vocabulary_flags(tmp_path):
    cat = {c["id"]: c for c in _service(tmp_path).catalog()["categories"]}
    assert "selected" not in cat["stt"] and "selected" not in cat["llm"]
    assert "selected" not in cat["tts"]
    assert cat["tts"]["variants"] == [{"provider": "yandex", "enabled": True}]
    stt = {p["id"]: p for p in cat["stt"]["providers"]}
    assert stt["vosk"]["supports_vocabulary"] is True
    assert stt["groq"]["supports_vocabulary"] is False
    llm = {p["id"]: p for p in cat["llm"]["providers"]}
    assert llm["simple-nlu"]["has_vocabulary"] is True
    assert llm["openrouter"]["has_vocabulary"] is False
    # core.chains is a list property: annotated with its apply class.
    assert _service(tmp_path).catalog()["core"]["schema"]["properties"]["chains"]["apply"] \
        == "rebuild_backends"


# --- TTS variants: validation, order, PATCH ---------------------------------------

@pytest.mark.parametrize("variants, message", [
    ([], "at least one enabled variant"),
    ([{"provider": "yandex", "enabled": False}], "at least one enabled variant"),
    ([{"provider": "yandex"}, {"provider": "yandex", "enabled": False}], "duplicate"),
    ([{"provider": "yandex"}, {"provider": "nope"}], "Unknown tts provider"),
])
def test_tts_variants_validated_on_init_and_apply(tmp_path, variants, message):
    bad = _doc()
    bad["tts"]["variants"] = variants
    with pytest.raises(ValueError, match=message):
        ConfigService(bad, _deps(), path=str(tmp_path / "config.json"))
    svc = _service(tmp_path)
    with pytest.raises(ValueError, match=message):
        svc.apply({"tts": {"variants": variants}})
    assert svc.tts_variants() == ["yandex"]   # nothing applied


def test_disabled_tts_variant_instance_is_validated(tmp_path):
    # A disabled variant is validated too: enabling it later must not break the runtime.
    svc = _service(tmp_path)
    with pytest.raises(ValidationError):
        svc.apply({"tts": {"variants": [{"provider": "yandex"},
                                        {"provider": "fishaudio", "enabled": False}],
                           "instances": {"fishaudio": {"speed": 9.9}}}})


def test_tts_variants_patch_replaces_list_and_skips_disabled(tmp_path):
    # PATCH replaces the whole list; tts_variants() is the enabled ones in list order.
    svc = _service(tmp_path)
    svc.apply({"tts": {"variants": [{"provider": "piper", "enabled": True},
                                    {"provider": "silero", "enabled": False},
                                    {"provider": "yandex", "enabled": True}]}})
    assert svc.tts_variants() == ["piper", "yandex"]
    svc.apply({"tts": {"variants": [{"provider": "yandex", "enabled": True}]}})
    assert svc.document()["tts"]["variants"] == [{"provider": "yandex", "enabled": True}]
