"""Tests for the composition-root helpers in src.app.

These cover the small pure helpers extracted out of async main(): first-boot config
creation, the legacy-mcp warning, and the empty-public_base_url warning. They assert
observable behaviour (returned doc, save calls, emitted log records), so each fails
if the targeted logic is broken.
"""

import contextlib

from loguru import logger

from src import app, config_store
from src.core_config import CoreConfig


@contextlib.contextmanager
def capture_logs(level="INFO"):
    """Yield a growing list of formatted loguru records emitted within the block.

    Mirrors the suite's loguru-capture idiom (add a sink to a list, remove it after).
    """
    records = []
    sink_id = logger.add(records.append, level=level, format="{level.name} {message}")
    try:
        yield records
    finally:
        logger.remove(sink_id)


# --- load_or_create_config -------------------------------------------------


def test_load_or_create_config_first_boot(monkeypatch):
    # An empty store means first boot: the template must be read, saved, and returned.
    template_doc = {"version": 1, "core": {"audio": {"public_base_url": "http://x:8080"}}}
    saved = []

    monkeypatch.setattr(config_store, "load", lambda *a, **k: {})
    monkeypatch.setattr(config_store, "save", lambda doc, path: saved.append((doc, path)))

    opened = []

    class _FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_open(path, *args, **kwargs):
        opened.append(path)
        return _FakeFile()

    # Intercept the template read inside app.load_or_create_config.
    monkeypatch.setattr(app, "open", fake_open, raising=False)
    monkeypatch.setattr(app.json, "load", lambda f: dict(template_doc))

    result = app.load_or_create_config()

    # Returned the template doc...
    assert result == template_doc
    # ...read it from the template path...
    assert opened == ["templates/default_config.json"]
    # ...and persisted it to the default config path exactly once.
    assert len(saved) == 1
    saved_doc, saved_path = saved[0]
    assert saved_doc == template_doc
    assert saved_path == config_store.DEFAULT_PATH


def test_load_or_create_config_existing_config(monkeypatch):
    # A non-empty store must be returned verbatim, WITHOUT saving or touching the
    # template (guards against clobbering an operator's config every boot).
    existing = {"version": 1, "core": {"audio": {"public_base_url": "http://op:9000"}}}

    monkeypatch.setattr(config_store, "load", lambda *a, **k: existing)

    def boom_save(*a, **k):
        raise AssertionError("save must not be called for an existing config")

    def boom_open(*a, **k):
        raise AssertionError("template must not be opened for an existing config")

    monkeypatch.setattr(config_store, "save", boom_save)
    monkeypatch.setattr(app, "open", boom_open, raising=False)

    result = app.load_or_create_config()

    assert result is existing


# --- migrate_vad_plugin ------------------------------------------------------


def test_migrate_vad_plugin_moves_aggressiveness_and_copies_auto_gain():
    # Old doc: core.vad.aggressiveness present, mic_normalize on, no vad slot.
    doc = {
        "version": 1,
        "core": {"vad": {"aggressiveness": 3, "mic_normalize": True, "silence_ms": 800}},
    }
    assert app.migrate_vad_plugin(doc) is True
    # aggressiveness MOVED into the new vad/webrtc instance; core key deleted.
    assert doc["vad"] == {
        "selected": "webrtc",
        "instances": {"webrtc": {"aggressiveness": 3, "auto_gain": True}},
    }
    assert "aggressiveness" not in doc["core"]["vad"]
    # mic_normalize was COPIED, not moved: it still gates the pre-STT normalization.
    assert doc["core"]["vad"]["mic_normalize"] is True
    # Untouched policy fields survive.
    assert doc["core"]["vad"]["silence_ms"] == 800


def test_migrate_vad_plugin_without_mic_normalize_leaves_auto_gain_unset():
    doc = {"core": {"vad": {"aggressiveness": 1, "mic_normalize": False}}}
    assert app.migrate_vad_plugin(doc) is True
    assert doc["vad"]["instances"]["webrtc"] == {"aggressiveness": 1}


def test_migrate_vad_plugin_noop_on_already_migrated_doc():
    # A doc that already carries the vad slot and no core aggressiveness must be
    # left alone — in particular a panel-disabled auto_gain is NOT re-enabled even
    # though mic_normalize is still on (the copy is one-time).
    doc = {
        "vad": {"selected": "webrtc", "instances": {"webrtc": {"auto_gain": False}}},
        "core": {"vad": {"mic_normalize": True, "silence_ms": 800}},
    }
    assert app.migrate_vad_plugin(doc) is False
    assert doc["vad"]["instances"]["webrtc"] == {"auto_gain": False}


def test_migrate_vad_plugin_noop_without_core_vad():
    doc = {"version": 1, "core": {"audio": {"port": 8200}}}
    assert app.migrate_vad_plugin(doc) is False
    assert "vad" not in doc


def test_migrate_vad_plugin_merges_into_existing_slot():
    # An existing vad slot (e.g. operator pre-created it) is reused, not clobbered:
    # selected and unrelated instance fields survive, aggressiveness lands inside.
    doc = {
        "vad": {"selected": "webrtc", "instances": {"webrtc": {"auto_gain": True}}},
        "core": {"vad": {"aggressiveness": 0}},
    }
    assert app.migrate_vad_plugin(doc) is True
    assert doc["vad"]["selected"] == "webrtc"
    assert doc["vad"]["instances"]["webrtc"] == {"auto_gain": True, "aggressiveness": 0}


def test_migrate_vad_plugin_no_core_key_returns_false():
    # The migration runs at boot before everything; a doc missing "core" entirely
    # must be a clean no-op (False, no vad slot), not a boot-killing exception.
    doc = {"version": 1}
    assert app.migrate_vad_plugin(doc) is False
    assert "vad" not in doc


def test_migrate_vad_plugin_corrupt_core_shapes_return_false():
    # Each hand-corrupted shape trips one of the migration's guards: a non-dict
    # "core" (isinstance(core, dict)) or a null/non-dict "core.vad"
    # (isinstance(core_vad, dict)). All must be quiet no-ops — no exception, no
    # "vad" slot created (a crash here would brick startup).
    corrupt_docs = [
        {"core": None},            # core null
        {"core": "oops"},          # core non-dict
        {"core": {"vad": None}},   # vad-source value null
        {"core": {"vad": "oops"}}, # vad-source value non-dict
        {"core": {"vad": 3}},      # vad-source value scalar
    ]
    for doc in corrupt_docs:
        assert app.migrate_vad_plugin(doc) is False, doc
        assert "vad" not in doc, doc


def test_migrate_vad_plugin_creates_instances_in_slot_without_instances():
    # An existing vad slot WITHOUT an "instances" key: the migration creates the
    # instances dict and the migrated value lands inside it (selected survives).
    doc = {
        "vad": {"selected": "webrtc"},
        "core": {"vad": {"aggressiveness": 2}},
    }
    assert app.migrate_vad_plugin(doc) is True
    assert doc["vad"]["instances"] == {"webrtc": {"aggressiveness": 2}}
    assert doc["vad"]["selected"] == "webrtc"
    assert "aggressiveness" not in doc["core"]["vad"]


def test_load_or_create_config_saves_migrated_doc(monkeypatch):
    # An existing OLD config triggers the migration and is saved back exactly once.
    existing = {
        "version": 1,
        "core": {"vad": {"aggressiveness": 2, "mic_normalize": False}},
    }
    saved = []
    monkeypatch.setattr(config_store, "load", lambda *a, **k: existing)
    monkeypatch.setattr(config_store, "save", lambda doc, path: saved.append((doc, path)))

    with capture_logs() as records:
        result = app.load_or_create_config()

    assert result["vad"]["instances"]["webrtc"]["aggressiveness"] == 2
    assert len(saved) == 1
    assert saved[0][0] is result
    assert saved[0][1] == config_store.DEFAULT_PATH
    assert any("config migrated" in r for r in records)


# --- migrate_mcp_slow --------------------------------------------------------


def test_migrate_mcp_slow_marker_name_gets_true():
    # A server whose NAME matches a legacy SLOW_TOOL_MARKERS substring is seeded
    # slow=true, with an info line stating the (heuristic) decision.
    doc = {"core": {"mcp_servers": [{"name": "web-search", "url": "http://x"}]}}
    with capture_logs() as records:
        assert app.migrate_mcp_slow(doc) is True
    assert doc["core"]["mcp_servers"][0]["slow"] is True
    assert any("'web-search': slow=true (migrated; was name-heuristic)" in r for r in records)
    # All-true migration: no operator warning needed.
    assert not any("WARNING" in r for r in records)


def test_migrate_mcp_slow_non_marker_gets_false_written_and_warns():
    # A non-marker server gets an EXPLICIT slow=false (so the migration never
    # re-runs) plus a warning pointing the operator at the panel toggle.
    doc = {"core": {"mcp_servers": [{"name": "nodered", "url": "http://x"}]}}
    with capture_logs() as records:
        assert app.migrate_mcp_slow(doc) is True
    assert doc["core"]["mcp_servers"][0]["slow"] is False
    assert any("'nodered': slow=false (migrated; was name-heuristic)" in r for r in records)
    warnings = [r for r in records if "WARNING" in r]
    assert len(warnings) == 1
    assert "Slow tools" in warnings[0]


def test_migrate_mcp_slow_existing_flag_untouched():
    # Entries that already carry "slow" (either value) are never touched — in
    # particular a marker-named server with an operator-set false stays false.
    doc = {"core": {"mcp_servers": [
        {"name": "web-search", "slow": False},
        {"name": "nodered", "slow": True},
    ]}}
    assert app.migrate_mcp_slow(doc) is False
    assert doc["core"]["mcp_servers"][0]["slow"] is False
    assert doc["core"]["mcp_servers"][1]["slow"] is True


def test_migrate_mcp_slow_no_mcp_servers_key_is_noop_without_save(monkeypatch):
    # No mcp_servers key (and no other migration trigger): the helper is a no-op
    # and load_or_create_config must NOT save the doc back.
    doc = {"version": 1, "core": {"audio": {"public_base_url": "http://x"}}}
    assert app.migrate_mcp_slow(doc) is False

    monkeypatch.setattr(config_store, "load", lambda *a, **k: dict(doc))

    def boom_save(*a, **k):
        raise AssertionError("save must not be called when no migration ran")

    monkeypatch.setattr(config_store, "save", boom_save)
    result = app.load_or_create_config()
    assert "mcp_servers" not in result["core"]


def test_migrate_mcp_slow_mixed_list():
    # Mixed list: marker name -> true, plain name -> false, existing flag kept.
    doc = {"core": {"mcp_servers": [
        {"name": "Wiki-lookup"},          # marker (case-insensitive) -> true
        {"name": "ha-bridge"},            # no marker -> explicit false
        {"name": "google-events", "slow": False},  # already set -> untouched
    ]}}
    assert app.migrate_mcp_slow(doc) is True
    assert [s["slow"] for s in doc["core"]["mcp_servers"]] == [True, False, False]


def test_migrate_mcp_slow_second_run_is_noop():
    # Idempotency: the first run writes every flag explicitly, so a second run
    # finds nothing to do and returns False (no repeated save at every boot).
    doc = {"core": {"mcp_servers": [{"name": "web-search"}, {"name": "nodered"}]}}
    assert app.migrate_mcp_slow(doc) is True
    snapshot = [dict(s) for s in doc["core"]["mcp_servers"]]
    assert app.migrate_mcp_slow(doc) is False
    assert doc["core"]["mcp_servers"] == snapshot


def test_migrate_mcp_slow_corrupt_shapes_are_safe():
    # Hand-broken configs must never crash the boot-time migration (mirrors the
    # migrate_vad_plugin corrupt-shapes guard): a null core or a non-list
    # mcp_servers is a quiet no-op.
    assert app.migrate_mcp_slow({"core": None}) is False
    assert app.migrate_mcp_slow({"core": {"mcp_servers": "oops"}}) is False

    # A list mixing non-dict garbage with one real entry: the garbage elements are
    # skipped untouched, the dict entry still gains its explicit flag ("web" hits
    # a legacy slow marker, so it is seeded slow=true).
    doc = {"core": {"mcp_servers": [42, "x", {"name": "web"}]}}
    assert app.migrate_mcp_slow(doc) is True
    assert doc["core"]["mcp_servers"][:2] == [42, "x"]
    assert doc["core"]["mcp_servers"][2]["slow"] is True


def test_load_or_create_config_saves_mcp_slow_migration(monkeypatch):
    # A pre-R6 config (mcp_servers without "slow") triggers the migration and is
    # saved back via load_or_create_config.
    existing = {"version": 1, "core": {"mcp_servers": [{"name": "websearch", "url": "http://x"}]}}
    saved = []
    monkeypatch.setattr(config_store, "load", lambda *a, **k: existing)
    monkeypatch.setattr(config_store, "save", lambda doc, path: saved.append((doc, path)))

    with capture_logs() as records:
        result = app.load_or_create_config()

    assert result["core"]["mcp_servers"][0]["slow"] is True
    assert len(saved) == 1
    assert saved[0] == (result, config_store.DEFAULT_PATH)
    assert any("explicit 'slow' flags written" in r for r in records)


# --- warn_legacy_mcp -------------------------------------------------------


def test_warn_legacy_mcp_fires_on_legacy_key():
    # Legacy 'core.mcp' present -> warning fires.
    doc = {"core": {"mcp": {"url": "http://legacy:1234"}}}
    with capture_logs() as records:
        app.warn_legacy_mcp(doc)
    legacy = [r for r in records if "legacy 'core.mcp'" in r]
    assert len(legacy) == 1
    assert "WARNING" in legacy[0]


def test_warn_legacy_mcp_silent_on_new_key():
    # Only the new 'core.mcp_servers' present -> no warning.
    doc = {"core": {"mcp_servers": [{"name": "weather", "url": "http://new:1234"}]}}
    with capture_logs() as records:
        app.warn_legacy_mcp(doc)
    assert not any("legacy 'core.mcp'" in r for r in records)


# --- validate_boot_config --------------------------------------------------


def test_validate_boot_config_warns_on_empty_public_base_url():
    # Empty public_base_url -> "play nothing" warning fires.
    core = CoreConfig()  # default public_base_url is ""
    assert core.audio.public_base_url == ""
    with capture_logs() as records:
        app.validate_boot_config(core)
    warned = [r for r in records if "public_base_url is empty" in r]
    assert len(warned) == 1
    assert "WARNING" in warned[0]


def test_validate_boot_config_silent_on_set_public_base_url():
    # Non-empty public_base_url -> silent.
    core = CoreConfig(audio={"public_base_url": "http://this-host:8080"})
    with capture_logs() as records:
        app.validate_boot_config(core)
    assert not any("public_base_url is empty" in r for r in records)
