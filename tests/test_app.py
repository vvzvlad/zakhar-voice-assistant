"""Tests for the composition-root helpers in src.app.

These cover the small pure helpers extracted out of async main(): first-boot config
creation, the legacy-mcp warning and the empty-public_base_url warning. They assert
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
