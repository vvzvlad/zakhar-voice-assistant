"""Tests for the composition-root helpers in src.app.

These cover the small pure helper extracted out of async main(): first-boot config
creation. They assert observable behaviour (returned doc, save calls), so each fails
if the targeted logic is broken.
"""

from src import app, config_store


# --- load_or_create_config -------------------------------------------------


def test_load_or_create_config_first_boot(monkeypatch):
    # An empty store means first boot: the template must be read, saved, and returned.
    template_doc = {"version": 1, "core": {"audio": {"port": 8200}}}
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
    existing = {"version": 1, "core": {"audio": {"port": 9000}}}

    monkeypatch.setattr(config_store, "load", lambda *a, **k: existing)

    def boom_save(*a, **k):
        raise AssertionError("save must not be called for an existing config")

    def boom_open(*a, **k):
        raise AssertionError("template must not be opened for an existing config")

    monkeypatch.setattr(config_store, "save", boom_save)
    monkeypatch.setattr(app, "open", boom_open, raising=False)

    result = app.load_or_create_config()

    assert result is existing
