"""Unit tests for src.prompt_store.PromptStore (named system-prompt profiles)."""

import pytest

from src.prompt_store import DEFAULT_PROMPT_PATH, PromptStore


def _store(tmp_path, seed_text="seed body", name="prompts.db"):
    """A store seeded from a legacy file holding `seed_text`."""
    legacy = tmp_path / "system_prompt.md"
    legacy.write_text(seed_text, encoding="utf-8")
    return PromptStore(str(tmp_path / name), seed_path=str(legacy))


def _active_ids(store):
    return [p["id"] for p in store.list_profiles() if p["is_active"]]


# --- seeding ------------------------------------------------------------------

def test_seeds_default_profile_from_legacy_file(tmp_path):
    store = _store(tmp_path, "legacy prompt body")
    profiles = store.list_profiles()
    assert [p["name"] for p in profiles] == ["default"]
    assert profiles[0]["is_active"] is True
    assert profiles[0]["chars"] == len("legacy prompt body")
    # The list payload carries no full text; get() does.
    assert "text" not in profiles[0]
    assert store.get(profiles[0]["id"])["text"] == "legacy prompt body"
    # The legacy file is kept on disk as a safety backup.
    assert (tmp_path / "system_prompt.md").read_text(encoding="utf-8") == "legacy prompt body"
    store.close()


def test_seeds_default_profile_from_template_when_no_legacy_file(tmp_path):
    # No legacy file at seed_path -> the committed default template is used.
    store = PromptStore(str(tmp_path / "prompts.db"),
                        seed_path=str(tmp_path / "missing.md"))
    with open(DEFAULT_PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()
    active = store.active()
    assert active["name"] == "default"
    assert active["text"] == template
    store.close()


def test_does_not_reseed_existing_db(tmp_path):
    # Re-opening an already-seeded DB must not insert a second "default" row,
    # even when the legacy seed file changed in the meantime.
    store = _store(tmp_path, "first")
    store.close()
    (tmp_path / "system_prompt.md").write_text("second", encoding="utf-8")
    reopened = PromptStore(str(tmp_path / "prompts.db"),
                           seed_path=str(tmp_path / "system_prompt.md"))
    assert len(reopened.list_profiles()) == 1
    assert reopened.active_text() == "first"
    reopened.close()


# --- create / get / update / delete -------------------------------------------

def test_create_and_get(tmp_path):
    store = _store(tmp_path)
    created = store.create("work", "work prompt")
    assert created["name"] == "work"
    assert created["text"] == "work prompt"
    assert created["is_active"] is False
    assert created["created_ts"] is not None
    assert created["updated_ts"] is not None
    assert store.get(created["id"]) == created
    # Listing is ordered by name and includes both rows.
    assert [p["name"] for p in store.list_profiles()] == ["default", "work"]
    store.close()


def test_get_unknown_id_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get(999) is None
    store.close()


def test_create_duplicate_name_raises_value_error(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create("default", "x")
    store.close()


def test_update_partial_and_stamps_updated_ts(tmp_path):
    store = _store(tmp_path)
    created = store.create("work", "v1")
    upd = store.update(created["id"], text="v2")
    assert upd["text"] == "v2"
    assert upd["name"] == "work"  # untouched by a text-only update
    assert upd["updated_ts"] >= created["updated_ts"]
    renamed = store.update(created["id"], name="job")
    assert renamed["name"] == "job"
    assert renamed["text"] == "v2"
    store.close()


def test_update_unknown_id_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.update(999, text="x") is None
    store.close()


def test_update_duplicate_name_raises_value_error(tmp_path):
    store = _store(tmp_path)
    created = store.create("work", "x")
    with pytest.raises(ValueError):
        store.update(created["id"], name="default")
    store.close()


def test_delete_inactive_profile(tmp_path):
    store = _store(tmp_path)
    created = store.create("work", "x")
    assert store.delete(created["id"]) is True
    assert store.get(created["id"]) is None
    assert store.delete(created["id"]) is False  # already gone
    store.close()


def test_delete_active_profile_is_refused(tmp_path):
    store = _store(tmp_path)
    active_id = store.active()["id"]
    with pytest.raises(ValueError, match="cannot delete the active profile"):
        store.delete(active_id)
    # The row is still there and still active.
    assert store.active()["id"] == active_id
    store.close()


# --- activation invariant -------------------------------------------------------

def test_exactly_one_active_across_activations(tmp_path):
    store = _store(tmp_path, "default body")
    a = store.create("a", "A")
    b = store.create("b", "B")
    assert len(_active_ids(store)) == 1  # seeded default

    assert store.activate(a["id"]) is True
    assert _active_ids(store) == [a["id"]]
    assert store.active_text() == "A"

    assert store.activate(b["id"]) is True
    assert _active_ids(store) == [b["id"]]
    assert store.active_text() == "B"

    # Re-activating the already-active profile keeps the invariant.
    assert store.activate(b["id"]) is True
    assert _active_ids(store) == [b["id"]]
    store.close()


def test_activate_unknown_id_returns_false_and_keeps_active(tmp_path):
    store = _store(tmp_path)
    before = store.active()["id"]
    assert store.activate(999) is False
    assert store.active()["id"] == before
    store.close()


# --- active_text fallback --------------------------------------------------------

def test_active_text_falls_back_to_template_when_no_active_row(tmp_path):
    # Defensive path: the invariant guarantees an active row, but if the DB was
    # tampered with, active_text() must still return the default template.
    store = _store(tmp_path)
    store._conn.execute("UPDATE prompt_profiles SET is_active = 0")
    store._conn.commit()
    with open(DEFAULT_PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()
    assert store.active() is None
    assert store.active_text() == template
    store.close()
