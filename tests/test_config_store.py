import json
import os
import stat

import pytest

from src import config_store


def test_round_trip(tmp_path):
    path = str(tmp_path / "config.json")
    doc = {"version": 1, "voice": "захар", "n": 5}
    config_store.save(doc, path)
    assert config_store.load(path) == doc


def test_missing_file_returns_empty(tmp_path):
    assert config_store.load(str(tmp_path / "nope.json")) == {}


def test_save_creates_parent_dir(tmp_path):
    path = str(tmp_path / "sub" / "dir" / "config.json")
    config_store.save({"a": 1}, path)
    assert os.path.exists(path)
    assert config_store.load(path) == {"a": 1}


def test_bak_written_on_overwrite(tmp_path):
    path = str(tmp_path / "config.json")
    config_store.save({"v": 1}, path)
    config_store.save({"v": 2}, path)
    assert config_store.load(path) == {"v": 2}
    # The previous file is preserved as .bak.
    assert config_store.load(path + ".bak") == {"v": 1}


def test_no_bak_on_first_write(tmp_path):
    path = str(tmp_path / "config.json")
    config_store.save({"v": 1}, path)
    assert not os.path.exists(path + ".bak")


def test_file_mode_is_0600(tmp_path):
    path = str(tmp_path / "config.json")
    config_store.save({"a": 1}, path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        config_store.load(str(path))


def test_unicode_preserved(tmp_path):
    path = str(tmp_path / "config.json")
    config_store.save({"city": "Москва"}, path)
    raw = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "Москва" in raw  # ensure_ascii=False keeps it readable
