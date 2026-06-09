import json
import os

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


def test_successful_overwrite_leaves_no_tmp(tmp_path):
    # After a normal overwrite there must be exactly the config (+ its .bak), never
    # a leftover *.tmp file.
    path = str(tmp_path / "config.json")
    config_store.save({"v": 1}, path)
    config_store.save({"v": 2}, path)
    files = os.listdir(tmp_path)
    assert "config.json" in files
    assert not any(name.endswith(".tmp") for name in files)


def test_failed_save_leaves_no_tmp_and_keeps_good_file(tmp_path, monkeypatch):
    # Write a good file first, then force the next save to blow up mid-write.
    path = str(tmp_path / "config.json")
    config_store.save({"v": "good"}, path)

    def boom(*args, **kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(config_store.json, "dump", boom)
    with pytest.raises(RuntimeError):
        config_store.save({"v": "new"}, path)

    # No temp garbage left behind, and the previous good file is intact.
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))
    assert config_store.load(path) == {"v": "good"}


def test_save_swallows_directory_fsync_oserror(tmp_path, monkeypatch):
    # The directory fsync is best-effort: os.open(parent, O_DIRECTORY) can fail on some
    # platforms. The OSError guard must NOT abort the write — only the directory open is
    # forced to raise here; the temp-file write path (os.fdopen) is untouched.
    path = str(tmp_path / "config.json")
    parent = str(tmp_path)
    real_open = os.open

    def fake_open(p, *args, **kwargs):
        # Only the directory fsync open (the parent dir) raises; everything else works.
        if os.path.abspath(p) == os.path.abspath(parent):
            raise OSError("cannot open directory for fsync")
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(config_store.os, "open", fake_open)

    # Save still succeeds and writes the file correctly despite the directory fsync error.
    config_store.save({"v": "ok"}, path)
    assert config_store.load(path) == {"v": "ok"}
    # And no temp garbage was left behind.
    assert not any(name.endswith(".tmp") for name in os.listdir(tmp_path))
