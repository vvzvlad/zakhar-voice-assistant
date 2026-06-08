import json
import os
from datetime import datetime

from src.context import append_context, load_context
from src.settings import settings


def test_first_append_creates_file_and_loads(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "привет", "ну привет, мешок")
    assert os.path.exists(path)
    assert load_context(path) == [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "ну привет, мешок"},
    ]
    # Each raw line must be valid JSON.
    raw_lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln]
    for ln in raw_lines:
        json.loads(ln)


def test_two_appends_within_ttl_accumulate(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "раз", "первый")
    append_context(path, "два", "второй")  # age < ttl -> accumulate
    assert load_context(path) == [
        {"role": "user", "content": "раз"},
        {"role": "assistant", "content": "первый"},
        {"role": "user", "content": "два"},
        {"role": "assistant", "content": "второй"},
    ]


def test_trimming_drops_oldest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "context_max_turns", 2)
    path = str(tmp_path / "context_living.txt")
    append_context(path, "u1", "a1")
    append_context(path, "u2", "a2")
    append_context(path, "u3", "a3")
    # Only the last 2*2=4 messages remain; the first exchange is dropped.
    assert load_context(path) == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_stale_file_resets(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "старый", "ответ")
    # Simulate the file being idle past the TTL.
    old = datetime.now().timestamp() - (settings.context_ttl_seconds + 60)
    os.utime(path, (old, old))
    assert load_context(path) == []
    # A following append starts fresh: file holds only the new exchange.
    append_context(path, "новый", "ответ2")
    assert load_context(path) == [
        {"role": "user", "content": "новый"},
        {"role": "assistant", "content": "ответ2"},
    ]


def test_load_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "does_not_exist.txt")
    assert load_context(path) == []


def test_legacy_lines_are_skipped(tmp_path):
    path = str(tmp_path / "context_living.txt")
    # Write a fresh file in the old legacy format.
    with open(path, "w", encoding="utf-8") as f:
        f.write("USER: x\nGLADOS: y\n")
    # Legacy non-JSON lines are skipped -> empty.
    assert load_context(path) == []
    # A following append produces valid JSONL.
    append_context(path, "вопрос", "ответ")
    assert load_context(path) == [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    for ln in [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln]:
        json.loads(ln)


def test_creates_parent_dir(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "context_living.txt")
    append_context(path, "u", "a")
    assert os.path.exists(path)
    assert load_context(path) == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]


def test_cyrillic_round_trips_without_ascii_escapes(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "привет, как дела?", "отлично, мешок")
    assert load_context(path) == [
        {"role": "user", "content": "привет, как дела?"},
        {"role": "assistant", "content": "отлично, мешок"},
    ]
    raw = open(path, encoding="utf-8").read()
    assert "\\u" not in raw  # Cyrillic is not ASCII-escaped in the raw file
