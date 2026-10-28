import builtins
import json
import os
from datetime import datetime

from src.context import append_context, append_spoken, load_context

# Defaults baked into context.load_context / append_context.
DEFAULT_TTL_SECONDS = 300


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


def test_tool_calls_precede_the_answer_round_by_round(tmp_path):
    # An answer that ran tools carries each round's calls and their results before the
    # reply, so «отменил» does not read as enough by itself, and a call made after
    # seeing an earlier round's result stays after it.
    path = str(tmp_path / "context_living.txt")
    rounds = [[{"name": "list_reminders", "args": {}, "result": "№3 помыть посуду"}],
              [{"name": "cancel_reminder", "args": {"id": 3}, "result": "Отменено."},
               {"name": "set_timer", "args": {"seconds": 60}, "result": "ok"}]]
    append_context(path, "отмени напоминание", "Отменил.", rounds=rounds)
    user, made1, res1, made2, res2, res3, reply = load_context(path)
    assert user == {"role": "user", "content": "отмени напоминание"}
    assert reply == {"role": "assistant", "content": "Отменил."}
    for made, results, calls in ((made1, [res1], rounds[0]), (made2, [res2, res3], rounds[1])):
        assert made["role"] == "assistant" and made["content"] == ""
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in made["tool_calls"]] == [(c["name"], c["args"]) for c in calls]
        assert [(r["role"], r["tool_call_id"], r["content"]) for r in results] == [
            ("tool", tc["id"], c["result"]) for tc, c in zip(made["tool_calls"], calls)]


def test_tool_calls_do_not_shrink_the_window(tmp_path):
    # The window keeps max_turns exchanges however many of them called tools, each
    # kept answer with its calls and results.
    path = str(tmp_path / "context_living.txt")
    rounds = [[{"name": "set_timer", "args": {"seconds": 60}, "result": "ok"}]]
    for i in range(4):
        append_context(path, f"u{i}", f"a{i}", max_turns=2, rounds=rounds)
    assert [(m["role"], m["content"]) for m in load_context(path, max_turns=2)] == [
        ("user", "u2"), ("assistant", ""), ("tool", "ok"), ("assistant", "a2"),
        ("user", "u3"), ("assistant", ""), ("tool", "ok"), ("assistant", "a3"),
    ]


def test_window_cut_before_a_request_keeps_its_answers_calls_whole(tmp_path):
    # A spoken note shifts the cut onto a request: the window opens with that answer's
    # calls, results and reply, never with a call missing its result.
    path = str(tmp_path / "context_living.txt")
    rounds = [[{"name": "set_timer", "args": {"seconds": 60}, "result": "ok"}]]
    append_context(path, "u1", "a1", max_turns=2, rounds=rounds)
    append_spoken(path, "Таймер", max_turns=2)
    append_context(path, "u2", "a2", max_turns=2)
    window = load_context(path, max_turns=2)
    assert [(m["role"], m["content"]) for m in window] == [
        ("assistant", ""), ("tool", "ok"), ("assistant", "a1"),
        ("user", "[На колонке была произнесена фраза: «Таймер»]"),
        ("user", "u2"), ("assistant", "a2"),
    ]
    assert window[1]["tool_call_id"] == window[0]["tool_calls"][0]["id"]


def test_trimming_drops_oldest(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "u1", "a1", max_turns=2)
    append_context(path, "u2", "a2", max_turns=2)
    append_context(path, "u3", "a3", max_turns=2)
    # Only the last 2*2=4 messages remain; the first exchange is dropped.
    assert load_context(path, max_turns=2) == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_stale_file_resets(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "старый", "ответ")
    # Simulate the file being idle past the TTL.
    old = datetime.now().timestamp() - (DEFAULT_TTL_SECONDS + 60)
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


def test_load_context_open_oserror_returns_empty(tmp_path, monkeypatch):
    # The file exists and is fresh (passes the exists + not-stale guard), so load_context
    # proceeds to open it for reading. If that open() raises OSError, load_context must
    # swallow it and return [] rather than propagating the error.
    path = str(tmp_path / "context_living.txt")
    append_context(path, "вопрос", "ответ")
    # Sanity: the file is present and NOT stale, so we really reach the read-open.
    assert os.path.exists(path)
    assert load_context(path) == [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]

    real_open = builtins.open

    def boom(file, *args, **kwargs):
        if str(file) == path:
            raise OSError("read boom")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom)
    # Does not raise; returns [] on the read OSError.
    assert load_context(path) == []


def test_append_context_write_oserror_does_not_raise(tmp_path, monkeypatch):
    # A write failure in append_context must be swallowed (logged, never raised): the
    # pipeline depends on context persistence being best-effort. Force os.makedirs to
    # raise OSError so the write path fails before the file is opened.
    path = str(tmp_path / "nested" / "context_living.txt")

    def boom(*args, **kwargs):
        raise OSError("mkdir boom")

    monkeypatch.setattr("src.context.os.makedirs", boom)
    # No exception escapes; the helper returns None.
    assert append_context(path, "вопрос", "ответ") is None
    # The write never happened, so no file was created.
    assert not os.path.exists(path)
