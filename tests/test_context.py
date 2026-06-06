import os
from datetime import datetime

from src.context import append_context


def test_first_append_creates_file(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "привет", "ну привет, мешок")
    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert content == "USER: привет\nGLADOS: ну привет, мешок\n"


def test_second_immediate_append_appends(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "раз", "первый")
    append_context(path, "два", "второй")  # age < 60s → append
    content = open(path, encoding="utf-8").read()
    assert content == (
        "USER: раз\nGLADOS: первый\n"
        "USER: два\nGLADOS: второй\n"
    )


def test_old_file_is_reset(tmp_path):
    path = str(tmp_path / "context_living.txt")
    append_context(path, "старый", "ответ")
    # Simulate the file being older than 60 seconds.
    old = datetime.now().timestamp() - 120
    os.utime(path, (old, old))
    append_context(path, "новый", "ответ2")  # age > 60s → truncate first
    content = open(path, encoding="utf-8").read()
    assert content == "USER: новый\nGLADOS: ответ2\n"


def test_creates_parent_dir(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "context_living.txt")
    append_context(path, "u", "a")
    assert os.path.exists(path)
    assert open(path, encoding="utf-8").read() == "USER: u\nGLADOS: a\n"
