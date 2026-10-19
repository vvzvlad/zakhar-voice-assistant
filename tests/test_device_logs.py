"""Device log files: a day's file goes once that day ended more than device_log_days days ago,
for a device when its new day's file is created and for every device at startup; the days on
disk are listed, and a day is read in whole lines from a byte offset."""

from datetime import date, timedelta

from src import device_logs


def day_file(tmp_path, device, days_ago):
    path = tmp_path / "device_logs" / device / f"{date.today() - timedelta(days=days_ago):%Y-%m-%d}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")
    return path


def test_a_new_day_prunes_the_device_and_startup_prunes_every_device(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    kept, gone = day_file(tmp_path, "kitchen", 7), day_file(tmp_path, "kitchen", 8)
    other = (tmp_path / "device_logs" / "kitchen" / "notes.txt")
    other.write_text("mine", encoding="utf-8")
    hall = day_file(tmp_path, "hall", 30)

    # Today's first lines create today's file: the kitchen's day 8 days ago goes, the
    # hall's files wait for their own new day or the startup prune.
    device_logs.append("kitchen", "I (1) a: b\n", 7)
    assert kept.exists() and not gone.exists() and other.exists() and hall.exists()
    # A later append to the same day prunes nothing.
    gone = day_file(tmp_path, "kitchen", 9)
    device_logs.append("kitchen", "I (2) a: c\n", 7)
    assert gone.exists()

    device_logs.prune(7)
    assert not gone.exists() and not hall.exists() and kept.exists() and other.exists()
    device_logs.prune(0)  # only today stays
    assert not kept.exists()
    assert sorted(p.name for p in (tmp_path / "device_logs" / "kitchen").iterdir()) == [
        f"{date.today():%Y-%m-%d}.log", "notes.txt"]


def test_prune_without_any_device_log(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    device_logs.prune(7)
    device_logs.prune(7, "kitchen")


def test_days_list_each_device_with_day_files_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    assert device_logs.days() == {}
    for days_ago in (2, 0, 1):
        day_file(tmp_path, "kitchen", days_ago)
    day_file(tmp_path, "hall", 5)
    (tmp_path / "device_logs" / "kitchen" / "notes.txt").write_text("mine", encoding="utf-8")
    (tmp_path / "device_logs" / "porch").mkdir()  # no day files: left out
    assert device_logs.days() == {
        "hall": [f"{date.today() - timedelta(days=5):%Y-%m-%d}"],
        "kitchen": [f"{date.today() - timedelta(days=n):%Y-%m-%d}" for n in (0, 1, 2)],
    }


def test_read_gives_whole_lines_from_an_offset(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    path = day_file(tmp_path, "kitchen", 0)
    day = path.stem
    whole = "10:00:00.000 \x1b[0;32mI (1) wifi: связь\x1b[0m\n10:00:01.000 I (2) a: b\n"
    path.write_bytes((whole + "10:00:02.000 I (3) a: par").encode())

    # The line still being written is left for the next read.
    text, offset = device_logs.read("kitchen", day, 0)
    assert (text, offset) == (whole, len(whole.encode()))
    with open(path, "ab") as f:
        f.write(b"tial\n")
    assert device_logs.read("kitchen", day, offset) == (
        "10:00:02.000 I (3) a: partial\n", path.stat().st_size)
    assert device_logs.read("kitchen", day, path.stat().st_size) == ("", path.stat().st_size)
    # An offset past the end reads from the start.
    assert device_logs.read("kitchen", day, path.stat().st_size + 1)[0].startswith(whole)

    yesterday = f"{date.today() - timedelta(days=1):%Y-%m-%d}"
    for device, name in (("hall", day), ("kitchen", yesterday), ("..", f"kitchen/{day}"),
                         ("kitchen", f"../kitchen/{day}"), ("kitchen", "notes")):
        assert device_logs.read(device, name, 0) is None
