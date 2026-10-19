"""Device log files: a day's file goes once that day ended more than device_log_days days ago,
for a device when its new day's file is created and for every device at startup."""

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
