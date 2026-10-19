"""Device logs on the server's disk: the lines a device sends in `log` messages while its
core.devices[].log is on (docs/device-protocol.md, "Settings and diagnostics").

One file per device and local day, `<DATA_DIR>/device_logs/<device>/<YYYY-MM-DD>.log`. Each line
is the device's own (its timestamp counts ms since the device booted) behind the server's local
wall-clock time of arrival, `HH:MM:SS.mmm `. A day's file is deleted once that day ended more
than core.device_log_days days ago, so the last device_log_days × 24 hours stay on disk: checked
for a device when its new day's file is created, and for every device at startup.
"""

import os
import re
from datetime import date, datetime, timedelta

from src import config_store

_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.log")


def root() -> str:
    # DATA_DIR is read at call time (not bound at import) so tests can monkeypatch it.
    return os.path.join(config_store.DATA_DIR, "device_logs")


def append(device: str, text: str, days: int) -> None:
    """Append the lines of `text` to the device's file of today, each behind the time now; a new
    day's file prunes the device's files older than `days` days. A small synchronous write, as a
    capture's frame is."""
    now = datetime.now()
    directory = os.path.join(root(), device)
    path = os.path.join(directory, f"{now:%Y-%m-%d}.log")
    new = not os.path.exists(path)
    if new:
        os.makedirs(directory, exist_ok=True)
    stamp = f"{now:%H:%M:%S}.{now.microsecond // 1000:03d} "
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(f"{stamp}{line}\n" for line in text.splitlines())
    if new:
        prune(days, device)


def prune(days: int, device: str | None = None) -> None:
    """Delete the day files whose day ended more than `days` days ago: the device's, or every
    device's when `device` is None. Other files are left alone."""
    oldest = (date.today() - timedelta(days=days)).isoformat()  # the oldest day kept
    if device is not None:
        devices = [device]
    else:
        devices = os.listdir(root()) if os.path.isdir(root()) else []
    for name in devices:
        directory = os.path.join(root(), name)
        if not os.path.isdir(directory):
            continue
        for f in os.listdir(directory):
            m = _NAME_RE.fullmatch(f)
            if m is not None and m.group(1) < oldest:
                os.remove(os.path.join(directory, f))
