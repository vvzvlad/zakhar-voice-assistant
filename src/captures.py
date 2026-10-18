"""Device captures on the server's disk (docs/device-protocol.md, "Capture").

A capture is `<DATA_DIR>/captures/<device>/<YYYYmmdd-HHMMSS>-<id>.wav` (mono s16le 16 kHz behind
a 44-byte header) plus a `<same>.json` sidecar {id, device, seconds, started, base, gaps, state,
error}. The audio is written at its offset as the frames arrive, so neither a lost link nor a
server restart loses what already came, and the written end is the file's size, so it survives a
restart too. A capture stays until it is deleted.

`base` is the frame of the file the device's offset 0 lands on: 0, or later once the device
restarted mid-capture and records the time left as a new segment under a new id. A write past
the written end leaves a zero-filled hole, kept in `gaps` as [start_frame, frames].
`started` is the unix time the capture was created. States: starting (no audio yet) ->
recording (from the first frame) -> done | error.
"""

import asyncio
import json
import os
import random
import re
import struct
import time

from src import config_store

RATE = 16000
HEADER_BYTES = 44
RUNNING = ("starting", "recording")

_NAME_RE = re.compile(r"\d{8}-\d{6}-\d+")
_FIELDS = ("id", "device", "seconds", "started", "base", "gaps", "state", "error")


def root() -> str:
    # DATA_DIR is read at call time (not bound at import) so tests can monkeypatch it.
    return os.path.join(config_store.DATA_DIR, "captures")


def new_id() -> int:
    return random.randint(1, 2**31 - 1)


def wav_header(frames: int) -> bytes:
    """The 44-byte header of a mono 16-bit 16 kHz WAV holding `frames` frames."""
    data = frames * 2
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data, b"WAVE", b"fmt ", 16, 1, 1,
                       RATE, RATE * 2, 2, 16, b"data", data)


class Capture:
    """One capture: its files and its sidecar fields. `end` is the written end in frames."""

    def __init__(self, stem: str, meta: dict):
        self.stem = stem  # the path of both files without the extension
        for field in _FIELDS:
            setattr(self, field, meta[field])
        self.end = 0
        self.finished = asyncio.Event()
        self._fd: int | None = None

    @classmethod
    def create(cls, device: str, seconds: int) -> "Capture":
        started = time.time()
        cid = new_id()
        directory = os.path.join(root(), device)
        os.makedirs(directory, exist_ok=True)
        stem = os.path.join(directory, time.strftime("%Y%m%d-%H%M%S", time.localtime(started)) + f"-{cid}")
        cap = cls(stem, {"id": cid, "device": device, "seconds": seconds, "started": started,
                         "base": 0, "gaps": [], "state": "starting", "error": None})
        cap._fd = os.open(cap.wav, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.pwrite(cap._fd, wav_header(0), 0)
        cap.save()
        return cap

    @classmethod
    def load(cls, stem: str) -> "Capture":
        with open(stem + ".json", encoding="utf-8") as f:
            cap = cls(stem, json.load(f))
        cap.end = max(0, (os.path.getsize(cap.wav) - HEADER_BYTES) // 2)
        return cap

    @property
    def name(self) -> str:
        return os.path.basename(self.stem)

    @property
    def wav(self) -> str:
        return self.stem + ".wav"

    @property
    def running(self) -> bool:
        return self.state in RUNNING

    def _fileno(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.wav, os.O_RDWR)
        return self._fd

    def save(self) -> None:
        tmp = self.stem + ".json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({field: getattr(self, field) for field in _FIELDS}, f)
        os.replace(tmp, self.stem + ".json")

    def write(self, offset: int, pcm: bytes) -> None:
        """Write mono samples whose first frame is `offset` frames into the current segment."""
        start = self.base + offset
        changed = self.state == "starting"
        self.state = "recording"
        if start > self.end:
            self.gaps.append([self.end, start - self.end])
            changed = True
        os.pwrite(self._fileno(), pcm, HEADER_BYTES + 2 * start)
        self.end = max(self.end, start + len(pcm) // 2)
        if changed:
            self.save()

    def segment(self, now: float) -> None:
        """The device lost the capture (it restarted): the time left goes on as a new segment
        under a new id, placed by the wall clock. The hole before it becomes a gap with the
        segment's first frame (or the tail gap at finish)."""
        self.id = new_id()
        self.base = max(self.end, int((now - self.started) * RATE))
        self.save()

    def finish(self, error: str | None = None) -> None:
        """End the capture: `done` with whatever audio came (a missing tail is a gap), or
        `error` with `error` or with no audio at all. Fixes the header's sizes."""
        if not self.running:
            return
        if error is None and self.end == 0:
            error = "no audio"
        total = self.seconds * RATE
        if error is None and self.end < total:
            self.gaps.append([self.end, total - self.end])
        fd = self._fileno()
        os.pwrite(fd, wav_header(self.end), 0)
        os.close(fd)
        self._fd = None
        self.state, self.error = ("error", error) if error else ("done", None)
        self.save()
        self.finished.set()

    def remove(self) -> None:
        # The sidecar goes first: a WAV left without one is not listed.
        for path in (self.stem + ".json", self.wav):
            if os.path.exists(path):
                os.remove(path)

    def snapshot(self) -> dict:
        """The capture for the panel API; `remaining` counts down while it runs."""
        remaining = max(0, round(self.started + self.seconds - time.time())) if self.running else 0
        return {
            "state": self.state,
            "device": self.device,
            "name": self.name,
            "seconds": self.seconds,
            "remaining": remaining,
            "duration": round(self.end / RATE, 3),
            "gaps": [{"at": round(start / RATE, 3), "seconds": round(frames / RATE, 3)}
                     for start, frames in self.gaps],
            "error": self.error,
        }


def idle(device: str) -> dict:
    """The snapshot of a device that has no capture."""
    return {"state": "idle", "device": device, "name": None, "seconds": 0, "remaining": 0,
            "duration": 0, "gaps": [], "error": None}


def _valid_device(device: str) -> bool:
    """Whether `device` names a directory right under the captures directory."""
    return device not in ("", ".", "..") and not any(c in device for c in "/\\\0")


def valid(device: str, name: str) -> bool:
    """Whether `device` and `name` name a capture file without leaving the captures directory."""
    return _valid_device(device) and _NAME_RE.fullmatch(name) is not None


def of_device(device: str) -> list[Capture]:
    """The device's captures on disk, newest first. A capture that cannot be read is left out:
    one moved into a dataset between the listing and the read, or a sidecar without its WAV."""
    directory = os.path.join(root(), device)
    if not _valid_device(device) or not os.path.isdir(directory):
        return []
    caps = []
    for f in os.listdir(directory):
        if f.endswith(".json"):
            try:
                caps.append(Capture.load(os.path.join(directory, f[:-len(".json")])))
            except (OSError, ValueError, KeyError):
                continue
    return sorted(caps, key=lambda c: c.started, reverse=True)


def everything() -> list[Capture]:
    """Every capture on disk, newest first."""
    if not os.path.isdir(root()):
        return []
    caps = [c for e in os.scandir(root()) if e.is_dir() for c in of_device(e.name)]
    return sorted(caps, key=lambda c: c.started, reverse=True)


def newest(device: str, state: str | None = None) -> Capture | None:
    return next((c for c in of_device(device) if state is None or c.state == state), None)


def wav_bytes(cap: Capture) -> bytes:
    """The capture's WAV with a header fixed from the file's size, so an unfinished one is
    valid too."""
    with open(cap.wav, "rb") as f:
        f.seek(HEADER_BYTES)
        pcm = f.read()
    pcm = pcm[:len(pcm) - len(pcm) % 2]
    return wav_header(len(pcm) // 2) + pcm


def find(device: str, name: str) -> Capture | None:
    """The capture `name` of `device` on disk, None when there is none or the names are invalid."""
    if not valid(device, name):
        return None
    stem = os.path.join(root(), device, name)
    return Capture.load(stem) if os.path.exists(stem + ".json") else None
