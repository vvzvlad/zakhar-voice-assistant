"""Wake-word dataset recording: named sets of numbered clips recorded on a device's beep.

A set is the directory `<DATA_DIR>/datasets/<name>/` of `<NNNN>.wav` clips. A recording
session is a server-side background task (a 200-take session runs ~15 minutes, longer than
an HTTP request should): per take the device beeps, then records a capture of `seconds`
(src/captures.py), and a whole capture is moved into the set as its next clip. Numbering
continues after the highest existing index, so later sessions extend a set. Takes run back
to back. A take whose capture failed or has a gap is deleted and recorded again: a gap (audio
lost to the link) does not count, while the device is offline the session waits for it, any
other failure counts, and TAKE_ATTEMPTS failures in a row of one take end the session; the
clips already written stay.
"""

import asyncio
import io
import os
import re
import zipfile

from loguru import logger

from src import config_store
from src.device_server import CaptureBusyError, DeviceOfflineError

MAX_TAKES = 1000
MAX_SECONDS = 10
DEFAULT_SECONDS = 3
TAKE_ATTEMPTS = 3
# How often a session waiting for an offline device tries the take again.
OFFLINE_RETRY_S = 1.0
# The pause after a failed take before it is tried again.
FAILED_TAKE_PAUSE_S = 5.0

# Unicode \w (str pattern), so a Cyrillic set name is valid; no "." or "/" -> no traversal.
_NAME_RE = re.compile(r"[\w-]{1,64}")
_CLIP_RE = re.compile(r"(\d+)\.wav")


def valid_name(name: str) -> bool:
    return _NAME_RE.fullmatch(name) is not None


def set_dir(name: str) -> str:
    # DATA_DIR is read at call time (not bound at import) so tests can monkeypatch it.
    return os.path.join(config_store.DATA_DIR, "datasets", name)


def _wavs(path: str) -> list[str]:
    return sorted(f for f in os.listdir(path) if f.endswith(".wav"))


def next_index(path: str) -> int:
    """One past the highest numbered clip in the set directory (1 for an empty set)."""
    indices = [int(m.group(1)) for f in os.listdir(path) if (m := _CLIP_RE.fullmatch(f))]
    return max(indices, default=0) + 1


def add_clip(name: str, wav_path: str) -> None:
    """Move the WAV at `wav_path` into the set as its next numbered clip.

    Synchronous on purpose: picking the index and moving the file happen without an await
    in between, so two sessions on the same set can never pick the same number.
    """
    path = set_dir(name)
    os.replace(wav_path, os.path.join(path, f"{next_index(path):04d}.wav"))


def list_sets() -> list[tuple[str, int]]:
    """Every set directory on disk as (name, WAV count), sorted by name."""
    root = os.path.join(config_store.DATA_DIR, "datasets")
    if not os.path.isdir(root):
        return []
    return [(e.name, len(_wavs(e.path)))
            for e in sorted(os.scandir(root), key=lambda e: e.name) if e.is_dir()]


def zip_set(name: str) -> bytes | None:
    """The set's WAVs as an uncompressed zip (WAV barely compresses); None if no such set."""
    path = set_dir(name)
    if not os.path.isdir(path):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in _wavs(path):
            zf.write(os.path.join(path, f), arcname=f)
    return buf.getvalue()


class DatasetJob:
    """One recording session. `state` walks recording (waiting while the device is offline)
    -> done|stopped|error; `done` counts the clips this session has written."""

    def __init__(self, name: str, device: str, takes: int, seconds: int):
        self.name = name
        self.device = device
        self.takes = takes
        self.seconds = seconds
        self.state = "recording"  # recording | waiting | done | stopped | error
        self.done = 0
        self.error: str | None = None
        self.stop_requested = False
        self.task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "device": self.device,
            "takes": self.takes,
            "seconds": self.seconds,
            "done": self.done,
            "error": self.error,
        }


class DatasetJobManager:
    """Runs recording sessions, at most one per device.

    Construct with the DeviceManager: a take beeps with its play_chime and records with
    Device.record_capture. Keeps the latest session of every set (running or finished)
    until the next session of that set starts.
    """

    def __init__(self, devices):
        self._devices = devices
        self._latest: dict[str, DatasetJob] = {}  # set name -> its latest session
        self._running: dict[str, DatasetJob] = {}  # device -> its in-flight session

    async def _take(self, job: DatasetJob) -> None:
        """Record one take into the set, again until its capture comes out whole or a stop
        is asked. A take that lost audio (a gap) or found the device offline is recorded
        again without counting; any other failure counts and pauses FAILED_TAKE_PAUSE_S (a
        voice session that made the device busy gets to end), and TAKE_ATTEMPTS of them in a
        row raise the last one."""
        failures = 0
        while not job.stop_requested:
            try:
                # The beep's result is ignored: a device without a speaker lands in
                # "offline" there and just records without the cue.
                await self._devices.play_chime("", job.device)
                cap = await self._devices.find(job.device).record_capture(job.seconds)
            except DeviceOfflineError:
                job.state = "waiting"
                await asyncio.sleep(OFFLINE_RETRY_S)
                continue
            except Exception as e:
                cap, error = None, str(e)
            job.state = "recording"
            if cap is not None:
                if cap.state == "done" and not cap.gaps:
                    # The sidecar goes first: a WAV left without one is not listed.
                    os.remove(cap.stem + ".json")
                    add_clip(job.name, cap.wav)
                    job.done += 1
                    return
                cap.remove()
                if cap.state == "done":
                    logger.info(f"dataset {job.name!r} on {job.device!r}: take {job.done + 1} "
                                f"lost audio ({len(cap.gaps)} gap(s)); recording it again")
                    continue
                error = cap.error
            failures += 1
            logger.warning(f"dataset {job.name!r} on {job.device!r}: take {job.done + 1} failed "
                           f"({failures}/{TAKE_ATTEMPTS}): {error}")
            if failures == TAKE_ATTEMPTS:
                raise RuntimeError(error)
            await asyncio.sleep(FAILED_TAKE_PAUSE_S)

    async def _run(self, job: DatasetJob) -> None:
        """Background task body: take after take until all are done or a stop is asked."""
        try:
            while job.done < job.takes and not job.stop_requested:
                await self._take(job)
            job.state = "done" if job.done == job.takes else "stopped"
        except asyncio.CancelledError:
            # Shutdown cancellation: never swallow it, the task must be observed cancelled.
            job.state = "stopped"
            raise
        except Exception as e:
            job.state = "error"
            job.error = str(e)
            logger.warning(f"dataset {job.name!r} on {job.device!r} failed at take {job.done + 1}: {e}")
        finally:
            del self._running[job.device]

    def start(self, name: str, device: str, takes: int, seconds: int) -> dict:
        """Start a session; return its initial snapshot.

        Creates the set directory up front, so the set is listed (with this session's
        outcome) even when the first take fails. Raises CaptureBusyError if a session is
        already running on the device.
        """
        if device in self._running:
            raise CaptureBusyError(f"{device} dataset recording already in progress")
        os.makedirs(set_dir(name), exist_ok=True)
        job = DatasetJob(name, device, takes, seconds)
        self._latest[name] = job
        self._running[device] = job
        job.task = asyncio.create_task(self._run(job))
        return job.snapshot()

    def stop(self, name: str) -> dict:
        """Ask the set's running session to end after the current take; return its snapshot.

        Raises LookupError if no session of that set is running.
        """
        jobs = [j for j in self._running.values() if j.name == name]
        if not jobs:
            raise LookupError(f"no dataset recording running for {name!r}")
        for j in jobs:
            j.stop_requested = True
        return jobs[-1].snapshot()

    def sets(self) -> list[dict]:
        """Every set on disk with its clip count and its latest session (or None)."""
        out = []
        for name, clips in list_sets():
            job = self._latest.get(name)
            out.append({"name": name, "clips": clips, "job": job.snapshot() if job else None})
        return out

    async def close(self) -> None:
        """Cancel and await all running sessions (used on server shutdown)."""
        tasks = [j.task for j in self._running.values() if j.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
