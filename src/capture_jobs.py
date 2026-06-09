"""Background capture jobs: decouple device recording from the browser request.

The admin panel's "Capture sample" used to be a single long-lived POST that held
the HTTP connection open for the WHOLE recording (up to CAPTURE_MAX_SECONDS) and
returned the WAV in the response body. That coupling was the root cause of two
field bugs:

  * Closing the browser mid-recording aborted the request -> aiohttp cancelled the
    handler -> CancelledError tore through DeviceClient.capture() and left the
    device's voice_assistant run broken mid-stream -> the speaker emergency-rebooted.
  * After accidentally closing the modal there was no way to release a device that
    was still "busy" with an in-flight capture.

This module runs the recording as a SERVER-SIDE background asyncio task. The
browser only starts it (202), polls its status (for the live countdown),
downloads the result, or cancels it. None of those calls is tied to the
recording's lifetime, so closing the browser no longer cancels the recording ->
no device reboot.

IMPORTANT firmware fact: the device self-times the recording entirely in firmware
(button press -> stop wake word -> voice_assistant.start -> delay(seconds) ->
voice_assistant.stop -> start wake word). The server CANNOT interrupt the device
mid-recording. So "cancel" only means: stop waiting for / discard the result and
reset the UI. The device finishes its physical recording on its own in the
background; the pipeline drains and emits RUN_END cleanly (no reboot). New
captures on the same device stay blocked until that physical window has elapsed.
"""

import asyncio
import time

from loguru import logger

from src.pipeline import CaptureBusyError

# How long a terminal job (done/error/cancelled) is retained after it finishes so
# the browser can still read its final status / download the WAV before it is
# evicted. After this it is dropped and a new capture on the device is allowed.
RESULT_TTL = 180.0


class CaptureJob:
    """One background capture for a single device.

    `seconds` is the firmware-timed recording duration; `started_at`/`finished_at`
    use time.monotonic(). `state` walks recording -> done|error|cancelled.
    """

    def __init__(self, device: str, seconds: int, started_at: float):
        self.device = device
        self.seconds = seconds
        self.started_at = started_at
        self.state = "recording"  # recording | done | error | cancelled
        self.wav: bytes | None = None
        self.error: str | None = None
        self.task: asyncio.Task | None = None
        self.finished_at: float | None = None
        self.cancel_requested = False

    def remaining(self, now: float) -> int:
        """Whole seconds left in the (firmware-timed) recording, clamped at 0."""
        return max(0, int(round(self.seconds - (now - self.started_at))))

    def snapshot(self, now: float) -> dict:
        """Serializable status for the browser; `remaining` drives the countdown.

        The countdown is only meaningful while the device is physically recording
        (state recording or cancelled — the device is still draining); for any
        terminal-success/error state it is 0.
        """
        countdown = self.remaining(now) if self.state in ("recording", "cancelled") else 0
        return {
            "state": self.state,
            "device": self.device,
            "seconds": self.seconds,
            "remaining": countdown,
            "error": self.error,
        }


class CaptureJobManager:
    """Owns at most one in-flight capture job PER device.

    Construct with the async capture callable `(device, seconds) -> wav bytes`
    (DeviceManager.capture). All public methods are synchronous except `_run`
    (the background task body) and `close` (shutdown).
    """

    def __init__(self, capture_fn):
        self._capture_fn = capture_fn
        self._jobs: dict[str, CaptureJob] = {}  # device -> job

    # --- internals -----------------------------------------------------------
    def _evict_stale(self, now: float) -> None:
        """Drop terminal jobs (task finished) older than RESULT_TTL."""
        for device, job in list(self._jobs.items()):
            done = job.task is None or job.task.done()
            if done and job.finished_at is not None and (now - job.finished_at) > RESULT_TTL:
                del self._jobs[device]

    def _active(self, device: str) -> CaptureJob | None:
        """The device's job iff its task exists and is NOT done, else None."""
        job = self._jobs.get(device)
        if job is not None and job.task is not None and not job.task.done():
            return job
        return None

    async def _run(self, job: CaptureJob) -> None:
        """Background task body: run the capture and record the outcome on `job`."""
        try:
            wav = await self._capture_fn(job.device, job.seconds)
        except asyncio.CancelledError:
            # Shutdown cancellation: mark cancelled and re-raise so the task is
            # observed as cancelled (important — never swallow CancelledError).
            job.state = "cancelled"
            job.finished_at = time.monotonic()
            raise
        except Exception as e:
            job.state = "error"
            job.error = str(e)
            job.finished_at = time.monotonic()
            logger.warning(f"capture failed for {job.device!r}: {e}")
            return
        job.finished_at = time.monotonic()
        if job.cancel_requested:
            # The operator cancelled while the device kept recording; discard the
            # now-arrived result. The device drained cleanly on its own (no reboot).
            job.state = "cancelled"
            job.wav = None
            logger.info(f"capture for {job.device!r} finished but was cancelled; discarding result")
        else:
            job.state = "done"
            job.wav = wav

    # --- public API ----------------------------------------------------------
    def start(self, device: str, seconds: int) -> dict:
        """Start a background capture; return its initial snapshot.

        Raises CaptureBusyError if a capture is already in progress on the device.
        """
        now = time.monotonic()
        self._evict_stale(now)
        if self._active(device) is not None:
            raise CaptureBusyError(f"{device} capture already in progress")
        job = CaptureJob(device, seconds, now)
        job.task = asyncio.create_task(self._run(job))
        self._jobs[device] = job
        return job.snapshot(now)

    def status(self, device: str) -> dict:
        """Current status for a device; an idle dict when there is no live job.

        A cancelled job whose task has finished (the device has drained) is removed
        and reported as idle, so a fresh capture can start.
        """
        now = time.monotonic()
        self._evict_stale(now)
        job = self._jobs.get(device)
        if job is None:
            return self._idle(device)
        if job.state == "cancelled" and (job.task is None or job.task.done()):
            del self._jobs[device]
            return self._idle(device)
        return job.snapshot(now)

    def cancel(self, device: str) -> dict:
        """Cancel an in-flight capture (discard the eventual result) or clear a job.

        The device self-times its recording and cannot be interrupted, so an active
        job is only FLAGGED cancelled — it keeps draining and the result is dropped
        on arrival. A terminal job is simply deleted. Returns the resulting status.
        """
        now = time.monotonic()
        self._evict_stale(now)
        job = self._jobs.get(device)
        if job is None:
            return self._idle(device)
        if job.task is not None and not job.task.done():
            job.cancel_requested = True
            if job.state == "recording":
                job.state = "cancelled"
            return job.snapshot(now)
        # Terminal job: drop it so the device is free again.
        del self._jobs[device]
        return self._idle(device)

    def take_result(self, device: str):
        """One-shot: pop a completed capture's WAV. Returns (wav, seconds) or None."""
        job = self._jobs.get(device)
        if job is None or job.state != "done" or job.wav is None:
            return None
        wav, seconds = job.wav, job.seconds
        del self._jobs[device]  # one-shot download: consume the result
        return wav, seconds

    async def close(self) -> None:
        """Cancel and await all active capture tasks (used on server shutdown)."""
        tasks = [job.task for job in self._jobs.values() if job.task is not None and not job.task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _idle(device: str) -> dict:
        return {"state": "idle", "device": device, "seconds": 0, "remaining": 0, "error": None}
