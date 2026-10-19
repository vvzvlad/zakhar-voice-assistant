"""Cross-device wake arbitration: one phrase heard by several devices keeps one session.

docs/device-protocol.md, "Voice session": when several online devices wake on one
phrase, the server keeps exactly one session; the others never get `listening`. Each
wake run registers here and is held by its pipeline until the verdict. A lost run
stays held — its device streams the rest of the phrase on silently — until its VAD
end-points; it then ends with `state idle` and is recorded as a duplicate, its
phrase recognized for the run log only (Pipeline.arbitration_verdict). A device
without a speaker that lost the phrase (in the group or in its cooldown) is shown the
winner's answer as an announcement, a `reply` without `session` (run_event).

A group of wakes is collected for `core.arbitration.window_ms` (less
once every online device has woken; a lone online device is not held) and ruled on:
a device with a speaker over one without, between equals the higher signal-to-noise
ratio of the pre-roll, then the earlier wake. For `core.arbitration.cooldown_ms`
after the group's first wake, a wake of any device but the group's winner loses at
once.

A candidate is ready once it no longer waits for the verdict (its verifier rejected
it, its run was cancelled, its device vanished), or once its wake-word verifier has
ruled (the verifier's own timeout bounds that) and — when there are several
candidates — its pre-roll is in or `window_ms` has passed since its own wake (its
SNR is then taken on what it has): the pre-roll streams in only after the `wake`.
So the winner is always a wake the verifier accepted, and a group whose candidates
all dropped out leaves no cooldown. The group is ruled once every candidate is ready
and either every online device has woken or the group's window has run out.

A device answering when its wake word fires holds that wake until its verifier rules
(the device server's pending wake) and registers only once the verifier confirms it.
While any online device holds a wake, the open group is not ruled: the held wake may
be the same phrase, and ruling without it would leave it, once confirmed, to the
cooldown after its answer was already stopped.

Everything here is synchronous on the event loop. The group is re-checked, and a
verdict delivered by calling `pipeline.arbitration_verdict(token, won)`, from
inside `register()` (a pipeline's on_start), `preroll_in()` (a pipeline's on_audio,
or the candidate's own window timer), `run_changed()` (a pipeline whose verifier
ruled, or whose run stopped waiting), `recheck()` (the device server, once a held
wake is rejected, dropped with its connection, or confirmed and its run registered)
or the group's window timer — so a verdict for one pipeline
may arrive inside a call made by another.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger

from src.pipeline_events import StageEvent
from src.vad import SAMPLE_RATE

# The SNR's analysis window: 100 ms of 16 kHz samples.
_SNR_WINDOW = SAMPLE_RATE // 10


def wake_snr_db(pcm: bytes) -> float:
    """Signal-to-noise ratio (dB) of a pre-roll (16 kHz int16 mono): the RMS level of
    the loudest 100 ms window over the 10th-percentile window's, which stands for the
    noise floor. 0.0 when there is less than one window."""
    n = len(pcm) // 2 // _SNR_WINDOW
    if n < 1:
        return 0.0
    samples = np.frombuffer(pcm[: n * _SNR_WINDOW * 2], dtype="<i2").astype(np.float64)
    rms = np.sqrt(np.mean(samples.reshape(n, _SNR_WINDOW) ** 2, axis=1))
    # Floored at one LSB so a digitally silent window reads -90 dBFS, not -inf.
    db = 20 * np.log10(np.maximum(rms, 1.0) / 32768)
    return float(db.max() - np.percentile(db, 10))


@dataclass
class _Candidate:
    """One wake run in the open group."""
    token: int
    # Fires window_ms after this run's own wake and settles it.
    timer: asyncio.TimerHandle
    # Its pre-roll is in, or window_ms has passed since its wake.
    settled: bool = False


class WakeArbiter:
    """Decides which of the devices that woke on one phrase keeps its session."""

    def __init__(self, runtime, online_count: Callable[[], int],
                 held_wakes: Callable[[], bool] = lambda: False,
                 send_reply: Callable | None = None):
        # Window and cooldown are read live off runtime.core.arbitration on every wake.
        self.rt = runtime
        self._online_count = online_count
        # Whether an online device holds a wake over its running answer until its
        # wake-word check rules (the device server's pending wake); the open group is
        # not ruled meanwhile.
        self._held_wakes = held_wakes
        # send_reply(pipeline, text) shows text on that pipeline's device as an
        # announcement; the winner's answer goes to the speakerless devices that lost
        # its phrase (run_event).
        self._send_reply = send_reply
        # While the latest group's winner has not answered yet: the speakerless
        # pipelines that lost its phrase, in the group or in its cooldown.
        self._awaiting_answer = False
        self._answer_to: list = []
        # The open group's candidates, pipeline -> _Candidate, in arrival order.
        self._candidates: dict = {}
        self._open = False
        # The group's window: fires window_ms after its first wake.
        self._timer: asyncio.TimerHandle | None = None
        self._window_fired = False
        # Monotonic time of the latest group's first wake, and the pipeline that won
        # that group (exempt from its cooldown).
        self._group_start: float | None = None
        self._winner = None

    def register(self, pipeline, token) -> None:
        """Take a wake run of `pipeline` into arbitration. Its verdict arrives as
        pipeline.arbitration_verdict(token, won), possibly before this returns."""
        cfg = self.rt.core.arbitration
        loop = asyncio.get_running_loop()
        if not self._open:
            now = time.monotonic()
            if (self._group_start is not None
                    and (now - self._group_start) * 1000 < cfg.cooldown_ms
                    and pipeline is not self._winner):
                if self._awaiting_answer and not pipeline.has_speaker:
                    self._answer_to.append(pipeline)
                pipeline.arbitration_verdict(token, False)
                return
            self._group_start = now
            self._winner = None
            self._awaiting_answer = False
            self._answer_to = []
            self._open = True
            self._window_fired = False
            self._timer = loop.call_later(cfg.window_ms / 1000, self._window_end)
        old = self._candidates.get(pipeline)
        if old is not None:
            old.timer.cancel()
        # A re-register of the same pipeline replaces its entry in place.
        self._candidates[pipeline] = _Candidate(
            token, loop.call_later(cfg.window_ms / 1000, self.preroll_in, pipeline, token))
        self._maybe_close()

    def preroll_in(self, pipeline, token) -> None:
        """The pre-roll of `pipeline`'s run registered under `token` is fully in — or,
        called by the candidate's own timer, window_ms has passed since its wake and
        its SNR is taken on whatever pre-roll it has. The verdict may be delivered
        before this returns, to this pipeline too."""
        candidate = self._candidates.get(pipeline)
        if candidate is None or candidate.token != token or candidate.settled:
            return
        candidate.settled = True
        candidate.timer.cancel()
        self._maybe_close()

    def run_changed(self, pipeline, token) -> None:
        """The wake-word verifier ruled on `pipeline`'s run registered under `token`,
        or the run stopped waiting for the verdict: re-check the group."""
        candidate = self._candidates.get(pipeline)
        if candidate is not None and candidate.token == token:
            self._maybe_close()

    def recheck(self) -> None:
        """A held wake was rejected, dropped with its connection, or confirmed and its run
        registered: re-check the open group it kept from being ruled. Without an open group there is nothing to rule
        (and ruling an empty one would end the latest group's cooldown)."""
        if self._open:
            self._maybe_close()

    def run_event(self, pipeline, event, data) -> None:
        """A stage event of `pipeline`. The latest winner's first answer (TTS_START) is
        sent to the speakerless devices that lost its phrase, once; that answer, or its
        run ending without one (RUN_END, ERROR), ends the wait, so a later run of the
        winner (a follow-up) is never forwarded."""
        if pipeline is not self._winner or not self._awaiting_answer:
            return
        if event is StageEvent.TTS_START and self._send_reply is not None:
            for loser in self._answer_to:
                self._send_reply(loser, data.get("text", ""))
        if event in (StageEvent.TTS_START, StageEvent.RUN_END, StageEvent.ERROR):
            self._awaiting_answer = False
            self._answer_to = []

    def _window_end(self) -> None:
        self._timer = None
        self._window_fired = True
        self._maybe_close()

    def _maybe_close(self) -> None:
        """Rule once every candidate is ready and either every online device has
        woken or the window has run out, and no device holds a wake. A lone candidate
        needs no SNR, so it does not wait for its pre-roll; every candidate waits for
        its verifier."""
        n = len(self._candidates)
        ready = all(
            not p.awaits_arbitration(c.token)
            or ((c.settled or n == 1) and not p.verifier_pending)
            for p, c in self._candidates.items()
        ) and (n >= self._online_count() or self._window_fired)
        if ready and not self._held_wakes():
            self._close()

    def _close(self) -> None:
        """Rule on the open group: of the candidates still waiting for this verdict,
        the best keeps its session and the rest lose."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        for c in self._candidates.values():
            c.timer.cancel()
        self._open = False
        candidates, self._candidates = self._candidates, {}
        scored = [
            ((p.has_speaker, wake_snr_db(p.preroll_pcm), -i), p, c.token)
            for i, (p, c) in enumerate(candidates.items())
            if p.awaits_arbitration(c.token)
        ]
        if not scored:
            # Nobody kept the phrase, so no cooldown: the next wake opens a new group.
            self._group_start = None
            return
        _, winner, _ = max(scored, key=lambda s: s[0])
        self._winner = winner
        self._awaiting_answer = True
        self._answer_to = [p for _, p, _ in scored if p is not winner and not p.has_speaker]
        logger.info(
            "wake arbitration: "
            + ", ".join(
                f"{p.name} (speaker {'yes' if key[0] else 'no'}, SNR {key[1]:.1f} dB)"
                for key, p, _ in scored
            )
            + f" -> {winner.name} keeps the phrase"
        )
        for _, p, token in scored:
            p.arbitration_verdict(token, p is winner)
