"""Cross-device wake arbitration: one phrase heard by several devices keeps one session.

docs/device-protocol.md, "Voice session": when several online devices wake on one
phrase, the server keeps exactly one session. Each wake run registers here at its wake
and is held by its pipeline until the verdict; its device is shown `listening` as soon as
its wake word is confirmed, not at the verdict, unless it has lost by then. A lost run
stays held — its device streams the rest of the phrase on — until its VAD end-points; it
then ends (with `state idle` once shown `listening`) and is recorded as a duplicate, its
phrase recognized for the run log only (Pipeline._end_held). A device
without a speaker that lost the phrase (in the group or in its cooldown) is shown the
winner's answer as an announcement, a `reply` without `session` (run_event).

A group of wakes is collected for WINDOW_MS (less
once every online device has woken; a lone online device is not held) and ruled on:
a device with a speaker over one without, between equals the higher signal-to-noise
ratio of the pre-roll, then the earlier wake. An online device that is no candidate
but whose follow-up window already hears speech — its VAD heard it at most COOLDOWN_MS
before the group's first wake, and the window's phrase had not ended by that wake —
keeps the phrase instead (a device sends no wake
inside its own phrase): every candidate loses to it as to a winner, and the
speakerless ones are shown its answer. For COOLDOWN_MS
after the group's first wake, a wake of any device but the group's winner loses at
once.

A candidate is ready once it no longer waits for the verdict (its verifier rejected
it, its run was cancelled, its device vanished), or once its wake-word verifier has
ruled (the verifier's own timeout bounds that) and — when there are several
candidates — its pre-roll is in or WINDOW_MS has passed since its own wake (its
SNR is then taken on what it has): the pre-roll streams in only after the `wake`.
Only a wake over the device's running answer waits for the verifier; any other has
its verdict at its wake, and its phrase is checked for the wake word only after the
ruling (Pipeline._run). So the winner is a wake the verifier did not reject — a false
wake not over an answer can win and start the cooldown — and a group whose candidates
all dropped out leaves no cooldown. The group is ruled once every candidate is ready
and either every online device has woken or the group's window has run out. A
candidate without a speaker is not waited for once a ready candidate has one: it
cannot outrank it, so it loses as it is. Its verifier still rules on it later: a
reject ends its run there and then (logged as wakeword_reject, untranscribed), not
at its end-point as a duplicate; either way it is shown the winner's answer.

Everything here is synchronous on the event loop. The group is re-checked, and a
verdict delivered by calling `pipeline.arbitration_verdict(token, won)`, from
inside `register()` (a pipeline's on_wake), `preroll_in()` (a pipeline's on_audio,
or the candidate's own window timer), `run_changed()` (a pipeline whose verifier
ruled, or whose run stopped waiting) or the group's window timer — so a verdict for
one pipeline may arrive inside a call made by another.
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

# How long a group of wakes waits for the wakes of the other online devices (ms).
WINDOW_MS = 500
# For this long after a group's first wake, a wake of any device but the group's
# winner loses as a duplicate (ms). It also bounds how long before the group's first
# wake a follow-up window's speech may have started for the window to keep the phrase.
COOLDOWN_MS = 2000


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
    # Fires WINDOW_MS after this run's own wake and settles it.
    timer: asyncio.TimerHandle
    # Its pre-roll is in, or WINDOW_MS has passed since its wake.
    settled: bool = False


class WakeArbiter:
    """Decides which of the devices that woke on one phrase keeps its session."""

    def __init__(self, online_count: Callable[[], int],
                 online_pipelines: Callable[[], list] = lambda: [],
                 send_reply: Callable | None = None):
        self._online_count = online_count
        # The online devices' pipelines: one whose follow-up window already hears the
        # phrase keeps it over the group (_close).
        self._online_pipelines = online_pipelines
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
        # The group's window: fires WINDOW_MS after its first wake.
        self._timer: asyncio.TimerHandle | None = None
        self._window_fired = False
        # Monotonic time of the latest group's first wake, and the pipeline that won
        # that group (exempt from its cooldown).
        self._group_start: float | None = None
        self._winner = None

    def register(self, pipeline, token) -> None:
        """Take a wake run of `pipeline` into arbitration. Its verdict arrives as
        pipeline.arbitration_verdict(token, won), possibly before this returns."""
        loop = asyncio.get_running_loop()
        if not self._open:
            now = time.monotonic()
            if (self._group_start is not None
                    and (now - self._group_start) * 1000 < COOLDOWN_MS
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
            self._timer = loop.call_later(WINDOW_MS / 1000, self._window_end)
        old = self._candidates.get(pipeline)
        if old is not None:
            old.timer.cancel()
        # A re-register of the same pipeline replaces its entry in place.
        self._candidates[pipeline] = _Candidate(
            token, loop.call_later(WINDOW_MS / 1000, self.preroll_in, pipeline, token))
        self._maybe_close()

    def preroll_in(self, pipeline, token) -> None:
        """The pre-roll of `pipeline`'s run registered under `token` is fully in — or,
        called by the candidate's own timer, WINDOW_MS has passed since its wake and
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
        """Rule once the verdict is settled and either every online device has woken
        or the window has run out. A lone candidate needs no SNR, so it does not wait
        for its pre-roll; every candidate waits for its verifier — except a speakerless
        one once a ready candidate has a speaker: it cannot outrank that one, so it
        loses without being waited for (a panel whose audio lags would otherwise hold
        the speaker's phrase for seconds)."""
        n = len(self._candidates)
        waiting = [(p, c) for p, c in self._candidates.items()
                   if p.awaits_arbitration(c.token)]
        ready = [(c.settled or n == 1) and not p.verifier_pending for p, c in waiting]
        unready = [p for (p, _), r in zip(waiting, ready) if not r]
        can_rule = not unready or (
            not any(p.has_speaker for p in unready)
            and any(p.has_speaker and r for (p, _), r in zip(waiting, ready)))
        if can_rule and (n >= self._online_count() or self._window_fired):
            self._close()

    def _followup_keeper(self, candidates):
        """(pipeline, its speech's monotonic start) of an online device, not a
        candidate of the group, whose follow-up window already hears speech that
        started at most COOLDOWN_MS before the group's first wake and was still going
        at that wake (a phrase that ends between the first wake and the ruling still
        counts); None when there is none."""
        since = self._group_start - COOLDOWN_MS / 1000
        for p in self._online_pipelines():
            heard = p.followup_speech()
            if p in candidates or heard is None:
                continue
            speech_at, ended_at = heard
            if speech_at >= since and (ended_at is None or ended_at >= self._group_start):
                return p, speech_at
        return None

    def _close(self) -> None:
        """Rule on the open group: of the candidates still waiting for this verdict,
        the best keeps its session and the rest lose — all of them lose to a device
        whose follow-up window already hears the phrase."""
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
        keeper = self._followup_keeper(candidates)
        if keeper is not None:
            winner, speech_at = keeper
            lead = self._group_start - speech_at
            kept = (f"{winner.name}'s follow-up window keeps the phrase (speech "
                    f"{abs(lead):.1f} s {'before' if lead >= 0 else 'after'} the first wake)")
        else:
            _, winner, _ = max(scored, key=lambda s: s[0])
            kept = f"{winner.name} keeps the phrase"
        self._winner = winner
        self._awaiting_answer = True
        self._answer_to = [p for _, p, _ in scored if p is not winner and not p.has_speaker]
        logger.info(
            "wake arbitration: "
            + ", ".join(
                f"{p.name} (speaker {'yes' if key[0] else 'no'}, SNR {key[1]:.1f} dB)"
                for key, p, _ in scored
            )
            + f" -> {kept}"
        )
        for _, p, token in scored:
            p.arbitration_verdict(token, p is winner)
