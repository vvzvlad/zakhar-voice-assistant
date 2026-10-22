"""The cross-device wake arbiter (src/wake_arbiter.py) against fake pipelines: which
device keeps a phrase several devices woke on, and when the verdicts arrive."""

import asyncio

import numpy as np

from src.wake_arbiter import WakeArbiter


def preroll(noise: int, speech: int) -> bytes:
    """2 s of pre-roll: 1.5 s at the noise level, then 0.5 s of wake word at the
    speech level; its SNR is 20·log10(speech / noise) dB."""
    return np.concatenate([np.full(24000, noise), np.full(8000, speech)]).astype("<i2").tobytes()


class FakePipeline:
    """The arbiter's side of a Pipeline: one wake run at a time, its verdicts recorded."""

    def __init__(self, name, *, speaker, pcm=b""):
        self.name = name
        self.has_speaker = speaker
        self.preroll_pcm = pcm
        self.verifier_pending = False
        self.token = 0
        self.waiting = False
        self.verdicts = []

    def wake(self, arbiter, preroll_in=True):
        """A wake, then (unless preroll_in is False) its whole pre-roll arriving."""
        self.token += 1
        self.waiting = True
        arbiter.register(self, self.token)
        if preroll_in:
            arbiter.preroll_in(self, self.token)

    def awaits_arbitration(self, token):
        return token == self.token and self.waiting

    def arbitration_verdict(self, token, won, together=False):
        self.verdicts.append(won)
        self.waiting = False


def make_arbiter(monkeypatch, online, *, window_ms=20):
    """An arbiter whose online-device count is online[0], mutable by the test, with the
    group window shrunk to window_ms."""
    monkeypatch.setattr("src.wake_arbiter.WINDOW_MS", window_ms)
    return WakeArbiter(online_count=lambda: online[0])


async def test_a_device_with_a_speaker_beats_one_without_even_with_a_lower_snr(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [2])
    panel = FakePipeline("panel", speaker=False, pcm=preroll(100, 3162))   # 30 dB
    speaker = FakePipeline("speaker", speaker=True, pcm=preroll(100, 316))  # 10 dB
    panel.wake(arbiter)
    speaker.wake(arbiter)
    assert (speaker.verdicts, panel.verdicts) == ([True], [False])


async def test_between_equal_devices_the_higher_snr_wins(monkeypatch):
    # Three devices online, two wake: the window runs out. The later, quieter device
    # has the cleaner wake word.
    arbiter = make_arbiter(monkeypatch, [3])
    loud = FakePipeline("loud", speaker=True, pcm=preroll(1000, 3162))   # 10 dB
    clean = FakePipeline("clean", speaker=True, pcm=preroll(100, 1000))  # 20 dB
    loud.wake(arbiter)
    clean.wake(arbiter)
    assert loud.verdicts == clean.verdicts == []
    await asyncio.sleep(0.1)
    assert (clean.verdicts, loud.verdicts) == ([True], [False])


async def test_a_lone_online_device_gets_its_verdict_at_once(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [1], window_ms=10000)
    speaker = FakePipeline("speaker", speaker=True)
    speaker.wake(arbiter, preroll_in=False)
    assert speaker.verdicts == [True]


async def test_the_window_closes_once_every_online_device_has_woken(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [2], window_ms=10000)
    first = FakePipeline("first", speaker=True, pcm=preroll(100, 1000))
    second = FakePipeline("second", speaker=True, pcm=preroll(100, 1000))
    first.wake(arbiter)
    assert first.verdicts == []
    second.wake(arbiter)
    # Equal devices, equal SNR: the earlier wake keeps the phrase.
    assert (first.verdicts, second.verdicts) == ([True], [False])


async def test_a_candidate_whose_pre_roll_never_comes_is_ruled_a_window_after_its_wake(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [2])
    first = FakePipeline("first", speaker=True, pcm=preroll(100, 1000))
    silent = FakePipeline("silent", speaker=True)
    first.wake(arbiter)
    silent.wake(arbiter, preroll_in=False)
    assert first.verdicts == silent.verdicts == []
    await asyncio.sleep(0.1)
    assert (first.verdicts, silent.verdicts) == ([True], [False])


async def test_the_cooldown_drops_another_devices_wake_but_not_the_winners(monkeypatch):
    online = [2]
    arbiter = make_arbiter(monkeypatch, online)
    speaker = FakePipeline("speaker", speaker=True)
    panel = FakePipeline("panel", speaker=False)
    speaker.wake(arbiter)
    panel.wake(arbiter)
    assert (speaker.verdicts, panel.verdicts) == ([True], [False])

    # The panel's late copy of the same phrase is dropped at once.
    panel.wake(arbiter)
    assert panel.verdicts == [False, False]
    # The winner's own next wake opens a new window instead.
    online[0] = 1
    speaker.wake(arbiter)
    assert speaker.verdicts == [True, True]


async def test_a_ready_speaker_does_not_wait_for_panels_whose_check_is_pending(monkeypatch):
    # The panels' audio lags, so their wake-word check has not ruled: neither can
    # outrank the speaker, which keeps the phrase without waiting for them.
    arbiter = make_arbiter(monkeypatch, [3], window_ms=10000)
    speaker = FakePipeline("speaker", speaker=True, pcm=preroll(100, 1000))
    panels = [FakePipeline(f"panel{i}", speaker=False) for i in (1, 2)]
    for panel in panels:
        panel.verifier_pending = True
        panel.wake(arbiter, preroll_in=False)
    speaker.wake(arbiter)
    assert (speaker.verdicts, [p.verdicts for p in panels]) == ([True], [[False], [False]])


async def test_a_speaker_whose_check_is_pending_is_still_waited_for(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [2], window_ms=10000)
    pending = FakePipeline("pending", speaker=True, pcm=preroll(100, 3162))
    ready = FakePipeline("ready", speaker=True, pcm=preroll(100, 1000))
    pending.verifier_pending = True
    pending.wake(arbiter)
    ready.wake(arbiter)
    assert pending.verdicts == ready.verdicts == []
    pending.verifier_pending = False
    arbiter.run_changed(pending, pending.token)
    assert (pending.verdicts, ready.verdicts) == ([True], [False])


async def test_a_candidate_whose_run_moved_on_is_skipped(monkeypatch):
    arbiter = make_arbiter(monkeypatch, [3])
    speaker = FakePipeline("speaker", speaker=True)
    panel = FakePipeline("panel", speaker=False)
    speaker.wake(arbiter)
    panel.wake(arbiter)
    speaker.token += 1  # its registered run was replaced before the window closed
    await asyncio.sleep(0.1)
    assert (panel.verdicts, speaker.verdicts) == ([True], [])


async def test_a_group_left_without_candidates_starts_no_cooldown(monkeypatch):
    online = [3]
    arbiter = make_arbiter(monkeypatch, online)
    gone = FakePipeline("gone", speaker=True)
    gone.wake(arbiter)
    gone.waiting = False  # stopped before the window ran out
    await asyncio.sleep(0.1)
    online[0] = 1
    other = FakePipeline("other", speaker=False)
    other.wake(arbiter)
    assert (gone.verdicts, other.verdicts) == ([], [True])
