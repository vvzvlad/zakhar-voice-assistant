"""Brickwall compression of the audio played on a device (core.devices[].compress_replies):
a streaming look-ahead peak limiter with makeup gain that lifts quiet parts of speech and
holds its peaks at CEILING_DBFS, so a reply reads louder over music at the same dial."""

import math

import numpy as np

# The firmware's DAC adds up to +10.5 dB of digital gain at the top of the dial
# (firmware/main/audio_core.c volume_dac_byte(100) = 21 half-dB steps), so peaks at -11.5 dBFS
# stay under full scale with 1 dB to spare.
CEILING_DBFS = -11.5
# The most a quiet part is lifted: without a cap the gain of a near-silent pause grows without end.
MAX_GAIN_DB = 20
# How fast the gain recovers after a peak. On a Fish Audio phrase at dial 85 % the RMS rose
# +2.6 dB at 20 ms against +1.9 dB at 50 ms; a faster release lifts more and pumps more.
RELEASE_MS = 20
# The span of one gain value; one frame is also the look-ahead, so the gain is down before a peak.
FRAME_MS = 2


class ReplyCompressor:
    """One stream's limiter over int16 mono samples at `rate`. Per frame the peak of that
    frame and the next (one frame of look-ahead) drives an envelope with an instant attack
    and an exponential release; the frame's gain brings the envelope to the ceiling, at
    most MAX_GAIN_DB. The gain is interpolated per sample between frame centres, and what
    still exceeds the ceiling is clipped to it.

    process() returns what it can already finish, which trails its input by 1.5 to 2.5
    frames; flush() returns the rest, so the output is as long as the input and the same
    however the input was chunked."""

    def __init__(self, rate: int):
        self._frame = int(rate * FRAME_MS / 1000)
        self._release = math.exp(-self._frame / rate / (RELEASE_MS / 1000))
        self._ceiling = 10 ** (CEILING_DBFS / 20)
        self._max_gain = 10 ** (MAX_GAIN_DB / 20)
        self._held = np.zeros(0)  # input not returned yet, as floats in [-1, 1)
        self._start = 0  # the stream index of _held[0]
        self._scanned = 0  # the end of the whole frames whose peak is taken
        self._peak = None  # the peak of the last such frame, waiting for the next one's
        self._env = 0.0
        self._gains: list[float] = []  # frame gains from frame _first on
        self._first = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        self._held = np.concatenate([self._held, samples / 32768.0])
        end = self._start + len(self._held)
        while self._scanned + self._frame <= end:
            at = self._scanned - self._start
            self._take(np.abs(self._held[at:at + self._frame]).max())
            self._scanned += self._frame
        if not self._gains:
            return np.zeros(0, dtype=np.int16)
        # A sample is finished once the gains of the frame centres on both sides of it are known.
        last = self._first + len(self._gains) - 1
        return self._emit(math.ceil(last * self._frame + self._frame / 2))

    def flush(self) -> np.ndarray:
        end = self._start + len(self._held)
        if self._scanned < end:  # a part frame at the end counts as a frame
            self._take(np.abs(self._held[self._scanned - self._start:]).max())
            self._scanned = end
        if self._peak is not None:  # the last frame looks ahead into silence
            self._gain(self._peak)
            self._peak = None
        return self._emit(end)

    def _take(self, peak: float) -> None:
        """A whole frame's peak: it finishes the gain of the frame before it."""
        if self._peak is not None:
            self._gain(max(self._peak, peak))
        self._peak = peak

    def _gain(self, peak: float) -> None:
        r = self._release
        self._env = peak if peak > self._env else self._env * r + peak * (1 - r)
        self._gains.append(min(self._max_gain, self._ceiling / self._env) if self._env > 0 else self._max_gain)

    def _emit(self, end: int) -> np.ndarray:
        """The held samples up to stream index `end`, through their gains, as int16."""
        n = end - self._start
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        centres = np.arange(self._first, self._first + len(self._gains)) * self._frame + self._frame / 2
        gain = np.interp(np.arange(self._start, end), centres, self._gains)
        out = np.clip(self._held[:n] * gain, -self._ceiling, self._ceiling)
        self._held = self._held[n:]
        self._start = end
        # The next sample needs the gains from the frame centre at or before it on.
        last = self._first + len(self._gains) - 1
        drop = min(max(0, math.floor((end - self._frame / 2) / self._frame)), last) - self._first
        del self._gains[:drop]
        self._first += drop
        return np.round(out * 32768).astype(np.int16)
