"""The streaming reply compressor: peaks held at the ceiling, quiet audio lifted by the
maximum gain, silence untouched, and the same output however the input is chunked."""

import numpy as np
import pytest

from src.reply_compressor import CEILING_DBFS, MAX_GAIN_DB, ReplyCompressor

RATE = 48000
CEILING = 10 ** (CEILING_DBFS / 20) * 32768  # in int16 units


def sine(dbfs: float, seconds: float, hz: float = 1000) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return np.round(10 ** (dbfs / 20) * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def run(x: np.ndarray, *sizes: int) -> np.ndarray:
    """`x` through one compressor in chunks of `sizes` in turn (one chunk without sizes)."""
    c = ReplyCompressor(RATE)
    out, i, k = [], 0, 0
    while i < len(x):
        size = sizes[k % len(sizes)] if sizes else len(x)
        out.append(c.process(x[i:i + size]))
        i, k = i + size, k + 1
    out.append(c.flush())
    return np.concatenate(out)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


def test_a_full_scale_burst_peaks_at_the_ceiling():
    quiet = np.zeros(RATE // 10, dtype=np.int16)
    out = np.abs(run(np.concatenate([quiet, sine(0, 0.3), quiet])).astype(int))
    assert CEILING - 1 <= out.max() <= CEILING + 1


def test_a_quiet_sine_comes_out_max_gain_louder_under_the_ceiling():
    x = sine(-40, 0.5)
    out = run(x)
    assert 20 * np.log10(rms(out) / rms(x)) == pytest.approx(MAX_GAIN_DB, abs=0.01)
    assert np.abs(out.astype(int)).max() <= CEILING


def test_any_chunking_gives_the_one_shot_output():
    # Loud and quiet stretches, and a length that is not a whole number of frames.
    t = np.arange(RATE // 2 + 37) / RATE
    x = np.round(32767 * np.abs(np.sin(2 * np.pi * 3 * t)) ** 3 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    whole = run(x)
    for size in (1, 7, 96, 1000):
        assert np.array_equal(run(x, size), whole), size
    assert np.array_equal(run(x, 1, 333, 4097, 7), whole)


@pytest.mark.parametrize("n", [0, 1, 95, 96, 97, 4801])
def test_the_output_is_as_long_as_the_input(n):
    assert len(run(sine(-6, 1)[:n], 50)) == n


def test_silence_stays_silence():
    assert not run(np.zeros(RATE // 2, dtype=np.int16), 1000).any()
