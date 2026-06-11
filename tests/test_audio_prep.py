"""Edge-case tests for the pure pre-STT DSP helpers in src/audio_prep.py.

The mainline behavior (DC removal, peak targets, stereo interleave/pad) is
covered in tests/test_pipeline.py; these pin the documented edge contracts:
odd-byte preservation, sub-sample inputs, output-length invariants, real
low-frequency attenuation and the WAV container headers.
"""

import io
import wave

import numpy as np

from src.audio_prep import highpass, normalize_peak, pcm_to_wav_bytes, write_wav
from src.vad import SAMPLE_RATE


# --- highpass: odd-byte / short-input handling -----------------------------------

def test_highpass_preserves_trailing_odd_byte():
    # Docstring promise: a trailing odd byte (a torn int16 sample) survives verbatim
    # and the output length matches the input length exactly.
    samples = np.array([1000, -2000, 3000, -4000, 5000, -6000, 7000, -8000], dtype="<i2")
    pcm = samples.tobytes() + b"\x07"  # 17 bytes: 8 whole samples + 1 stray byte
    out = highpass(pcm)
    assert len(out) == len(pcm)
    assert out[-1:] == b"\x07"


def test_highpass_inputs_shorter_than_four_bytes_are_identity():
    # Fewer than 4 bytes of whole samples -> the n < 4 early return hands the
    # input back byte-identical (2 bytes = 1 sample, 3 bytes = 1 sample + odd byte).
    two = b"\x12\x34"
    three = b"\x12\x34\x56"
    assert highpass(two) == two
    assert highpass(three) == three


def test_highpass_odd_sample_count_keeps_length():
    # An ODD number of int16 samples (5 samples = 10 bytes): np.fft.irfft without an
    # explicit n= would round the output down to an even sample count and silently
    # shrink the buffer; the n=x.size argument must keep length(out) == length(in).
    pcm = np.array([100, -200, 300, -400, 500], dtype="<i2").tobytes()
    out = highpass(pcm)
    assert len(out) == len(pcm)


def test_highpass_attenuates_subsonic_tone():
    # A 30 Hz tone sits below half the 80 Hz cutoff (the mask is 0 below 40 Hz), so
    # nearly all of its energy must be removed — not just a DC offset. 8000 samples
    # at 16 kHz hold exactly 15 full cycles, so the tone lands on a single FFT bin
    # with no spectral leakage above the stopband edge.
    t = np.arange(8000) / SAMPLE_RATE
    sig = (10000 * np.sin(2 * np.pi * 30 * t)).astype("<i2")
    rms_before = float(np.sqrt(np.mean(sig.astype(np.float32) ** 2)))
    out = np.frombuffer(highpass(sig.tobytes()), "<i2").astype(np.float32)
    rms_after = float(np.sqrt(np.mean(out ** 2)))
    assert rms_after < rms_before * 0.05  # the rumble is essentially gone


# --- normalize_peak: odd-byte / sub-sample edges ----------------------------------

def test_normalize_peak_preserves_trailing_odd_byte_after_boost():
    # The whole-sample prefix is boosted toward -3 dBFS while the stray trailing
    # byte rides along untouched (and the total length never changes).
    pcm = np.array([1000, -1000], dtype="<i2").tobytes() + b"\x07"
    out = normalize_peak(pcm)
    assert len(out) == len(pcm)
    assert out[-1:] == b"\x07"
    boosted = np.frombuffer(out[:-1], "<i2")
    assert int(np.max(np.abs(boosted))) > 1000  # the prefix really was boosted


def test_normalize_peak_single_byte_is_identity():
    # 1 byte holds no whole int16 sample (n == 0): returned unchanged, no exception.
    assert normalize_peak(b"\x07") == b"\x07"


# --- write_wav: container header round-trip ----------------------------------------

def test_write_wav_header_and_pcm_round_trip(tmp_path):
    # The file must be a 16 kHz / mono / 16-bit WAV whose frames are the input PCM
    # verbatim (the header params are asserted nowhere else in the suite).
    pcm = np.array([1, -2, 3, -4, 5, -6], dtype="<i2").tobytes()
    path = str(tmp_path / "out.wav")
    write_wav(path, pcm)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == 6
        assert w.readframes(w.getnframes()) == pcm


# --- pcm_to_wav_bytes: odd-length / empty edges ------------------------------------

def _parse_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getnframes(), w.readframes(w.getnframes())


def test_pcm_to_wav_bytes_stereo_drops_odd_bytes_and_pads_to_longer():
    # Both channels carry a stray trailing byte: per the documented contract the odd
    # byte is dropped, the shorter channel is zero-padded and the frame count equals
    # the LONGER channel's whole-sample count. No exception.
    left = np.array([100, -200, 300, -400], dtype="<i2")
    right = np.array([9, 8], dtype="<i2")
    wav = pcm_to_wav_bytes(left.tobytes() + b"\x07", right.tobytes() + b"\x07")
    nch, nframes, frames = _parse_wav(wav)
    assert nch == 2
    assert nframes == 4
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, 2)
    assert (samples[:, 0] == left).all()
    assert (samples[:, 1] == [9, 8, 0, 0]).all()


def test_pcm_to_wav_bytes_mono_empty_pcm_yields_valid_empty_wav():
    # The manual-capture path can hand over an empty buffer: the result must still be
    # a parseable mono 16 kHz WAV container with zero frames.
    wav = pcm_to_wav_bytes(b"")
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == 0
