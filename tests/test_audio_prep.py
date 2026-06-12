"""Edge-case tests for the pure pre-STT DSP helpers in src/audio_prep.py.

The mainline behavior (DC removal, peak targets, stereo interleave/pad) is
covered in tests/test_pipeline.py; these pin the documented edge contracts:
odd-byte preservation, sub-sample inputs, output-length invariants, real
low-frequency attenuation and the WAV container headers.
"""

import io
import wave

import numpy as np

from src.audio_prep import (
    VAD_BOOST_FLOOR,
    VAD_BOOST_TARGET,
    highpass,
    normalize_peak,
    pcm_to_wav_bytes,
    vad_boost,
    write_wav,
)
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


# --- vad_boost: the decision-only per-chunk makeup gain ---------------------------
# Moved here from the vad/webrtc plugin: the pipeline now applies it to every chunk
# before any VAD engine sees it (gated by core.vad.mic_auto_gain).

def test_vad_boost_lifts_quiet_frame_toward_target():
    # A frame whose own samples are small, with a representative running utterance
    # peak (~3000), is scaled by min(target/peak, max_gain) so quiet speech reaches
    # the VAD engine's range. The gain is well below the max_gain cap here.
    peak = 3000
    gain = VAD_BOOST_TARGET / peak
    samples = np.array([100, -200, 300, -50] * 80, dtype="<i2")  # 320 samples = 640 B
    frame = samples.tobytes()
    out = vad_boost(frame, peak)
    out_samples = np.frombuffer(out, dtype="<i2")
    expected = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype("<i2")
    assert np.array_equal(out_samples, expected)
    # Sanity: the frame really was amplified (gain > 1).
    assert gain > 1.0
    assert np.max(np.abs(out_samples)) > np.max(np.abs(samples))


def test_vad_boost_below_floor_is_identity():
    # peak below VAD_BOOST_FLOOR (30) -> treated as pre-roll silence -> frame unchanged.
    frame = (np.array([10, -20, 30, -40] * 80, dtype="<i2")).tobytes()
    assert vad_boost(frame, VAD_BOOST_FLOOR - 10) == frame


def test_vad_boost_caps_at_max_gain():
    # A peak just above the floor gives a target/peak ratio above the max_gain cap
    # (128); the cap must clamp it. peak=40 -> target/40 ≈ 145 > 128, so the gain is 128.
    peak = 40
    assert VAD_BOOST_TARGET / peak > 128.0  # uncapped ratio really does exceed the cap
    samples = np.array([5, -6, 7, -8] * 80, dtype="<i2")  # small -> no clipping at 128x
    frame = samples.tobytes()
    out = np.frombuffer(vad_boost(frame, peak), dtype="<i2")
    expected = np.clip(samples.astype(np.float32) * 128.0, -32768, 32767).astype("<i2")
    assert np.array_equal(out, expected)


def test_vad_boost_never_attenuates_loud_signal():
    # When the running peak already sits at or above the boost target, gain <= 1:
    # the frame must come back byte-identical (the boost only ever lifts, never cuts).
    frame = np.array([100, -200, 300, -400] * 80, dtype="<i2").tobytes()
    assert vad_boost(frame, int(VAD_BOOST_TARGET)) == frame   # gain == 1 exactly
    assert vad_boost(frame, 8000) == frame                    # gain < 1


def test_vad_boost_handles_empty_and_odd_length():
    # Empty frame -> unchanged. Odd-length (a stray trailing byte): the whole-sample
    # prefix is boosted, the torn trailing byte survives verbatim (same contract as
    # the other audio_prep helpers) and no exception is raised.
    assert vad_boost(b"", 3000) == b""
    odd = (np.array([100, -200], dtype="<i2")).tobytes() + b"\x07"  # 5 bytes
    out = vad_boost(odd, 3000)
    assert len(out) == len(odd)
    assert out[-1:] == b"\x07"
    assert out[:4] != odd[:4]  # the whole samples really were boosted
    # A single-byte frame has no whole int16 sample -> returned unchanged.
    assert vad_boost(b"\x01", 3000) == b"\x01"
