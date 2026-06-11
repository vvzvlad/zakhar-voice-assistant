"""Pre-STT conditioning of the whole captured utterance + WAV container helpers.

Distinct from VAD (the speech/no-speech end-pointing stage): these helpers are
applied ONCE by the pipeline to the finalized utterance — high-pass / peak
normalization / lead-in trim before STT, and the WAV builders for the capture
and diagnostic audio paths — never per chunk.
"""

import io
import wave

import numpy as np

from src.vad import SAMPLE_RATE


def highpass(pcm: bytes, cutoff_hz: float = 80.0, sample_rate: int = SAMPLE_RATE) -> bytes:
    """High-pass-filter 16-bit mono PCM via a numpy rFFT (no SciPy).

    Removes DC offset and low-frequency rumble (table thumps, HVAC) below ~cutoff_hz
    that carry no speech and skew normalization / hurt STT. The whole utterance is
    filtered at once with a smooth raised-cosine transition band (0 below 0.5*cutoff,
    1 above cutoff), so there is no per-chunk state and no SciPy dependency. Empty /
    too-short input is returned unchanged; a trailing odd byte is preserved.
    """
    if not pcm:
        return pcm
    n = len(pcm) - (len(pcm) % 2)  # whole int16 samples only
    if n < 4:
        return pcm
    x = np.frombuffer(pcm[:n], dtype="<i2").astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    lo = cutoff_hz * 0.5
    ramp = np.clip((freqs - lo) / (cutoff_hz - lo), 0.0, 1.0)
    mask = 0.5 - 0.5 * np.cos(np.pi * ramp)  # raised-cosine 0->1 across [lo, cutoff_hz]
    y = np.fft.irfft(spec * mask, n=x.size)
    filtered = np.clip(y, -32768, 32767).astype("<i2").tobytes()
    return filtered + pcm[n:]  # keep any trailing odd byte unchanged


def normalize_peak(pcm: bytes, target_dbfs: float = -3.0, max_gain: float = 30.0) -> bytes:
    """Peak-normalize 16-bit mono PCM so its loudest sample hits ``target_dbfs``.

    A per-utterance adaptive replacement for a fixed gain: the quiet less-processed
    mic channel is brought to a consistent level without clipping, while already-loud
    samples scale down. ``max_gain`` caps the boost so a near-silent clip does not blow
    up the noise floor. Empty / near-silent input is returned unchanged; a trailing odd
    byte is preserved.
    """
    if not pcm:
        return pcm
    n = len(pcm) - (len(pcm) % 2)
    if n == 0:
        return pcm
    x = np.frombuffer(pcm[:n], dtype="<i2").astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1.0:
        return pcm  # silence — leave untouched
    target = 32767.0 * (10.0 ** (target_dbfs / 20.0))
    gain = min(target / peak, max_gain)
    boosted = np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()
    return boosted + pcm[n:]


def trim_start_pcm(pcm: bytes, trim_ms: int) -> bytes:
    """Drop the first ``trim_ms`` of 16 kHz / mono / 16-bit PCM from an utterance.

    Used to cut the wake-word tail / button-press lead-in off the captured sample
    before STT, so it does not pollute the transcription. The cut is sample-aligned
    (SAMPLE_RATE * 2 / 1000 = 32 bytes/ms is always even). If the trim would consume
    the whole sample (or more), the PCM is returned unchanged so we never hand empty
    audio to STT.
    """
    if trim_ms <= 0:
        return pcm
    trim_bytes = int(trim_ms * SAMPLE_RATE * 2 / 1000)
    if trim_bytes <= 0 or trim_bytes >= len(pcm):
        return pcm
    return pcm[trim_bytes:]


def write_wav(path: str, pcm: bytes) -> None:
    """Write 16 kHz / mono / 16-bit PCM to a WAV file at `path`."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def pcm_to_wav_bytes(pcm: bytes, pcm2: bytes = b"") -> bytes:
    """Build a 16 kHz / 16-bit WAV container from PCM, fully in memory.

    With only `pcm`, builds a mono WAV — used by the manual (ephemeral) capture
    path, where the bytes are handed straight back to the API caller. With a
    non-empty `pcm2`, builds a STEREO WAV for the stored per-run diagnostic
    audio: LEFT = `pcm` (the pipeline/STT channel, exactly what STT received),
    RIGHT = `pcm2` (the other raw mic channel, for channel comparison). The
    shorter channel is zero-padded to the longer one; a trailing odd byte is
    dropped (acceptable for diagnostic audio).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        if not pcm2:
            w.setnchannels(1)
            w.writeframes(pcm)
        else:
            left = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")
            right = np.frombuffer(pcm2[: len(pcm2) - len(pcm2) % 2], dtype="<i2")
            n = max(left.size, right.size)
            if left.size < n:
                left = np.pad(left, (0, n - left.size))
            if right.size < n:
                right = np.pad(right, (0, n - right.size))
            w.setnchannels(2)
            w.writeframes(np.column_stack([left, right]).astype("<i2").tobytes())
    return buf.getvalue()
