"""Server-side ack chime assets: the synthesized two-tone clip and file-based clip
loading. Not a TTS stage concern — the chime is a pipeline/device asset."""

import io
import os
import wave

from loguru import logger


def make_ack_chime_wav(
    *,
    sample_rate: int = 22050,
    tones_hz: tuple[float, float] = (880.0, 1320.0),
    tone_ms: int = 130,
    gap_ms: int = 20,
    amplitude: float = 0.5,
) -> bytes:
    """Synthesize a short two-tone confirmation chime ("блям") and return it as WAV.

    A pleasant ~300 ms rising two-tone beep with a per-tone attack/decay envelope (so
    there are no clicks), built with numpy. Pure/deterministic — the pipeline builds it ONCE and
    caches the bytes (see Pipeline._ack_clip / Pipeline._ack_clip_bytes). numpy is
    already a dependency.
    """
    import numpy as np

    def _tone(freq: float, ms: int) -> "np.ndarray":
        n = int(sample_rate * ms / 1000)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        t = np.arange(n, dtype=np.float32) / sample_rate
        wave_arr = np.sin(2 * np.pi * freq * t)
        # Short raised-cosine attack/release so the tone fades in and out (no clicks).
        # Clamp edge to n // 2 so the attack (env[:edge]) and release (env[-edge:])
        # ramps never overlap for a very short tone (when 2*edge > n); the defaults
        # (130 ms tone) are far above this bound, so they are unchanged.
        edge = max(1, min(int(sample_rate * 0.008), n // 2))  # ~8 ms ramps
        env = np.ones(n, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        env[:edge] = ramp
        env[-edge:] = ramp[::-1]
        return wave_arr * env

    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    signal = np.concatenate([_tone(tones_hz[0], tone_ms), gap, _tone(tones_hz[1], tone_ms)])
    pcm16 = np.clip(signal * amplitude, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")  # little-endian 16-bit PCM
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setframerate(sample_rate)
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def build_ack_clip(sound_path: str, *, name: str = "") -> tuple[str, bytes]:
    """Build the end-of-phrase ack clip (mime, audio) from a sound_path.

    Chimes are WAV only: a sound_path naming an existing 16-bit .wav file is loaded
    as is; an empty or missing path, any other file, or a bad/unreadable WAV yields
    the synthesized two-tone chime, so playback never goes silent. `name` is only
    used for log context. Does file IO, so callers on the event loop should run it via
    asyncio.to_thread. No caching here — callers cache as needed.
    """
    path = (sound_path or "").strip()
    if path and os.path.isfile(path):
        if path.lower().endswith(".wav"):
            with open(path, "rb") as f:
                raw = f.read()
            try:
                with wave.open(io.BytesIO(raw), "rb") as w:
                    if w.getsampwidth() != 2:
                        raise ValueError(f"{w.getsampwidth() * 8}-bit, expected 16-bit")
                return "audio/wav", raw
            except (wave.Error, EOFError, ValueError) as e:
                logger.warning(f"{name}: ack chime {path} is unusable: {e}; using synthesized chime")
        else:
            logger.warning(f"{name}: ack chime {path} is not a WAV file; using synthesized chime")
    return "audio/wav", make_ack_chime_wav()
