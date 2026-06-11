"""Server-side ack chime assets: the synthesized two-tone clip and file-based clip
loading. Not a TTS stage concern — the chime is a pipeline/device asset."""

import io
import os
import wave

from loguru import logger

from src.audio_codec import wav_to_mp3


def make_ack_chime_mp3(
    *,
    sample_rate: int = 22050,
    tones_hz: tuple[float, float] = (880.0, 1320.0),
    tone_ms: int = 130,
    gap_ms: int = 20,
    amplitude: float = 0.5,
) -> bytes:
    """Synthesize a short two-tone confirmation chime ("блям") and return it as MP3.

    A pleasant ~300 ms rising two-tone beep with a per-tone attack/decay envelope (so
    there are no clicks), built with numpy and transcoded to MP3 via wav_to_mp3 so the
    speaker firmware can decode it. Pure/deterministic — the pipeline builds it ONCE and
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
    return wav_to_mp3(buf.getvalue())


def build_ack_clip(sound_path: str, *, name: str = "") -> tuple[str, bytes]:
    """Build the end-of-phrase ack clip (mime, audio) from a sound_path.

    If sound_path points to an existing file, load it — transcoding a WAV to MP3 (the
    speaker firmware can't decode WAV; a bad/unreadable WAV falls back to the
    synthesized chime so playback never goes silent). An empty or missing path yields
    the synthesized two-tone chime. `name` is only used for log context. Does file IO
    and (for WAV) a blocking transcode, so callers on the event loop should run it via
    asyncio.to_thread. No caching here — callers cache as needed.
    """
    path = (sound_path or "").strip()
    use_file = bool(path) and os.path.isfile(path)
    if use_file:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext == "wav":
            try:
                return "audio/mpeg", wav_to_mp3(raw)
            except Exception as e:
                logger.warning(
                    f"{name}: ack chime transcode failed for {path}: {e}; "
                    f"using synthesized chime"
                )
                return "audio/mpeg", make_ack_chime_mp3()
        if ext == "flac":
            return "audio/flac", raw
        # mp3 (and any other) served verbatim as audio/mpeg
        return "audio/mpeg", raw
    return "audio/mpeg", make_ack_chime_mp3()
