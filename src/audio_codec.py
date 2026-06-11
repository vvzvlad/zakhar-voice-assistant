"""Cross-stage audio transcoding helpers (WAV -> MP3 for the speaker firmware)."""

import io
import wave


# Formats the speaker firmware can decode; anything else is transcoded
# at the delivery boundary (currently: WAV -> MP3 via lameenc).
PLAYABLE_MIMES = {"audio/mpeg", "audio/flac"}


def to_playable(mime: str, audio: bytes) -> tuple[str, bytes]:
    """Adapt an audio clip to a speaker-decodable format. Native
    playable formats pass through untouched; WAV is transcoded to MP3.
    Lives at the delivery boundary, NOT inside synthesis backends —
    a backend returns its engine's native format. Blocking (lameenc)
    for WAV input, so call it via asyncio.to_thread on the event loop."""
    if mime in PLAYABLE_MIMES:
        return mime, audio
    if mime == "audio/wav":
        return "audio/mpeg", wav_to_mp3(audio)
    # Unknown formats are served as-is (same lenient behavior tts_url
    # has today for unknown mimes).
    return mime, audio


def wav_to_mp3(wav_bytes: bytes, bit_rate: int = 64, quality: int = 2) -> bytes:
    """Transcode a 16-bit PCM WAV (mono/stereo) to MP3 via lameenc.

    The speaker firmware can't decode WAV, so Piper output is served as MP3.
    """
    import lameenc  # local import: only needed when Piper is used

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_bit_rate(bit_rate)
    enc.set_quality(quality)
    return bytes(enc.encode(pcm) + enc.flush())
