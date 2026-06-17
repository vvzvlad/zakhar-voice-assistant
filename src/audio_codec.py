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
    import lameenc  # local import: pay the lameenc import cost only when a WAV is actually transcoded

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        # lameenc assumes 16-bit PCM input; feeding it 24/8-bit samples produces
        # distorted audio with no error, so make the silent bug loud instead.
        if sampwidth != 2:
            raise ValueError(f"wav_to_mp3 expects 16-bit PCM WAV, got {sampwidth * 8}-bit")
        pcm = wf.readframes(wf.getnframes())
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_bit_rate(bit_rate)
    enc.set_quality(quality)
    return bytes(enc.encode(pcm) + enc.flush())


def make_mp3_stream_encoder(
    sample_rate: int, channels: int, sample_width: int = 2, bit_rate: int = 64, quality: int = 2
):
    """Build a lameenc encoder for INCREMENTAL WAV->MP3 streaming.

    Feed raw 16-bit PCM with `enc.encode(pcm)` (returns MP3 bytes for that block,
    possibly b"") and call `enc.flush()` once at the end for the trailing frames.
    Lives at the delivery-boundary codec module (not inside a synthesis backend),
    next to wav_to_mp3 which shares the same lameenc config. lameenc assumes 16-bit
    PCM input; reject other widths loudly (same guard as wav_to_mp3)."""
    if sample_width != 2:
        raise ValueError(f"make_mp3_stream_encoder expects 16-bit PCM, got {sample_width * 8}-bit")
    import lameenc  # local import: pay the lameenc import cost only when streaming
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_bit_rate(bit_rate)
    enc.set_quality(quality)
    return enc
