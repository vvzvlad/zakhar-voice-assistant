"""STT stage contract: the SttBackend ABC plus shared pure helpers.

Backend implementations live with their providers in src/plugins/stt/ (one folder
per brick); the orchestrator imports only this contract module.
"""

import io
import wave
from abc import ABC, abstractmethod


def pcm_to_wav(
    pcm: bytes, sample_rate: int = 16000, sample_width: int = 2, channels: int = 1
) -> bytes:
    """Wrap raw PCM samples in a WAV container (pure, testable)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


# Known Whisper STT hallucination markers (lowercase). Whisper tends to emit
# leftover subtitle-credit / stock phrases (training-data artifacts) on
# silence/noise: "DimaTorzok" is one such credit string and "Продолжение
# следует..." ("to be continued") is a recurring stock caption. When a
# transcription contains one of these (substring, case-insensitive), we treat
# the run as if nothing was said and drop it.
STT_HALLUCINATION_MARKERS = ("dimatorzok", "продолжение следует")


def contains_stt_hallucination(text: str) -> bool:
    """Return True if the STT text contains a known Whisper hallucination marker.

    These are known STT (Whisper) hallucinations — subtitle-credit artifacts that
    surface on silence/noise — and are dropped as if nothing was said. The check is
    case-insensitive (Whisper varies the casing of the artifact).
    """
    folded = text.casefold()
    return any(marker in folded for marker in STT_HALLUCINATION_MARKERS)


class SttBackend(ABC):
    """Abstract STT backend: raw 16 kHz/16-bit/mono PCM -> transcript.

    Returns the transcript on success, "" when there is nothing to transcribe
    or no speech was recognized. Raises StageError("stt", ...) on failure.
    """

    @abstractmethod
    async def transcribe(self, pcm: bytes) -> str:
        ...
