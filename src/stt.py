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


class StreamingSttSession(ABC):
    """A live STT session fed PCM chunks as they are captured. The pipeline VAD
    owns end-pointing: it feed()s chunks during speech, then calls finish() to
    force the final transcript, or aclose() to abandon (silence/error)."""

    @abstractmethod
    def feed(self, pcm: bytes) -> None:
        """Enqueue a PCM chunk. Synchronous and non-blocking; never raises."""

    @abstractmethod
    async def finish(self) -> str:
        """Signal end-of-utterance and return the final transcript. Raises
        StageError('stt', ...) on failure."""

    @abstractmethod
    async def aclose(self) -> None:
        """Abandon the session and release resources. Idempotent; never raises."""


class StreamingTranscript:
    """Accumulates streaming STT events into a final transcript.
    Finals are keyed by final_index; final_refinement (normalized) overrides the
    raw final at the same index. result() joins finals in index order; if no
    final ever arrived, falls back to the last partial."""

    def __init__(self):
        self._finals: dict[int, str] = {}
        self._normalized: dict[int, str] = {}
        self._last_partial = ""

    def add_partial(self, text: str) -> None:
        if text:
            self._last_partial = text

    def add_final(self, index: int, text: str) -> None:
        self._finals[index] = text

    def add_normalized(self, index: int, text: str) -> None:
        self._normalized[index] = text

    def result(self) -> str:
        if self._finals:
            parts = [
                (self._normalized.get(i) or self._finals.get(i, "")).strip()
                for i in sorted(self._finals)
            ]
            joined = " ".join(p for p in parts if p).strip()
            # An empty/whitespace-only final must not clobber a valid partial:
            # Yandex v3 can emit a final/final_refinement with empty alternatives.
            if joined:
                return joined
        return self._last_partial.strip()


class SttBackend(ABC):
    """Abstract STT backend: raw 16 kHz/16-bit/mono PCM -> transcript.

    Returns the transcript on success, "" when there is nothing to transcribe
    or no speech was recognized. Raises StageError("stt", ...) on failure.
    """

    @abstractmethod
    async def transcribe(self, pcm: bytes) -> str:
        ...

    def open_stream(self) -> "StreamingSttSession | None":
        """Return a live streaming session, or None if this backend only does
        batch transcribe(). Default: no streaming."""
        return None
