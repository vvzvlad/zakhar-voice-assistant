"""TTS stage contract: the TtsBackend ABC plus shared pure helpers.

Backend implementations live with their providers in src/plugins/tts/ (one folder
per brick); the orchestrator imports only this contract module.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


# Sentence-ending punctuation; ellipsis "…" is normalized to "." first because
# espeak-ng does not treat the "…" character as a pause.
def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping terminal punctuation. Ellipsis "…" and
    runs of dots are normalized to a single ".". Returns non-empty, stripped parts."""
    text = text.replace("…", ".")
    text = re.sub(r"\.{2,}", ".", text)              # "..." -> "."
    parts = re.split(r"(?<=[.!?])\s+", text.strip())  # split after . ! ?
    # Keep only fragments with a word character, so punctuation-only pieces
    # (e.g. "." / "?" / "…"->".") that piper can't voice are dropped.
    return [p.strip() for p in parts if p.strip() and re.search(r"\w", p, re.UNICODE)]


# Raw audio from a TTS backend: signed 16-bit little-endian mono PCM, no header,
# at the rate carried in the mime ("audio/pcm;rate=48000").
def pcm_mime(rate: int) -> str:
    """Mime of raw s16le mono PCM at `rate` Hz."""
    return f"audio/pcm;rate={rate}"


def pcm_rate(mime: str) -> int | None:
    """Sample rate of a pcm_mime() string, or None for any other mime."""
    base, _, param = mime.partition(";")
    if base.strip() != "audio/pcm":
        return None
    key, _, value = param.partition("=")
    return int(value) if key.strip() == "rate" else None


class OwnedChunks:
    """A chunk iterator that owns an already-open resource (a streamed HTTP response).
    aclose() releases it even before the first chunk: an async generator closed
    before it started never runs its own finally."""

    def __init__(self, chunks, release):
        self._chunks = chunks
        self._release = release

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._chunks.__anext__()

    async def aclose(self):
        try:
            await self._chunks.aclose()
        finally:
            await self._release()


class TtsBackend(ABC):
    """Abstract TTS backend: text -> (mime, audio_bytes). The mime is "audio/wav"
    or a pcm_mime()."""

    @abstractmethod
    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        ...

    async def synthesize_stream(
        self, text: str, lang: str = "ru"
    ) -> tuple[str, AsyncIterator[bytes]]:
        """Streaming synthesis: (mime, async chunk iterator). Default adapter wraps
        the buffered synthesize() in a single-chunk stream, so every backend is
        streamable; engines with native chunked output override this. Connect/auth
        errors must be raised HERE (before the iterator is returned), so callers can
        fail the run cleanly; mid-stream errors surface from the iterator."""
        mime, audio = await self.synthesize(text, lang)

        async def _single():
            if audio:
                yield audio

        return (mime, _single())
