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


class TtsBackend(ABC):
    """Abstract TTS backend: text -> (mime, audio_bytes)."""

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
