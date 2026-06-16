"""Wakeword verification stage contract: second-stage wake-word gate.

Given the captured pre-roll PCM (16 kHz/16-bit/mono, containing the wake word),
decide whether the wake word is really present. A reject SUPPRESSES the rest of
the pipeline (STT/LLM/TTS never run). The PIPELINE owns when/where verify() is
called and the fail-open/closed timeout policy reads `timeout_ms`/`fail_open`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

SAMPLE_RATE = 16000  # 16 kHz/16-bit mono PCM — the pipeline's stream format


@dataclass(frozen=True)
class WakewordVerdict:
    accepted: bool
    score: float | None   # confidence if available; None when unknown/error


class WakewordVerifier(ABC):
    # Pipeline reads these to apply the timeout + fail-open/closed policy.
    timeout_ms: int = 300
    fail_open: bool = True
    # How much of the buffered pre-roll head (ms) the pipeline hands to verify().
    window_ms: int = 1500

    @abstractmethod
    async def verify(self, pcm: bytes) -> WakewordVerdict: ...


class PassthroughVerifier(WakewordVerifier):
    """No-op used when the stage is disabled: always accepts."""

    timeout_ms = 1000
    fail_open = True
    window_ms = 1500

    async def verify(self, pcm: bytes) -> WakewordVerdict:
        return WakewordVerdict(accepted=True, score=None)
