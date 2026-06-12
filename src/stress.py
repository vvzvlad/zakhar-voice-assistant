"""Accent (Russian stress) stage contract: the Accentizer ABC.

The accent stage sits BETWEEN the LLM and TTS: it takes the LLM reply text and
returns the same text with stress marks placed on the stressed vowels. Its output
is the canonical LLM->TTS text contract used across this repo — plain text with a
"+" before each stressed vowel (e.g. "прив+ет") — so every TTS backend already
adapts it via src/plugins/tts/_ru_text.py, and NO TTS backend change is needed.

Backend implementations live with their providers in src/plugins/stress/; the
orchestrator imports only this contract module.
"""

from abc import ABC, abstractmethod


class Accentizer(ABC):
    """Abstract accent backend: text -> text with "+vowel" stress marks."""

    @abstractmethod
    async def accentize(self, text: str) -> str:
        ...


class PassthroughAccentizer(Accentizer):
    """No-op accentizer used when the stage is disabled: returns the text
    unchanged so the pipeline can keep an Accentizer in hand without a model."""

    async def accentize(self, text: str) -> str:
        return text
