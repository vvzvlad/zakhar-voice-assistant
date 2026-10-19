"""Follow-up stage contract: does a phrase heard after the answer continue the dialog?

After the spoken answer of a session the pipeline keeps listening for `window_ms`
without a wake word (docs/device-protocol.md, "Follow-up"). The first transcript of a
phrase heard there goes to check() with the assistant's name (the session's wake word),
the last request and answer; a phrase that is not accepted is dropped without a sound.
The PIPELINE owns when the window opens and when check() is called.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class FollowupVerdict:
    accepted: bool
    probability: float | None   # the model's probability; None when the check failed


class FollowupGate(ABC):
    # How long (ms) the pipeline waits for speech to start after the answer.
    window_ms: int = 5000

    @abstractmethod
    async def check(self, name: str, request: str, reply: str,
                    utterance: str) -> FollowupVerdict:
        """Never raises: a failure or a timeout is FollowupVerdict(False, None)."""


class NoFollowup(FollowupGate):
    """Marker used when the stage is disabled: the pipeline opens no follow-up window."""

    window_ms = 0

    async def check(self, name: str, request: str, reply: str,
                    utterance: str) -> FollowupVerdict:
        return FollowupVerdict(accepted=False, probability=None)
