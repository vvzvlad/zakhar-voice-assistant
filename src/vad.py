"""VAD stage contract: speech/no-speech classification + end-pointing.

The PIPELINE owns the audio buffer, channel selection and memory caps;
a VadBackend owns everything about deciding when the utterance ended.
One VadSession is opened per voice run (sessions are stateful within a
run; backends are stateless factories), so a hot-swapped backend or a
changed policy simply applies on the next run.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# 16 kHz / 16-bit mono PCM is the device stream format the whole pipeline runs on.
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EndpointPolicy:
    """Generic end-pointing thresholds (engine-independent), read from
    core.vad per run by the pipeline and handed to open()."""

    silence_ms: int
    min_speech_ms: int
    max_utterance_ms: int
    no_speech_timeout_ms: int


class VadSession(ABC):
    """Per-utterance state. feed() consumes one PCM chunk (any size; the
    session does its own framing) and returns a finalize reason
    ("endpoint" | "maxlen" | "no_speech") or None to keep listening.
    Once a reason is returned the session is done; further feed() calls
    are undefined (the pipeline stops calling)."""

    @abstractmethod
    def feed(self, chunk: bytes) -> str | None: ...

    def debug_state(self) -> dict:
        """Small diagnostic dict for the pipeline's VAD-finalize debug log line.
        Optional: the default is empty so minimal sessions stay minimal."""
        return {}


class VadBackend(ABC):
    """Stateless session factory; one session per voice run."""

    @abstractmethod
    def open(self, policy: EndpointPolicy) -> VadSession: ...
