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
    # A wake-word run opens with this much pre-roll, which ends where the wake word
    # was detected; 0 = no pre-roll. Until speech follows it, the phrase ends only
    # after max(silence_ms, wake_pause_ms) of silence, so a pause between the wake
    # word and the command does not end the phrase. Any speech window counts, not
    # min_speech_ms of it: a one-word command («ночь») can be shorter than that.
    preroll_ms: int = 0
    wake_pause_ms: int = 0


class EndpointTracker:
    """Engine-independent end-pointing state machine shared by VAD
    sessions. A session classifies each fixed-size window as
    speech/non-speech (however its engine does that) and calls
    update(); the tracker accumulates the policy counters and decides
    when the utterance ended. Window duration is per-call, so engines
    with different frame sizes (WebRTC 20 ms, Silero 32 ms, TEN 16 ms)
    share the exact same decision semantics."""

    def __init__(self, policy: EndpointPolicy):
        self._policy = policy
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_detected = False
        self._elapsed_ms = 0
        self._command = False  # speech heard after the pre-roll, i.e. after the wake word

    def update(self, speech: bool, window_ms: int) -> str | None:
        """Consume one classified window; return the finalize reason
        ("endpoint" | "maxlen" | "no_speech") or None to keep listening."""
        self._elapsed_ms += window_ms
        if speech:
            self._speech_ms += window_ms
            if self._elapsed_ms > self._policy.preroll_ms:
                self._command = True
            self._silence_ms = 0
            if self._speech_ms >= self._policy.min_speech_ms:
                self._speech_detected = True
        else:
            # Trailing silence only counts once real speech has been observed.
            if self._speech_detected:
                self._silence_ms += window_ms

        # Decide end-of-utterance (reason or None) after the window,
        # in the same precedence order as before: endpoint, max length, no-speech.
        pause_ms = self._policy.silence_ms
        if not self._command:
            pause_ms = max(pause_ms, self._policy.wake_pause_ms)
        if self._speech_detected and self._silence_ms >= pause_ms:
            return "endpoint"
        if self._elapsed_ms >= self._policy.max_utterance_ms:
            return "maxlen"
        if not self._speech_detected and self._elapsed_ms >= self._policy.no_speech_timeout_ms:
            return "no_speech"
        return None

    def debug_state(self) -> dict:
        """speech_ms / silence_ms / elapsed_ms / speech_detected snapshot."""
        return {
            "speech_ms": self._speech_ms,
            "silence_ms": self._silence_ms,
            "elapsed_ms": self._elapsed_ms,
            "speech_detected": self._speech_detected,
        }


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
