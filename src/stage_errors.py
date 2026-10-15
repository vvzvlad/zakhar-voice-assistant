"""Uniform stage-failure contract shared by all pipeline stages."""


class StageError(Exception):
    """Uniform stage failure: backends raise this instead of sentinel
    return values. `kind` lets the orchestrator map specific failures
    (e.g. rate limits) to configured spoken phrases."""

    def __init__(self, stage: str, message: str, *, kind: str = "error"):
        super().__init__(message)
        self.stage = stage          # "vad" | "stt" | "llm" | "tts"
        self.kind = kind            # "error" | "rate_limit" | ...


class Declined(Exception):
    """A handler refuses an utterance it cannot take (it ran no tool), so the next
    recognition chain gets it. `reply` is what is spoken when no later chain answers."""

    def __init__(self, reply: str):
        super().__init__(reply)
        self.reply = reply
