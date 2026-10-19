"""Transport-neutral pipeline stage progress events.

The pipeline emits these; each front (the device server, tests)
injects its own translator via Pipeline.send_event. Nothing in the
pipeline knows about any concrete transport protocol."""
from enum import Enum, auto


class StageEvent(Enum):
    RUN_START = auto()
    STT_START = auto()
    STT_VAD_START = auto()  # device state: listening
    STT_VAD_END = auto()    # device state: transcribing (end of phrase)
    STT_END = auto()        # data: {"text": str}
    INTENT_START = auto()
    INTENT_END = auto()     # data: {"conversation_id"}
    TTS_START = auto()      # data: {"text": str}
    ERROR = auto()          # data: {"code", "message"}
    RUN_END = auto()
