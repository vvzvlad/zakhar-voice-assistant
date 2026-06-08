"""Per-voice-run ambient context shared with in-process tools.

A ContextVar lets a built-in tool know which speaker the current run belongs to
WITHOUT exposing that as an LLM-visible tool argument. asyncio runs each speaker's
voice run in its own task (its own context copy), so values never leak between
concurrent speakers; within one run the value is visible across awaits.
"""
import contextvars

# Name of the speaker the current voice run belongs to; None outside a run.
current_device: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_device", default=None
)
