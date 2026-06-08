"""LLM backend interface and the shared provider config schema.

The agentic tool-calling loop stays in core (Phase 2). A backend here is a thin,
single round-trip chat-completions client; the loop drives it.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class LlmBackend(ABC):
    @abstractmethod
    async def complete(self, messages: list, tools: list | None) -> dict:
        """One chat-completions round-trip. Returns the raw provider JSON
        (with 'choices'/'usage'/'model'). Raises httpx.HTTPStatusError on non-2xx."""


class LlmConfig(BaseModel):
    api_key: str = Field("", json_schema_extra={"apply": "live"})
    model: str = "anthropic/claude-haiku-4.5"
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    max_tool_rounds: int = Field(5, ge=1)
    # Per-request HTTP timeout for one chat-completions round-trip (seconds).
    timeout: int = Field(300, json_schema_extra={"apply": "rebuild"})
    # Spoken fallbacks (read per request). Never let an empty/null model reply or a
    # rate-limit reach TTS as "".
    reply_rate_limit: str = Field(
        "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии",
        json_schema_extra={"apply": "live"},
    )
    reply_empty_after_tools: str = Field("Готово.", json_schema_extra={"apply": "live"})
    reply_empty: str = Field(
        "Я тебя не расслышала, повтори.", json_schema_extra={"apply": "live"}
    )
