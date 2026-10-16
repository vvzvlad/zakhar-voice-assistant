"""LLM backend interface and the shared provider config schema.

The agentic tool-calling loop stays in core (Phase 2). A backend here is a thin,
single round-trip chat-completions client; the loop drives it.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

# MODEL_FIELD_EXTRA now lives in src.plugins.base (shared across categories);
# re-exported here so existing `from src.plugins.llm.base import MODEL_FIELD_EXTRA`
# imports keep working.
from src.plugins.base import (  # noqa: F401
    ADVANCED_FIELD_EXTRA,
    LABEL_FIELD_EXTRA,
    MODEL_FIELD_EXTRA,
    SECRET_FIELD_EXTRA,
)


class LlmBackend(ABC):
    @abstractmethod
    async def complete(self, messages: list, tools: list | None, *, has_next: bool = False) -> dict:
        """One chat-completions round-trip. Returns the raw provider JSON
        (with 'choices'/'usage'/'model'). Raises httpx.HTTPStatusError on non-2xx.
        `has_next` is True when a later recognition chain would take an utterance
        this handler declines (by raising stage_errors.Declined)."""


class LlmConfig(BaseModel):
    # The per-field apply class is computed centrally by reconfig.action_for and injected
    # into the catalog schema by ConfigService.catalog() (single source of truth), so the
    # fields below carry no per-field "apply" annotation.
    api_key: str = Field("", title="API key", json_schema_extra=SECRET_FIELD_EXTRA)
    model: str = Field("anthropic/claude-haiku-4.5", title="Model", json_schema_extra=MODEL_FIELD_EXTRA)
    # Human label of the selected model (catalog display name), persisted so the panel
    # shows the model name immediately on load instead of the bare id. Hidden companion
    # field (see LABEL_FIELD_EXTRA). Not sent to the chat-completions API.
    model_label: str = Field("", json_schema_extra=LABEL_FIELD_EXTRA)
    temperature: float = Field(0.8, ge=0.0, le=2.0, title="Temperature")
    max_tokens: int = Field(4096, ge=1, title="Max reply tokens",
                            json_schema_extra=ADVANCED_FIELD_EXTRA)
    max_tool_rounds: int = Field(
        5, ge=1,
        title="Max rounds",
        description="Model calls per request, tool-calling rounds included.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    # Per-request HTTP timeout for one chat-completions round-trip (seconds).
    timeout: int = Field(300, title="Request timeout",
                         json_schema_extra={**ADVANCED_FIELD_EXTRA, "unit": "s"})
    # Spoken fallbacks (read per request). Never let an empty/null model reply or a
    # rate-limit reach TTS as "".
    reply_rate_limit: str = Field(
        "Лимит запросов исчерпан. Попробуй ещё раз чуть позже.",
        title="Reply on rate limit", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_empty_after_tools: str = Field(
        "Готово.",
        title="Reply when empty after a tool",
        description="Spoken when the model returns no text after a tool ran.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_empty: str = Field(
        "Я тебя не расслышал, повтори.",
        title="Reply when empty",
        description="Spoken when the model returns no text.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    # Spoken when the LLM stage fails for any reason other than a rate limit; the
    # raw error text goes to the run log, never to the user.
    reply_error: str = Field(
        "Что-то сломалось, попробуй ещё раз попозже.",
        title="Reply on error", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    # Spoken when the LLM stage fails AFTER a tool already ran: the action happened,
    # so no later chain retries it and the user hears that only the answer is missing.
    reply_error_after_tools: str = Field(
        "Сделал, но ответить не смог.",
        title="Reply on error after a tool",
        description="Spoken when the handler fails after a tool already ran: the action "
        "happened, only the answer is missing.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
