"""LLM backend interface and the shared provider config schema.

The agentic tool-calling loop stays in core (Phase 2). A backend here is a thin,
single round-trip chat-completions client; the loop drives it.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

# MODEL_FIELD_EXTRA now lives in src.plugins.base (shared across categories);
# re-exported here so existing `from src.plugins.llm.base import MODEL_FIELD_EXTRA`
# imports keep working.
from src.plugins.base import LABEL_FIELD_EXTRA, MODEL_FIELD_EXTRA  # noqa: F401


class LlmBackend(ABC):
    @abstractmethod
    async def complete(self, messages: list, tools: list | None) -> dict:
        """One chat-completions round-trip. Returns the raw provider JSON
        (with 'choices'/'usage'/'model'). Raises httpx.HTTPStatusError on non-2xx."""


class LlmConfig(BaseModel):
    # The per-field apply class is computed centrally by reconfig.action_for and injected
    # into the catalog schema by ConfigService.catalog() (single source of truth), so the
    # fields below carry no per-field "apply" annotation.
    api_key: str = ""
    model: str = Field("anthropic/claude-haiku-4.5", json_schema_extra=MODEL_FIELD_EXTRA)
    # Human label of the selected model (catalog display name), persisted so the panel
    # shows the model name immediately on load instead of the bare id. Hidden companion
    # field (see LABEL_FIELD_EXTRA). Not sent to the chat-completions API.
    model_label: str = Field("", json_schema_extra=LABEL_FIELD_EXTRA)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    max_tool_rounds: int = Field(5, ge=1)
    # Per-request HTTP timeout for one chat-completions round-trip (seconds).
    timeout: int = Field(300, json_schema_extra={"unit": "s"})
    # Spoken fallbacks (read per request). Never let an empty/null model reply or a
    # rate-limit reach TTS as "".
    reply_rate_limit: str = "Лимит запросов исчерпан. Попробуй ещё раз чуть позже."
    reply_empty_after_tools: str = "Готово."
    reply_empty: str = "Я тебя не расслышал, повтори."
    # Spoken when the LLM stage fails for any reason other than a rate limit; the
    # raw error text goes to the run log, never to the user.
    reply_error: str = "Что-то сломалось, попробуй ещё раз попозже."
