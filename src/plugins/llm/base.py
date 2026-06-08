"""LLM backend interface and the shared provider config schema.

The agentic tool-calling loop stays in core (Phase 2). A backend here is a thin,
single round-trip chat-completions client; the loop drives it.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, SecretStr


class LlmBackend(ABC):
    @abstractmethod
    async def complete(self, messages: list, tools: list | None) -> dict:
        """One chat-completions round-trip. Returns the raw provider JSON
        (with 'choices'/'usage'/'model'). Raises httpx.HTTPStatusError on non-2xx."""


class LlmConfig(BaseModel):
    api_key: SecretStr = Field(
        default=SecretStr(""), json_schema_extra={"secret": True, "apply": "live"}
    )
    model: str = "anthropic/claude-haiku-4.5"
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1)
    max_tool_rounds: int = Field(5, ge=1)
