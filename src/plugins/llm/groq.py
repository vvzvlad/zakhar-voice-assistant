"""Groq LLM provider (OpenAI-compatible)."""

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import LlmConfig

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLlmConfig(LlmConfig):
    # Groq serves open-weight models; default to a fast, capable one.
    model: str = "llama-3.3-70b-versatile"


@register
class GroqLlmProvider(Provider):
    category = "llm"
    id = "groq"
    label = "Groq"
    ConfigModel = GroqLlmConfig

    def create(self, cfg: GroqLlmConfig, deps: Deps):
        return OpenAICompatLlmBackend(
            url=GROQ_API_URL,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=deps.http_cloud,
        )
