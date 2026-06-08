"""OpenRouter LLM provider (OpenAI-compatible)."""

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import LlmConfig

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@register
class OpenRouterProvider(Provider):
    category = "llm"
    id = "openrouter"
    label = "OpenRouter"
    ConfigModel = LlmConfig

    def create(self, cfg: LlmConfig, deps: Deps):
        return OpenAICompatLlmBackend(
            url=OPENROUTER_API_URL,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=deps.http_cloud,
            extra_headers={"X-Title": "Zakhar Voice Assistant"},
            timeout=cfg.timeout,
        )
