"""OpenRouter LLM provider (OpenAI-compatible)."""

import time

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import LlmConfig

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Public model catalog; no auth required. Each entry carries "id" (the value the
# chat-completions API expects) and a human "name".
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Module-level TTL cache for the model list: the catalog weighs ~1MB and changes
# rarely, so reopening the settings page must not refetch it every time.
# Failures are never cached.
_MODELS_CACHE_TTL = 300.0
_models_cache: dict = {"at": 0.0, "data": None}


@register
class OpenRouterProvider(Provider):
    category = "llm"
    id = "openrouter"
    label = "OpenRouter"
    ConfigModel = LlmConfig
    uses_http_cloud = True

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

    def options(self, field: str, cfg: LlmConfig, deps: Deps, query: str = ""):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(deps)
        return None

    async def _fetch_models(self, deps: Deps):
        """Fetch the OpenRouter model catalog as [{"value", "label"}, ...], TTL-cached."""
        now = time.monotonic()
        if _models_cache["data"] is not None and now - _models_cache["at"] < _MODELS_CACHE_TTL:
            return _models_cache["data"]
        resp = await deps.http_cloud.get(OPENROUTER_MODELS_URL)
        resp.raise_for_status()
        models = resp.json().get("data") or []
        options = [
            {"value": m["id"], "label": m.get("name") or m["id"]}
            for m in models
        ]
        options.sort(key=lambda o: o["label"].lower())
        _models_cache["at"] = now
        _models_cache["data"] = options
        return options
