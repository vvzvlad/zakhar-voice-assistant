"""Groq LLM provider (OpenAI-compatible)."""

import time

from pydantic import Field

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import MODEL_FIELD_EXTRA, LlmConfig

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# OpenAI-style model listing; REQUIRES a Bearer api_key (401 otherwise).
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Module-level TTL cache for the model list. Keyed by the api_key that fetched it,
# so changing the key never serves a list obtained with another key. Failures are
# never cached.
_MODELS_CACHE_TTL = 300.0
_models_cache: dict = {"at": 0.0, "api_key": None, "data": None}


class GroqLlmConfig(LlmConfig):
    # Groq serves open-weight models; default to a fast, capable one.
    # The json_schema_extra must be re-attached: pydantic does not inherit Field
    # metadata on overridden fields (see MODEL_FIELD_EXTRA in src/plugins/base.py).
    model: str = Field("llama-3.3-70b-versatile", json_schema_extra=MODEL_FIELD_EXTRA)


@register
class GroqLlmProvider(Provider):
    category = "llm"
    id = "groq"
    label = "Groq"
    ConfigModel = GroqLlmConfig
    uses_http_cloud = True

    def create(self, cfg: GroqLlmConfig, deps: Deps):
        return OpenAICompatLlmBackend(
            url=GROQ_API_URL,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=deps.http_cloud,
            timeout=cfg.timeout,
        )

    def options(self, field: str, cfg: GroqLlmConfig, deps: Deps, query: str = ""):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(cfg.api_key, deps)
        return None

    async def _fetch_models(self, api_key: str, deps: Deps):
        """Fetch Groq model ids (plain strings, sorted), TTL-cached per api_key."""
        if not api_key:
            return []  # the endpoint requires auth; don't even try
        now = time.monotonic()
        if (
            _models_cache["data"] is not None
            and _models_cache["api_key"] == api_key
            and now - _models_cache["at"] < _MODELS_CACHE_TTL
        ):
            return _models_cache["data"]
        resp = await deps.http_cloud.get(
            GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        models = resp.json().get("data") or []
        options = sorted((m["id"] for m in models), key=str.lower)
        _models_cache["at"] = now
        _models_cache["api_key"] = api_key
        _models_cache["data"] = options
        return options
