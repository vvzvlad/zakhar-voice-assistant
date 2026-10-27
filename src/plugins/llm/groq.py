"""Groq LLM provider (OpenAI-compatible)."""

import logging
import time

import httpx
from pydantic import Field

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import MODEL_FIELD_EXTRA, LlmConfig

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# OpenAI-style model listing; REQUIRES a Bearer api_key (401 otherwise).
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Module-level TTL cache for the model list. Keyed by the api_key that fetched it,
# so changing the key never serves a list obtained with another key. Failures are
# never cached. Holds the raw catalog entries; the options() ids and the 404
# fallback pick both derive from them.
_MODELS_CACHE_TTL = 300.0
_models_cache: dict = {"at": 0.0, "api_key": None, "data": None}


class GroqLlmConfig(LlmConfig):
    # Groq serves open-weight models; default to a fast, capable one.
    # The json_schema_extra must be re-attached: pydantic does not inherit Field
    # metadata on overridden fields (see MODEL_FIELD_EXTRA in src/plugins/base.py).
    model: str = Field("openai/gpt-oss-120b", title="Model", json_schema_extra=MODEL_FIELD_EXTRA)


class _GroqLlmBackend(OpenAICompatLlmBackend):
    """Retries once on another live model when Groq 404s the configured one: Groq
    decommissions models (llama-3.3-70b-versatile did), and every request then
    failed until the config was edited by hand."""

    def __init__(self, *args, swap_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._swap_model = swap_model  # async () -> live model id | None
        self._swapped = False

    async def complete(self, messages, tools, *, has_next=False):
        try:
            return await super().complete(messages, tools, has_next=has_next)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404 or self._swapped:
                raise
            first = e
        fallback = await self._swap_model() if self._swap_model else None
        if not fallback or fallback == self.model:
            raise first
        logger.warning("groq: model %s is gone (404), answering with %s", self.model, fallback)
        self.model = fallback
        self._swapped = True
        return await super().complete(messages, tools, has_next=has_next)


@register
class GroqLlmProvider(Provider):
    category = "llm"
    id = "groq"
    label = "Groq"
    ConfigModel = GroqLlmConfig
    uses_http_cloud = True

    def create(self, cfg: GroqLlmConfig, deps: Deps):
        return _GroqLlmBackend(
            url=GROQ_API_URL,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=deps.http_cloud,
            timeout=cfg.timeout,
            swap_model=lambda: self._fallback_model(cfg.api_key, deps),
        )

    def options(self, field: str, cfg: GroqLlmConfig, deps: Deps, query: str = ""):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(cfg.api_key, deps)
        return None

    async def _fetch_models(self, api_key: str, deps: Deps):
        """Fetch Groq model ids (plain strings, sorted), TTL-cached per api_key."""
        entries = await self._model_entries(api_key, deps)
        return sorted((m["id"] for m in entries), key=str.lower)

    async def _model_entries(self, api_key: str, deps: Deps):
        """The raw catalog entries, TTL-cached per api_key."""
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
        entries = resp.json().get("data") or []
        _models_cache.update({"at": now, "api_key": api_key, "data": entries})
        return entries

    def _chat_fallback(self, entries: list) -> str | None:
        """The live model a 404 swap moves to: the chat model with the largest
        context window. The catalog also holds Whisper (STT) and Orpheus (TTS),
        which a chat handler cannot use — the modality filter drops them; the
        tiny prompt-guard classifiers lose on context."""
        chat = [
            m for m in entries
            if "text" in (m.get("output_modalities") or [])
            and "audio" not in (m.get("input_modalities") or [])
        ]
        if not chat:
            return None
        return sorted(chat, key=lambda m: (-(m.get("context_window") or 0), m["id"]))[0]["id"]

    async def _fallback_model(self, api_key: str, deps: Deps):
        return self._chat_fallback(await self._model_entries(api_key, deps))
