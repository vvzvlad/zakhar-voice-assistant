"""LLM provider for any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI,
DeepSeek, ...): the base URL, key and model are all set in the config."""

from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.plugins.llm._openai_compat import OpenAICompatLlmBackend
from src.plugins.llm.base import MODEL_FIELD_EXTRA, LlmConfig


class _EndpointConfig(BaseModel):
    url: str = Field(
        "",
        title="Endpoint URL",
        description="Base URL of an OpenAI-compatible API, e.g. http://host:11434/v1; "
        "requests go to <URL>/chat/completions, the model list comes from <URL>/models.",
    )


# _EndpointConfig is the LAST base so pydantic lists `url` first and the panel renders
# the endpoint above the inherited key and model fields.
class CustomLlmConfig(LlmConfig, _EndpointConfig):
    # No sensible default model for an unknown server. The json_schema_extra must be
    # re-attached: pydantic does not inherit Field metadata on overridden fields.
    model: str = Field("", title="Model", json_schema_extra=MODEL_FIELD_EXTRA)


def _base(url: str) -> str:
    return url.strip().rstrip("/")


@register
class CustomLlmProvider(Provider):
    category = "llm"
    id = "custom"
    label = "Custom (OpenAI-compatible)"
    ConfigModel = CustomLlmConfig
    uses_http_cloud = True

    def create(self, cfg: CustomLlmConfig, deps: Deps):
        return OpenAICompatLlmBackend(
            url=f"{_base(cfg.url)}/chat/completions",
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            client=deps.http_cloud,
            timeout=cfg.timeout,
        )

    def options(self, field: str, cfg: CustomLlmConfig, deps: Deps, query: str = ""):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(cfg, deps)
        return None

    async def _fetch_models(self, cfg: CustomLlmConfig, deps: Deps):
        """Fetch the server's model ids (plain strings, sorted) from <URL>/models."""
        if not _base(cfg.url):
            return []
        # Local servers usually run without a key; send one only when it is set.
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        resp = await deps.http_cloud.get(f"{_base(cfg.url)}/models", headers=headers)
        resp.raise_for_status()
        models = resp.json().get("data") or []
        return sorted((m["id"] for m in models), key=str.lower)
