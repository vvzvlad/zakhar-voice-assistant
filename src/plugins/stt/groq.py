"""Groq Whisper STT provider (cloud)."""

from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register


class GroqSttConfig(BaseModel):
    api_key: str = ""
    model: str = "whisper-large-v3-turbo"
    language: str = "ru"
    temperature: float = 0.0
    timeout: int = 60
    prompt: str = Field(
        "",
        title="Recognition vocabulary hint",
        json_schema_extra={"widget": "textarea", "maxLength": 896},
        description="Optional text passed to Whisper to bias recognition toward specific words — names, brands, places, technical terms. Write the words the way they should be spelled, in the audio language (e.g. a comma-separated list). Max 896 characters; longer text is truncated automatically.",
    )


@register
class GroqSttProvider(Provider):
    category = "stt"
    id = "groq"
    label = "Groq Whisper"
    ConfigModel = GroqSttConfig
    uses_http_cloud = True

    def create(self, cfg: GroqSttConfig, deps: Deps):
        from src.stt import GroqSttBackend

        return GroqSttBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            model=cfg.model,
            language=cfg.language,
            temperature=cfg.temperature,
            prompt=cfg.prompt,
            timeout=cfg.timeout,
        )
