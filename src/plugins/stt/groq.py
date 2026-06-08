"""Groq Whisper STT provider (cloud)."""

from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register


class GroqSttConfig(BaseModel):
    api_key: str = ""
    model: str = "whisper-large-v3-turbo"


@register
class GroqSttProvider(Provider):
    category = "stt"
    id = "groq"
    label = "Groq Whisper"
    ConfigModel = GroqSttConfig

    def create(self, cfg: GroqSttConfig, deps: Deps):
        from src.stt import GroqSttBackend

        return GroqSttBackend(deps.http_cloud, api_key=cfg.api_key, model=cfg.model)
