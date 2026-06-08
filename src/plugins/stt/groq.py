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

        # TODO(phase2): GroqSttBackend currently reads settings.stt_api_key /
        # settings.stt_model internally. Phase 2 will pass cfg.api_key/cfg.model
        # explicitly so this provider no longer depends on the global settings.
        return GroqSttBackend(deps.http_cloud)
