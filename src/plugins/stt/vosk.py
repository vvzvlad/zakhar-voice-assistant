"""Vosk offline STT provider."""

from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register


class VoskSttConfig(BaseModel):
    model_path: str = "models/vosk-model-small-ru-0.22"


@register
class VoskSttProvider(Provider):
    category = "stt"
    id = "vosk"
    label = "Vosk (offline)"
    ConfigModel = VoskSttConfig

    def create(self, cfg: VoskSttConfig, deps: Deps):
        from src.stt import VoskSttBackend

        return VoskSttBackend(cfg.model_path)
