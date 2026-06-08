"""Piper offline TTS provider."""

from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register


class PiperConfig(BaseModel):
    voice_path: str = "models/ru_RU-ruslan-medium.onnx"
    sentence_silence: float = 0.4


@register
class PiperProvider(Provider):
    category = "tts"
    id = "piper"
    label = "Piper (offline)"
    ConfigModel = PiperConfig

    def create(self, cfg: PiperConfig, deps: Deps):
        from src.tts import PiperTtsBackend

        return PiperTtsBackend(cfg.voice_path, sentence_silence=cfg.sentence_silence)
