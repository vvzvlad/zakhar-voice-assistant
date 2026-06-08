"""Yandex SpeechKit TTS provider."""

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from src.plugins.base import Deps, Provider, register

# Male/female RU voices offered by Yandex SpeechKit v1.
YANDEX_VOICES = ["zahar", "filipp", "ermil", "jane", "omazh", "alena", "madirus"]


class YandexTtsConfig(BaseModel):
    api_key: SecretStr = Field(
        default=SecretStr(""), json_schema_extra={"secret": True, "apply": "rebuild"}
    )
    voice: str = Field("zahar", json_schema_extra={"widget": "select", "options": "dynamic"})
    emotion: Literal["neutral", "good", "evil"] = "neutral"
    speed: float = Field(1.0, ge=0.1, le=3.0, json_schema_extra={"widget": "slider"})
    folder_id: str = ""
    url: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"


@register
class YandexTtsProvider(Provider):
    category = "tts"
    id = "yandex"
    label = "Yandex SpeechKit"
    ConfigModel = YandexTtsConfig

    def create(self, cfg: YandexTtsConfig, deps: Deps):
        from src.tts import YandexTtsBackend

        return YandexTtsBackend(
            deps.http_cloud,
            api_key=cfg.api_key.get_secret_value(),
            voice=cfg.voice,
            emotion=cfg.emotion,
            speed=cfg.speed,
            folder_id=cfg.folder_id,
            url=cfg.url,
            timeout=deps.tts_timeout,
        )

    def options(self, field: str, cfg: YandexTtsConfig, deps: Deps):
        return list(YANDEX_VOICES) if field == "voice" else None
