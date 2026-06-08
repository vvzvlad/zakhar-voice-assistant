"""TeraTTS HTTP provider (local service)."""

from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register


class TeraTtsConfig(BaseModel):
    base_url: str = ""


@register
class TeraTtsProvider(Provider):
    category = "tts"
    id = "teratts"
    label = "TeraTTS (HTTP)"
    ConfigModel = TeraTtsConfig

    def create(self, cfg: TeraTtsConfig, deps: Deps):
        from src.tts import TeraTtsHttpBackend

        return TeraTtsHttpBackend(cfg.base_url, deps.http_local, deps.tts_timeout)
