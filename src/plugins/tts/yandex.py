"""Yandex SpeechKit TTS provider."""

from pydantic import BaseModel, Field, model_validator

from src.plugins.base import Deps, Provider, register

# Static ru-RU catalog for Yandex SpeechKit v3 (utteranceSynthesis): voice -> list
# of selectable roles ("amplua"). Yandex exposes NO runtime "list voices" endpoint,
# so this catalog is hardcoded and must be kept in sync with the docs by hand
# (yandex.cloud/en/docs/speechkit/tts/voices, "API v3"). An empty list means the
# voice has no selectable role (only its built-in default; send NO role hint). The
# first role in each list is the voice's default role. Insertion order is the order
# shown to the user, so preserve it.
YANDEX_V3_VOICES: dict[str, list[str]] = {
    "alena": ["neutral", "good"],
    "filipp": [],
    "ermil": ["neutral", "good"],
    "jane": ["neutral", "good", "evil"],
    "omazh": ["neutral", "evil"],
    "zahar": ["neutral", "good"],
    "dasha": ["neutral", "good", "friendly"],
    "julia": ["neutral", "strict"],
    "lera": ["neutral", "friendly"],
    "masha": ["good", "strict", "friendly"],  # masha has NO neutral; default role is "good"
    "marina": ["neutral", "whisper", "friendly"],
    "alexander": ["neutral", "good"],
    "kirill": ["neutral", "strict", "good"],
    "anton": ["neutral", "good"],
    "madi_ru": [],
    "saule_ru": ["neutral", "strict", "whisper"],
    "zamira_ru": ["neutral", "strict", "friendly"],
    "zhanar_ru": ["neutral", "strict", "friendly"],
    "yulduz_ru": ["neutral", "strict", "friendly", "whisper"],
}


def _default_role(voice: str) -> str:
    """The role used when none is explicitly selected: the voice's first listed
    role (Yandex marks it as the default), or "" for voices without roles."""
    roles = YANDEX_V3_VOICES.get(voice, [])
    return roles[0] if roles else ""


class YandexTtsConfig(BaseModel):
    # apply class is computed centrally (reconfig.action_for) and injected by catalog().
    api_key: str = ""
    voice: str = Field("zahar", json_schema_extra={"widget": "select", "options": "dynamic"})
    # Role (amplua) is voice-dependent, so its option list is computed from the
    # selected voice (see options()); it is stored as a free string and coerced to
    # the voice's default by the validator below.
    role: str = Field("neutral", json_schema_extra={"widget": "select", "options": "dynamic"})
    speed: float = Field(1.0, ge=0.1, le=3.0, json_schema_extra={"widget": "slider"})
    url: str = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"

    @model_validator(mode="after")
    def _coerce_role(self):
        # Roles depend on the voice; a role left over from a previously selected
        # voice (or an unknown voice) would make Yandex reject the request. Keep the
        # value only when it is valid for the current voice, else fall back to the
        # voice's default role (possibly "").
        if self.role not in YANDEX_V3_VOICES.get(self.voice, []):
            self.role = _default_role(self.voice)
        return self


@register
class YandexTtsProvider(Provider):
    category = "tts"
    id = "yandex"
    label = "Yandex SpeechKit"
    ConfigModel = YandexTtsConfig
    uses_http_cloud = True

    def create(self, cfg: YandexTtsConfig, deps: Deps):
        from src.tts import YandexTtsBackend

        return YandexTtsBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            voice=cfg.voice,
            role=cfg.role,
            speed=cfg.speed,
            url=cfg.url,
            timeout=deps.tts_timeout,
        )

    def options(self, field: str, cfg: YandexTtsConfig, deps: Deps):
        if field == "voice":
            return list(YANDEX_V3_VOICES.keys())
        if field == "role":
            return list(YANDEX_V3_VOICES.get(cfg.voice, []))
        return None
