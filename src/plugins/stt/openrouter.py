"""OpenRouter STT brick: config schema and backend for the transcriptions endpoint."""

import base64
import time

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import MODEL_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend, contains_stt_hallucination, pcm_to_wav

OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
# Public model catalog filtered server-side to transcription-capable models;
# no auth required. Each entry carries "id" (the value the transcriptions API
# expects) and a human "name".
OPENROUTER_STT_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=transcription"

# Module-level TTL cache for the model list: the catalog changes rarely, so
# reopening the settings page must not refetch it every time. Failures are never
# cached. Own cache for this module — never shared with the OpenRouter LLM provider.
_MODELS_CACHE_TTL = 300.0
_models_cache: dict = {"at": 0.0, "data": None}


class OpenRouterSttBackend(SttBackend):
    """OpenRouter transcriptions HTTP backend. Posts base64 WAV JSON and returns the text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        language: str = "ru",
        temperature: float = 0.0,
        timeout: int = 60,
    ):
        self.client = client
        self.api_key = api_key
        self.model = model
        self.language = language
        self.temperature = temperature
        self.timeout = timeout

    async def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw 16 kHz/16-bit mono PCM via OpenRouter transcriptions.

        Returns the recognized text on success; "" on empty input (nothing to
        transcribe), a 200 with empty text (no speech recognized), or a known
        Whisper hallucination (discarded as if nothing was said). Raises
        StageError("stt", ...) on a non-200 response or any httpx error.
        """
        if not pcm:
            return ""

        wav_bytes = pcm_to_wav(pcm)
        # The endpoint takes the audio as raw base64 bytes (not a data URI) in a
        # JSON body. There is no "prompt" field on this endpoint.
        payload: dict = {
            "model": self.model,
            "input_audio": {
                "data": base64.b64encode(wav_bytes).decode("ascii"),
                "format": "wav",
            },
            "temperature": self.temperature,
        }
        # `language` is optional (ISO-639-1); omit it entirely when not configured.
        if self.language:
            payload["language"] = self.language
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-Title": "Zakhar Voice Assistant",
        }

        try:
            resp = await self.client.post(
                OPENROUTER_STT_URL, headers=headers, json=payload, timeout=self.timeout
            )
            if resp.status_code == 200:
                text = resp.json().get("text", "").strip()
                # The catalog is whisper-family heavy, so the same subtitle-credit
                # artifacts surface on silence/noise. Discard such hallucinated text
                # and return "" — the SttBackend contract for "no speech
                # recognized" — so the pipeline ends the run like an empty result.
                if contains_stt_hallucination(text):
                    logger.info(f"OpenRouter STT: discarding hallucination: {text!r}")
                    return ""
                return text
            logger.error(f"OpenRouter STT error: {resp.status_code} - {resp.text}")
            raise StageError("stt", f"OpenRouter STT error: {resp.status_code} - {resp.text}")
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter STT request failed: {str(e)}")
            raise StageError("stt", f"OpenRouter STT request failed: {e}") from e
        except ValueError as e:
            # resp.json() on a malformed 200 body raises json.JSONDecodeError — a
            # ValueError subclass, NOT an httpx.HTTPError — so without this clause
            # it would leak raw out of transcribe(), violating the SttBackend
            # contract (StageError("stt", ...) on any failure). StageError itself
            # is a plain Exception (not a ValueError), so the non-200 raise above
            # still propagates untouched.
            logger.error(f"OpenRouter STT malformed response: {str(e)}")
            raise StageError("stt", f"OpenRouter STT malformed response: {e}") from e


class OpenRouterSttConfig(BaseModel):
    api_key: str = ""
    # Dynamic select: the option list is fetched from OpenRouter's model catalog
    # (see MODEL_FIELD_EXTRA in src/plugins/base.py).
    model: str = Field("openai/whisper-large-v3-turbo", json_schema_extra=MODEL_FIELD_EXTRA)
    language: str = "ru"
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    timeout: int = 60


@register
class OpenRouterSttProvider(Provider):
    category = "stt"
    id = "openrouter"
    label = "OpenRouter"
    ConfigModel = OpenRouterSttConfig
    uses_http_cloud = True

    def create(self, cfg: OpenRouterSttConfig, deps: Deps):
        return OpenRouterSttBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            model=cfg.model,
            language=cfg.language,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )

    def options(self, field: str, cfg: OpenRouterSttConfig, deps: Deps):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(deps)
        return None

    async def _fetch_models(self, deps: Deps):
        """Fetch the OpenRouter STT model catalog as [{"value", "label"}, ...], TTL-cached."""
        now = time.monotonic()
        if _models_cache["data"] is not None and now - _models_cache["at"] < _MODELS_CACHE_TTL:
            return _models_cache["data"]
        resp = await deps.http_cloud.get(OPENROUTER_STT_MODELS_URL)
        resp.raise_for_status()
        models = resp.json().get("data") or []
        options = [
            {"value": m["id"], "label": m.get("name") or m["id"]}
            for m in models
        ]
        options.sort(key=lambda o: o["label"].lower())
        _models_cache["at"] = now
        _models_cache["data"] = options
        return options
