"""Groq Whisper STT brick: config schema, backend and Whisper-specific hacks."""

import time

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import MODEL_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
# The hallucination filter now lives in src.stt (shared pure helpers); re-exported
# here so existing `from src.plugins.stt.groq import ...` imports keep working.
from src.stt import (  # noqa: F401
    STT_HALLUCINATION_MARKERS,
    SttBackend,
    contains_stt_hallucination,
    pcm_to_wav,
)

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# OpenAI-style model listing; REQUIRES a Bearer api_key (401 otherwise).
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Groq rejects transcription prompts longer than this many characters (HTTP 400).
# Truncate locally so an over-long vocabulary hint degrades gracefully instead of
# failing the whole utterance.
GROQ_PROMPT_MAX_CHARS = 896

# Module-level TTL cache for the model list. Keyed by the api_key that fetched it,
# so changing the key never serves a list obtained with another key. Failures are
# never cached. Own cache for this module — never shared with the Groq LLM provider.
_MODELS_CACHE_TTL = 300.0
_models_cache: dict = {"at": 0.0, "api_key": None, "data": None}


class GroqSttBackend(SttBackend):
    """Groq Whisper HTTP backend. Posts a WAV-wrapped PCM and returns the text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        language: str = "ru",
        temperature: float = 0.0,
        prompt: str = "",
        timeout: int = 60,
    ):
        self.client = client
        self.api_key = api_key
        self.model = model
        self.language = language
        self.temperature = temperature
        self.prompt = prompt
        self.timeout = timeout

    async def transcribe(self, pcm: bytes) -> str:
        """Transcribe raw 16 kHz/16-bit mono PCM via Groq Whisper.

        Returns the recognized text on success; "" on empty input (nothing to
        transcribe), a 200 with empty text (no speech recognized), or a known
        Whisper hallucination (discarded as if nothing was said). Raises
        StageError("stt", ...) on a non-200 response or any httpx error.
        """
        if not pcm:
            return ""

        wav_bytes = pcm_to_wav(pcm)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.model,
            "language": self.language,
            "response_format": "json",
            "temperature": str(self.temperature),
        }
        # Bias recognition toward specific words only when a hint is provided;
        # never send an empty prompt to the API. Groq caps the prompt at
        # GROQ_PROMPT_MAX_CHARS characters, so truncate over-long hints (and warn)
        # rather than letting the API reject the whole request.
        prompt = self.prompt.strip()
        if len(prompt) > GROQ_PROMPT_MAX_CHARS:
            logger.warning(
                f"Groq STT prompt is {len(prompt)} chars; truncating to {GROQ_PROMPT_MAX_CHARS}"
            )
            prompt = prompt[:GROQ_PROMPT_MAX_CHARS]
        if prompt:
            data["prompt"] = prompt
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = await self.client.post(
                GROQ_STT_URL, headers=headers, data=data, files=files, timeout=self.timeout
            )
            if resp.status_code == 200:
                text = resp.json().get("text", "").strip()
                # Whisper emits leftover subtitle-credit artifacts (e.g.
                # "DimaTorzok") on silence/noise. Discard such hallucinated text
                # and return "" — the SttBackend contract for "no speech
                # recognized" — so the pipeline ends the run like an empty result.
                if contains_stt_hallucination(text):
                    logger.info(f"Groq STT: discarding hallucination: {text!r}")
                    return ""
                return text
            logger.error(f"Groq STT error: {resp.status_code} - {resp.text}")
            raise StageError("stt", f"Groq STT error: {resp.status_code} - {resp.text}")
        except httpx.HTTPError as e:
            logger.error(f"Groq STT request failed: {str(e)}")
            raise StageError("stt", f"Groq STT request failed: {e}") from e
        except ValueError as e:
            # resp.json() on a malformed 200 body raises json.JSONDecodeError — a
            # ValueError subclass, NOT an httpx.HTTPError — so without this clause
            # it would leak raw out of transcribe(), violating the SttBackend
            # contract (StageError("stt", ...) on any failure). StageError itself
            # is a plain Exception (not a ValueError), so the non-200 raise above
            # still propagates untouched.
            logger.error(f"Groq STT malformed response: {str(e)}")
            raise StageError("stt", f"Groq STT malformed response: {e}") from e


class GroqSttConfig(BaseModel):
    api_key: str = ""
    # Dynamic select: the option list is fetched from Groq's model-list API
    # (see MODEL_FIELD_EXTRA in src/plugins/base.py).
    model: str = Field("whisper-large-v3-turbo", json_schema_extra=MODEL_FIELD_EXTRA)
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
        return GroqSttBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            model=cfg.model,
            language=cfg.language,
            temperature=cfg.temperature,
            prompt=cfg.prompt,
            timeout=cfg.timeout,
        )

    def options(self, field: str, cfg: GroqSttConfig, deps: Deps):
        # `options` stays sync; the model list is network-backed, so return a
        # coroutine — the caller (panel_api) awaits it (see Provider.options).
        if field == "model":
            return self._fetch_models(cfg.api_key, deps)
        return None

    async def _fetch_models(self, api_key: str, deps: Deps):
        """Fetch Groq whisper model ids (plain strings, sorted), TTL-cached per api_key."""
        if not api_key:
            return []  # the endpoint requires auth; don't even try
        now = time.monotonic()
        if (
            _models_cache["data"] is not None
            and _models_cache["api_key"] == api_key
            and now - _models_cache["at"] < _MODELS_CACHE_TTL
        ):
            return _models_cache["data"]
        resp = await deps.http_cloud.get(
            GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}
        )
        resp.raise_for_status()
        models = resp.json().get("data") or []
        # Groq serves LLMs AND whisper models in one list; only whisper models are
        # valid for the transcriptions endpoint, so filter the rest out.
        options = sorted(
            (m["id"] for m in models if "whisper" in m["id"].lower()), key=str.lower
        )
        _models_cache["at"] = now
        _models_cache["api_key"] = api_key
        _models_cache["data"] = options
        return options
