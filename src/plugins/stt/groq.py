"""Groq Whisper STT brick: config schema, backend and Whisper-specific hacks."""

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend, pcm_to_wav

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Groq rejects transcription prompts longer than this many characters (HTTP 400).
# Truncate locally so an over-long vocabulary hint degrades gracefully instead of
# failing the whole utterance.
GROQ_PROMPT_MAX_CHARS = 896

# Known Whisper STT hallucination markers (lowercase). Whisper tends to emit
# leftover subtitle-credit / stock phrases (training-data artifacts) on
# silence/noise: "DimaTorzok" is one such credit string and "Продолжение
# следует..." ("to be continued") is a recurring stock caption. When a
# transcription contains one of these (substring, case-insensitive), we treat
# the run as if nothing was said and drop it.
STT_HALLUCINATION_MARKERS = ("dimatorzok", "продолжение следует")


def contains_stt_hallucination(text: str) -> bool:
    """Return True if the STT text contains a known Whisper hallucination marker.

    These are known STT (Whisper) hallucinations — subtitle-credit artifacts that
    surface on silence/noise — and are dropped as if nothing was said. The check is
    case-insensitive (Whisper varies the casing of the artifact).
    """
    folded = text.casefold()
    return any(marker in folded for marker in STT_HALLUCINATION_MARKERS)


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
        return GroqSttBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            model=cfg.model,
            language=cfg.language,
            temperature=cfg.temperature,
            prompt=cfg.prompt,
            timeout=cfg.timeout,
        )
