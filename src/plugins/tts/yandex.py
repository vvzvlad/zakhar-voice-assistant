"""Yandex SpeechKit TTS brick: config schema, voice catalog and v3 backend."""

import base64
import json

import httpx
from pydantic import BaseModel, Field, model_validator

from src.plugins.base import Deps, Provider, register
# The canonical LLM->TTS text is the model's own notation: plain text with "+"
# before the stressed vowel (e.g. "прив+ет") — Yandex's native stress markup, so
# only unit expansion and stray-'+' cleanup are needed.
from src.plugins.tts._ru_text import expand_units, sanitize_plus_stress
from src.tts import TtsBackend, split_sentences

# Yandex SpeechKit v3 utteranceSynthesis rejects requests whose text exceeds 250
# characters (and ~24 s of audio, but 250 chars is the binding limit). Long replies
# must be split into <=250-char parts, synthesized separately, and concatenated.
# Source: yandex.cloud/docs/speechkit limits, API v3.
YANDEX_V3_TEXT_LIMIT = 250


def _split_oversized(fragment: str, limit: int) -> list[str]:
    """Split a single over-limit fragment into <=limit pieces on word boundaries.
    A single word longer than the limit is hard-sliced (rare; may break a
    Yandex "+vowel" stress pair, acceptable for such pathological input)."""
    out: list[str] = []
    cur = ""
    for word in fragment.split():
        if len(word) > limit:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(word), limit):
                out.append(word[i:i + limit])
            continue
        candidate = f"{cur} {word}" if cur else word
        if len(candidate) <= limit:
            cur = candidate
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _chunk_for_v3(text: str, limit: int = YANDEX_V3_TEXT_LIMIT) -> list[str]:
    """Split already-stress-marked text into request chunks, each <=limit chars.
    Packs whole sentences greedily; an over-limit sentence is split on words.
    Returns [] for empty / punctuation-only input."""
    # split_sentences already drops fragments with no word char; inputs with zero
    # word characters (pure punctuation/emoji) are unvoiceable and rejected by
    # Yandex with 400, so returning [] for them (empty audio, no request) is correct.
    sentences = split_sentences(text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_oversized(s, limit))
            continue
        candidate = f"{cur} {s}" if cur else s
        if len(candidate) <= limit:
            cur = candidate
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def _decode_v3_audio(body: str) -> bytes:
    """Reassemble audio from a SpeechKit v3 utteranceSynthesis response.

    The REST response is a stream of JSON objects (newline-delimited or
    concatenated), each shaped like {"result": {"audioChunk": {"data": "<base64>"}}}.
    Decode and concatenate every audio chunk; an {"error": ...} object raises.
    Tolerant to a single object, NDJSON, or a JSON array of objects.
    """
    chunks = bytearray()
    decoder = json.JSONDecoder()
    idx, length = 0, len(body)
    while idx < length:
        while idx < length and body[idx] in " \r\n\t":
            idx += 1
        if idx >= length:
            break
        message, idx = decoder.raw_decode(body, idx)
        for obj in (message if isinstance(message, list) else [message]):
            if not isinstance(obj, dict):
                continue
            if "error" in obj:
                raise RuntimeError(f"Yandex TTS v3 error: {obj['error']}")
            data = (obj.get("result") or {}).get("audioChunk", {}).get("data")
            if data:
                chunks.extend(base64.b64decode(data))
    return bytes(chunks)


class YandexTtsBackend(TtsBackend):
    """Yandex SpeechKit v3 cloud TTS (utteranceSynthesis). The v3 REST endpoint is
    server-streaming: the response is a stream of JSON objects, each carrying a
    base64-encoded MP3 chunk; the chunks are decoded and concatenated into a valid
    MP3 (audio/mpeg), so no transcoding is needed. Auth uses an API key bound to a
    service account (`Authorization: Api-Key <key>`). The input text already
    arrives in Yandex's native "+vowel" stress notation (the canonical LLM->TTS
    contract), so no stress conversion is needed — only unit expansion and
    dropping stray '+' signs."""

    def __init__(self, client, *, api_key, voice, role, speed, url, timeout):
        if not api_key:
            raise ValueError("YANDEX_TTS_API_KEY is required when TTS_BACKEND=yandex")
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.role = role
        self.speed = speed
        self.url = url
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        # v3 caps each request at YANDEX_V3_TEXT_LIMIT chars; adapt the canonical
        # "+vowel" text once (expand units, drop stray '+'), then split into
        # bounded chunks, synthesize each, and concatenate the MP3 audio.
        marked = sanitize_plus_stress(expand_units(text))
        chunks = _chunk_for_v3(marked, YANDEX_V3_TEXT_LIMIT)
        # Nothing pronounceable (empty / punctuation-only) -> serve no audio, don't POST.
        audio = bytearray()
        for chunk in chunks:
            audio.extend(await self._synthesize_chunk(chunk))
        return ("audio/mpeg", bytes(audio))

    async def _synthesize_chunk(self, text: str) -> bytes:
        # `text` is already adapted (units expanded, stray '+' dropped) and within
        # the length limit; it carries Yandex-native "+vowel" stress markup as-is.
        # v3 carries voice/role/speed as "hints"; the role hint is sent only when a
        # role is configured (voices without an amplua reject an empty role).
        hints = [{"voice": self.voice}, {"speed": self.speed}]
        if self.role:
            hints.insert(1, {"role": self.role})
        payload = {
            "text": text,
            "hints": hints,
            "outputAudioSpec": {"containerAudio": {"containerAudioType": "MP3"}},
            "loudnessNormalizationType": "LUFS",
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        resp = await self.client.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface Yandex's diagnostic body (it names the real cause, e.g. text too
            # long / bad voice / bad role); raise_for_status() alone hides it. Same
            # philosophy as src/llm.py logging status + body.
            raise RuntimeError(
                f"Yandex TTS v3 {resp.status_code}: {resp.text[:500]}"
            ) from e
        return _decode_v3_audio(resp.text)


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
