"""Fish Audio cloud TTS brick: config schema, voice catalog and backend."""

import re
import time

import httpx
from pydantic import BaseModel, Field

from src.plugins.base import Deps, LABEL_FIELD_EXTRA, SECRET_FIELD_EXTRA, Provider, register
# The canonical LLM->TTS text arrives in "+vowel" stress notation (e.g.
# "прив+ет"). Fish Audio's neural models take plain unannotated text — they do
# not understand Yandex "+" markup, and a literal '+' could be voiced — so the
# stress markup is stripped entirely (plain vowel remains) after unit expansion.
from src.plugins.tts._ru_text import drop_plus_stress, expand_units
from src.tts import TtsBackend

# TTS synthesis endpoint; the same for every deployment, so it is hardcoded
# rather than configurable (default cloud address of a public third-party API).
FISH_TTS_URL = "https://api.fish.audio/v1/tts"
# Voice-model catalog endpoint (same Bearer auth as TTS).
FISH_MODELS_URL = "https://api.fish.audio/model"

# Module-level TTL cache for the voice catalog, keyed by api_key: half of the
# merged list (`self=true`) differs per account, so a list fetched with one key
# must never be served for another. Failures are never cached.
_VOICES_CACHE_TTL = 300.0
_voices_cache: dict[str, dict] = {}  # api_key -> {"at": monotonic, "data": [...]}


def _to_options(items: list[dict]) -> list[dict]:
    """Map catalog items to [{"value", "label"}, ...], deduped by model id;
    the label is "title [lang,lang]" (the bare id when the title is empty)."""
    options: list[dict] = []
    seen: set[str] = set()
    for item in items:
        model_id = item["_id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        title = item.get("title") or model_id
        languages = item.get("languages") or []
        label = f"{title} [{','.join(languages)}]" if languages else title
        options.append({"value": model_id, "label": label})
    return options


class FishAudioTtsBackend(TtsBackend):
    """Fish Audio cloud TTS (api.fish.audio). One JSON POST per synthesize call:
    the server chunks long text internally, so no client-side splitting is
    needed; the 200 response body is the raw MP3 audio. The TTS model
    generation ("s1" / "s2-pro") is selected via the `model` HTTP header; the
    voice is a `reference_id` voice-model id (omitted -> provider default)."""

    def __init__(self, client, *, api_key, reference_id, model, speed, timeout):
        if not api_key:
            raise ValueError(
                "Fish Audio TTS api_key is required (set tts.instances.fishaudio.api_key in data/config.json)"
            )
        self.client = client
        self.api_key = api_key
        self.reference_id = reference_id
        self.model = model
        self.speed = speed
        self.timeout = timeout

    def _build_payload(self, text: str) -> tuple[dict, dict] | None:
        """Adapt the canonical "+vowel" text and build (headers, payload) for the
        TTS POST. Returns None for unvoiceable input (empty / punctuation-only):
        serve no audio, don't POST. Shared by synthesize and synthesize_stream."""
        # Expand units, then strip the stress markup entirely — Fish Audio takes
        # plain text.
        adapted = drop_plus_stress(expand_units(text))
        if re.search(r"\w", adapted, re.UNICODE) is None:
            return None
        payload: dict = {
            "text": adapted,
            "format": "mp3",
            "prosody": {"speed": self.speed},
        }
        # Omit reference_id entirely when not configured: fish.audio then uses
        # its default voice.
        if self.reference_id:
            payload["reference_id"] = self.reference_id
        headers = {"Authorization": f"Bearer {self.api_key}", "model": self.model}
        return headers, payload

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        prepared = self._build_payload(text)
        if prepared is None:
            return ("audio/mpeg", b"")
        headers, payload = prepared
        resp = await self.client.post(FISH_TTS_URL, headers=headers, json=payload, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface Fish Audio's diagnostic body (401/402/422 carry the real
            # cause, e.g. bad key / no credit / bad reference_id);
            # raise_for_status() alone hides it.
            raise RuntimeError(
                f"Fish Audio TTS {resp.status_code}: {resp.text[:500]}"
            ) from e
        return ("audio/mpeg", resp.content)

    async def synthesize_stream(self, text: str, lang: str = "ru"):
        """Native chunked synthesis: fish.audio streams the MP3 body progressively,
        so chunks are yielded as they arrive. HTTP/auth errors raise HERE (before
        the iterator is returned), per the TtsBackend streaming contract."""
        prepared = self._build_payload(text)
        if prepared is None:
            # Unvoiceable input: empty stream, no request (same as synthesize).
            async def _empty():
                return
                yield  # pragma: no cover - marks this as an async generator

            return ("audio/mpeg", _empty())
        headers, payload = prepared
        req = self.client.build_request(
            "POST", FISH_TTS_URL, headers=headers, json=payload, timeout=self.timeout
        )
        resp = await self.client.send(req, stream=True)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Same RuntimeError shaping as synthesize(), but the body must be
            # read explicitly (streamed response) and the response ALWAYS closed
            # on the error path.
            await resp.aread()
            await resp.aclose()
            raise RuntimeError(
                f"Fish Audio TTS {resp.status_code}: {resp.text[:500]}"
            ) from e

        # The open streamed response is released by the generator's finally,
        # which only runs if the generator is iterated or aclose()d. The caller
        # (Pipeline.serve_audio_stream) owns the returned iterator and
        # guarantees exactly that — fully consumed or explicitly closed.
        async def _gen():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return ("audio/mpeg", _gen())


class FishAudioTtsConfig(BaseModel):
    api_key: str = Field("", json_schema_extra=SECRET_FIELD_EXTRA)
    # Voice model id. The option list is fetched from the fish.audio catalog
    # and is server-searchable ("search": "remote"): the panel re-queries the
    # catalog with the typed text instead of filtering the preloaded list.
    # Freeform: any model id copied from the site may be pasted. Empty
    # string = provider default voice.
    reference_id: str = Field("", json_schema_extra={"widget": "select", "options": "dynamic", "freeform": True, "search": "remote"})
    # Human label of the selected reference_id (catalog title), persisted so the panel
    # shows the voice name immediately on load instead of the bare id. Hidden companion
    # field (see LABEL_FIELD_EXTRA). Not sent to the TTS API.
    reference_id_label: str = Field("", json_schema_extra=LABEL_FIELD_EXTRA)
    # TTS model generation, sent as the `model` HTTP header. Freeform so future
    # model ids work without a code change.
    model: str = Field("s2-pro", json_schema_extra={"widget": "select", "options": "dynamic", "freeform": True})
    # Maps to prosody.speed; fish.audio's documented range is 0.5–2.0.
    speed: float = Field(1.0, ge=0.5, le=2.0, json_schema_extra={"widget": "slider"})


@register
class FishAudioTtsProvider(Provider):
    category = "tts"
    id = "fishaudio"
    label = "Fish Audio"
    ConfigModel = FishAudioTtsConfig
    uses_http_cloud = True

    def create(self, cfg: FishAudioTtsConfig, deps: Deps):
        return FishAudioTtsBackend(
            deps.http_cloud,
            api_key=cfg.api_key,
            reference_id=cfg.reference_id,
            model=cfg.model,
            speed=cfg.speed,
            timeout=deps.tts_timeout,
        )

    def describe(self, cfg: FishAudioTtsConfig) -> str:
        # The default describe() would name only the engine generation (`model`);
        # the actual voice is `reference_id`, so include it when set. An empty
        # reference_id means the provider-default voice -> engine generation alone.
        if cfg.reference_id:
            return f"{self.id}/{cfg.model}/{cfg.reference_id}"
        return f"{self.id}/{cfg.model}"

    def options(self, field: str, cfg: FishAudioTtsConfig, deps: Deps, query: str = ""):
        if field == "model":
            # The known TTS model generations; the field stays freeform for
            # future ids.
            return ["s1", "s2-pro"]
        if field == "reference_id":
            if not cfg.api_key:
                return []  # the catalog requires auth; don't even try
            # `options` stays sync; the voice catalog is network-backed, so
            # return a coroutine — the caller awaits it (see Provider.options).
            if query:
                # Non-empty query -> server-side catalog search by title.
                return self._search_voices(cfg, deps, query)
            return self._fetch_voices(cfg, deps)
        return None

    async def _search_voices(self, cfg: FishAudioTtsConfig, deps: Deps, query: str):
        """Server-side catalog search by title, as [{"value", "label"}, ...].
        Results are NOT cached: each search is user-triggered and every query
        differs, so caching would only grow memory for no hit rate."""
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        resp = await deps.http_cloud.get(
            FISH_MODELS_URL, headers=headers, params={"title": query, "page_size": 30}
        )
        resp.raise_for_status()
        return _to_options(resp.json().get("items") or [])

    async def _fetch_voices(self, cfg: FishAudioTtsConfig, deps: Deps):
        """Fetch the voice catalog as [{"value", "label"}, ...]: the account's
        own voice models first, then the popular public ones, deduped by id.
        TTL-cached per api_key."""
        now = time.monotonic()
        cached = _voices_cache.get(cfg.api_key)
        if cached is not None and now - cached["at"] < _VOICES_CACHE_TTL:
            return cached["data"]
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
        own_resp = await deps.http_cloud.get(
            FISH_MODELS_URL, headers=headers, params={"self": "true", "page_size": 100}
        )
        own_resp.raise_for_status()
        popular_resp = await deps.http_cloud.get(
            FISH_MODELS_URL, headers=headers, params={"sort_by": "task_count", "page_size": 50}
        )
        popular_resp.raise_for_status()
        options = _to_options(
            (own_resp.json().get("items") or []) + (popular_resp.json().get("items") or [])
        )
        # Cache only after both requests succeeded — failures are never cached.
        _voices_cache[cfg.api_key] = {"at": now, "data": options}
        return options
