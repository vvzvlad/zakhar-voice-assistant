"""Silero offline TTS brick: config schema and in-process backend.

Offline Silero TTS run locally via a torch.package `.pt` model (the v4_ru
multi-speaker Russian model). PyTorch is an OPTIONAL, lazily-imported dependency:
it is deliberately NOT in requirements.txt or the Docker image, so importing this
module (and `src.plugins`) must never import torch. torch is imported lazily inside
the backend constructor, only when a real model is loaded; numpy and loguru are the
only heavy-ish imports at module top-level.

Synthesis returns the engine's NATIVE format — WAV mono 16-bit at the configured
sample rate (audio/wav); adapting it to what the consumer can play is the delivery
boundary's job (see audio_codec.to_playable), not synthesis'. The canonical
LLM->TTS text arrives in "+vowel" stress notation (e.g. "прив+ет"), which is
EXACTLY Silero's own stress notation (the "+" precedes the stressed vowel), so the
markup is KEPT (like the Yandex backend) — only unit expansion and stray-'+' cleanup
are needed; no espeak combining-acute conversion and no phonetic hacks (Silero
pronounces "что"/"конечно" correctly).
"""

import asyncio
import io
import os
import wave

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import LOCAL_MODEL_FIELD_EXTRA, Deps, Provider, register
# The canonical LLM->TTS text is the model's own notation: plain text with "+"
# before the stressed vowel (e.g. "прив+ет") — Silero's native stress markup, so
# only unit expansion and stray-'+' cleanup are needed (KEEP the "+vowel" pairs).
from src.plugins.tts._ru_text import expand_units, sanitize_plus_stress
from src.tts import TtsBackend, split_sentences

# Speakers shipped in the v4_ru multi-speaker model.
V4_RU_SPEAKERS = ["aidar", "baya", "eugene", "kseniya", "xenia", "random"]
# Sample rates the v4_ru model can render at.
SILERO_SAMPLE_RATES = [8000, 24000, 48000]


class SileroTtsBackend(TtsBackend):
    """In-process Silero backend (neural TTS via torch.package). Returns WAV.

    The model is loaded once and shared. Synthesis is blocking (PyTorch on CPU),
    so it runs in a worker thread. Returns the engine's NATIVE format — WAV mono
    16-bit at the configured sample rate (audio/wav); adapting it to what the
    consumer can play is the delivery boundary's job (see audio_codec.to_playable),
    not synthesis'.
    """

    def __init__(self, model_path: str | None = None, *, speaker: str = "xenia",
                 sample_rate: int = 48000, put_accent: bool = True, put_yo: bool = True,
                 sentence_silence: float = 0.4, model=None):
        # The loaded model is injectable so _synth can be exercised in CI without
        # torch / the heavy model load. Production still passes model_path and gets
        # the real torch.package load path; model is only used by tests/from_model.
        if model is None:
            try:
                import torch  # lazy: torch is an optional dep, only needed when Silero TTS is selected
            except ImportError as e:
                raise RuntimeError(
                    "Silero TTS requires PyTorch. Install it (e.g. `pip install torch`) "
                    "to use the Silero TTS provider."
                ) from e

            model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            model.to(torch.device("cpu"))
            logger.info(f"Silero TTS model loaded: {model_path}")
        self._model = model
        self._speaker = speaker
        self._sample_rate = int(sample_rate)
        self._put_accent = bool(put_accent)
        self._put_yo = bool(put_yo)
        self._sentence_silence = max(0.0, float(sentence_silence))

    @classmethod
    def from_model(cls, model, *, speaker: str = "xenia", sample_rate: int = 48000,
                   put_accent: bool = True, put_yo: bool = True,
                   sentence_silence: float = 0.4) -> "SileroTtsBackend":
        """Build a backend around an already-loaded model (skips the torch load)."""
        return cls(model=model, speaker=speaker, sample_rate=sample_rate,
                   put_accent=put_accent, put_yo=put_yo, sentence_silence=sentence_silence)

    def _synth(self, text: str) -> bytes:
        # Adapt the canonical "+vowel" text to the engine: expand units, then KEEP
        # the "+vowel" stress pairs (Silero's native notation) and drop any stray '+'.
        text = sanitize_plus_stress(expand_units(text))
        sentences = split_sentences(text)
        pcm = bytearray()
        for sentence in sentences:
            try:
                audio = self._model.apply_tts(
                    text=sentence, speaker=self._speaker, sample_rate=self._sample_rate,
                    put_accent=self._put_accent, put_yo=self._put_yo,
                )
            except Exception:
                # Silero produced no audio for this fragment (e.g. unsupported
                # symbols only); skip it.
                continue
            # apply_tts returns a 1-D float32 torch tensor in [-1.0, 1.0]; convert
            # to int16 little-endian PCM bytes.
            arr = audio.numpy() if hasattr(audio, "numpy") else audio
            samples = np.clip(np.asarray(arr, dtype=np.float32), -1.0, 1.0)
            frames = (samples * 32767.0).astype("<i2").tobytes()
            if not frames:
                continue
            if pcm and self._sentence_silence > 0:
                # whole number of frames of silence (mono 16-bit -> 2 bytes/frame),
                # so samples stay aligned
                pcm += b"\x00" * (int(self._sample_rate * self._sentence_silence) * 2)
            pcm += frames
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setframerate(self._sample_rate)
            wf.setnchannels(1)
            wf.setsampwidth(2)
            # Nothing pronounceable -> pcm is empty -> a valid (empty/silent) WAV.
            wf.writeframes(bytes(pcm))
        return out.getvalue()

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        wav = await asyncio.to_thread(self._synth, text)
        return ("audio/wav", wav)


class SileroTtsConfig(BaseModel):
    model_path: str = Field("models/silero_tts_v4_ru.pt", json_schema_extra=LOCAL_MODEL_FIELD_EXTRA)
    speaker: str = Field("xenia", json_schema_extra={"widget": "select", "options": "dynamic", "freeform": True})
    sample_rate: int = Field(48000, json_schema_extra={"widget": "select", "options": "dynamic", "freeform": False})
    # Auto-place stress on words without a manual "+" mark; Silero honors existing "+".
    put_accent: bool = True
    # Restore е->ё where appropriate.
    put_yo: bool = True
    sentence_silence: float = 0.4


def _list_silero_models(base_dir: str) -> list[dict]:
    """Scan base_dir for Silero models: *.pt files (torch.package). Returns
    [{"value": <pt path>, "label": <name without .pt>}, ...] sorted by label
    (case-insensitive). Any filesystem error yields an empty list."""
    try:
        names = os.listdir(base_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith(".") or not name.endswith(".pt"):
            continue
        pt_path = os.path.join(base_dir, name)
        if not os.path.isfile(pt_path):
            continue
        out.append({"value": pt_path, "label": name[: -len(".pt")]})
    out.sort(key=lambda o: o["label"].lower())
    return out


@register
class SileroTtsProvider(Provider):
    category = "tts"
    id = "silero"
    label = "Silero (offline)"
    ConfigModel = SileroTtsConfig

    def create(self, cfg: SileroTtsConfig, deps: Deps):
        return SileroTtsBackend(
            cfg.model_path,
            speaker=cfg.speaker,
            sample_rate=cfg.sample_rate,
            put_accent=cfg.put_accent,
            put_yo=cfg.put_yo,
            sentence_silence=cfg.sentence_silence,
        )

    def describe(self, cfg: SileroTtsConfig) -> str:
        # Include the speaker (like fishaudio includes its voice): the model file
        # alone does not identify which of the multi-speaker voices is used.
        return f"{self.id}/{os.path.basename(cfg.model_path)}/{cfg.speaker}"

    def options(self, field: str, cfg: SileroTtsConfig, deps: Deps, query: str = ""):
        # Local-disk scan of installed Silero models next to the configured path
        # (defaults to models/). Synchronous: return a plain list. rstrip the
        # trailing slash for symmetry with the Vosk/Piper scan (no-op for a real
        # model file path, which never ends in a slash).
        if field == "model_path":
            return _list_silero_models(os.path.dirname(cfg.model_path.rstrip("/")) or "models")
        if field == "speaker":
            return list(V4_RU_SPEAKERS)
        if field == "sample_rate":
            return [{"value": r, "label": f"{r} Hz"} for r in SILERO_SAMPLE_RATES]
        return None
