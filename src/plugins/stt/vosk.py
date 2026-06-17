"""Vosk offline STT brick: config schema and in-process backend."""

import asyncio
import json
import os

from loguru import logger
from pydantic import BaseModel, Field

from src.logging_setup import capture_native_stderr
from src.plugins.base import LOCAL_MODEL_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend


class VoskSttBackend(SttBackend):
    """In-process Vosk backend (small offline model, 16 kHz mono CPU decode).

    The model is loaded once and shared; KaldiRecognizer instances are spawned
    per call (the model is thread-safe for spawning recognizers). Decoding is
    blocking C code, so it runs in a worker thread.
    """

    def __init__(self, model_path: str, *, model=None):
        # The model is injectable for testing. When None (production path), it is
        # loaded lazily so the heavy dependency/model are only required when the
        # Vosk backend is actually selected at runtime (never in tests/CI).
        if model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            # Capture the native WARN/ERR Kaldi can emit during the model load
            # into loguru (minimal window — only the load itself).
            with capture_native_stderr("vosk-stt"):
                model = Model(model_path)  # fail fast if the dir is missing
        self._model = model

    def _make_recognizer(self):
        """Build a KaldiRecognizer for the shared model (overridable in tests)."""
        from vosk import KaldiRecognizer

        # NO fd-2 capture here: this runs per transcribe in a worker thread, and
        # redirecting process-global fd 2 around every decode would swallow loguru
        # lines emitted by other threads. The model load in __init__ is the rare
        # window that's wrapped instead. STT has no grammar, so a plain full-vocab
        # recognizer is quiet at construction (nothing actionable to surface here).
        return KaldiRecognizer(self._model, 16000)

    def _decode(self, pcm: bytes) -> str:
        rec = self._make_recognizer()
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        # Speaker PCM is already 16 kHz mono 16-bit — no resampling needed.
        try:
            return await asyncio.to_thread(self._decode, pcm)
        except StageError:
            raise
        except Exception as e:
            # Honor the SttBackend contract: any decode failure (native Vosk error,
            # malformed recognizer JSON, ...) surfaces as StageError("stt", ...) so
            # the pipeline records the run as an STT error instead of dying raw.
            logger.error(f"Vosk STT decode failed: {str(e)}")
            raise StageError("stt", f"Vosk STT decode failed: {e}") from e


class VoskSttConfig(BaseModel):
    model_path: str = Field("models/vosk-model-small-ru-0.22", json_schema_extra=LOCAL_MODEL_FIELD_EXTRA)


def _list_vosk_models(base_dir: str) -> list[dict]:
    """Scan base_dir for Vosk model directories: immediate subdirs that contain
    both an `am` and a `conf` subdirectory (the structural markers of a standard
    Vosk model). Returns [{"value": <dir path>, "label": <dir name>}, ...] sorted
    by label (case-insensitive). Any filesystem error yields an empty list."""
    try:
        names = os.listdir(base_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(base_dir, name)
        if (
            os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "am"))
            and os.path.isdir(os.path.join(path, "conf"))
        ):
            out.append({"value": path, "label": name})
    out.sort(key=lambda o: o["label"].lower())
    return out


@register
class VoskSttProvider(Provider):
    category = "stt"
    id = "vosk"
    label = "Vosk (offline)"
    ConfigModel = VoskSttConfig

    def create(self, cfg: VoskSttConfig, deps: Deps):
        return VoskSttBackend(cfg.model_path)

    def options(self, field: str, cfg: VoskSttConfig, deps: Deps, query: str = ""):
        # Local-disk scan of installed Vosk models next to the configured path
        # (defaults to models/). Synchronous: return a plain list. The model path
        # is a directory; strip any trailing slash before dirname() so a value like
        # "models/foo/" scans "models" (its siblings) rather than descending into it.
        if field == "model_path":
            return _list_vosk_models(os.path.dirname(cfg.model_path.rstrip("/")) or "models")
        return None
