"""Vosk offline STT brick: config schema and in-process backend."""

import asyncio
import json

from loguru import logger
from pydantic import BaseModel

from src.logging_setup import capture_native_stderr
from src.plugins.base import Deps, Provider, register
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
    model_path: str = "models/vosk-model-small-ru-0.22"


@register
class VoskSttProvider(Provider):
    category = "stt"
    id = "vosk"
    label = "Vosk (offline)"
    ConfigModel = VoskSttConfig

    def create(self, cfg: VoskSttConfig, deps: Deps):
        return VoskSttBackend(cfg.model_path)
