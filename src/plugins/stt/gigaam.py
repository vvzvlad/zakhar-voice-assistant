"""GigaAM offline STT brick: config schema and in-process onnx-asr backend."""

import asyncio
import os

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import LOCAL_MODEL_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend


class GigaamSttBackend(SttBackend):
    """In-process GigaAM backend: the ONNX export of Sber's Russian ASR model, run on
    CPU by onnx-asr.

    The model is loaded once and shared; each transcribe() decodes the whole
    utterance in a worker thread (onnxruntime inference is blocking). Batch only:
    no live streaming session and no closed-vocabulary decode.
    """

    def __init__(self, model_path: str, *, model_name: str = "gigaam-v3-rnnt",
                 quantization: str = "int8", model=None):
        # The model is injectable for testing. When None (production path), onnx-asr
        # is imported here so it is only required when GigaAM is actually in use.
        if model is None:
            # onnx-asr fetches a model from Hugging Face when the path is missing;
            # fail fast instead, the models are mounted on the volume.
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"GigaAM model directory not found: {model_path}")
            import onnx_asr

            # An empty quantization selects the full-precision model files.
            model = onnx_asr.load_model(model_name, model_path, quantization=quantization or None)
        self._model = model

    def _decode(self, pcm: bytes) -> str:
        # Speaker PCM is 16 kHz mono signed 16-bit little-endian; onnx-asr takes float32 in [-1, 1).
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return self._model.recognize(samples, sample_rate=16000).strip()

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        try:
            return await asyncio.to_thread(self._decode, pcm)
        except Exception as e:
            # SttBackend contract: a decode failure surfaces as StageError("stt", ...).
            logger.error(f"GigaAM STT decode failed: {str(e)}")
            raise StageError("stt", f"GigaAM STT decode failed: {e}") from e


class GigaamSttConfig(BaseModel):
    model_path: str = Field("models/gigaam-v3-onnx", title="Model directory",
                            json_schema_extra=LOCAL_MODEL_FIELD_EXTRA)
    model: str = Field("gigaam-v3-rnnt", title="Model")
    quantization: str = Field("int8", title="Quantization")


def _list_model_dirs(base_dir: str) -> list[dict]:
    """List the immediate non-hidden subdirectories of base_dir that hold an onnx-asr
    model (a config.json, which Vosk model dirs lack) as
    [{"value": <dir path>, "label": <dir name>}, ...] sorted by label
    (case-insensitive). Any filesystem error yields an empty list."""
    try:
        names = os.listdir(base_dir)
    except OSError:
        return []
    out = [
        {"value": os.path.join(base_dir, name), "label": name}
        for name in names
        if not name.startswith(".")
        and os.path.isfile(os.path.join(base_dir, name, "config.json"))
    ]
    out.sort(key=lambda o: o["label"].lower())
    return out


@register
class GigaamSttProvider(Provider):
    category = "stt"
    id = "gigaam"
    label = "GigaAM (offline)"
    ConfigModel = GigaamSttConfig
    uses_http_cloud = False

    def create(self, cfg: GigaamSttConfig, deps: Deps):
        return GigaamSttBackend(cfg.model_path, model_name=cfg.model, quantization=cfg.quantization)

    def options(self, field: str, cfg: GigaamSttConfig, deps: Deps, query: str = ""):
        # Local-disk scan of the directories next to the configured model directory
        # (defaults to models/), the same way the Vosk brick scans.
        if field == "model_path":
            return _list_model_dirs(os.path.dirname(cfg.model_path.rstrip("/")) or "models")
        return None
