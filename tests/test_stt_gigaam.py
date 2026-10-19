"""Tests for the GigaAM offline STT provider, with a fake onnx-asr model (no real model)."""

import sys
import types

import numpy as np
import pytest

import src.plugins  # noqa: F401  register all providers
from src.plugins.base import get_provider
from src.plugins.stt.gigaam import GigaamSttBackend, GigaamSttConfig, GigaamSttProvider
from src.stage_errors import StageError


class _FakeModel:
    """Stands in for the onnx-asr model: records each recognize() call."""

    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def recognize(self, samples, *, sample_rate):
        self.calls.append((samples, sample_rate))
        return self.text


async def test_transcribe_feeds_float32_at_16k_and_strips_text():
    model = _FakeModel("  захар выключи ночник \n")
    backend = GigaamSttBackend("unused/path", model=model)
    pcm = np.array([0, 16384, -32768, 32767], dtype=np.int16).tobytes()

    assert await backend.transcribe(pcm) == "захар выключи ночник"

    [(samples, rate)] = model.calls
    assert rate == 16000
    assert samples.dtype == np.float32
    assert samples.tolist() == [0.0, 0.5, -1.0, 32767 / 32768]


async def test_empty_audio_skips_the_model_and_decode_failure_is_an_stt_error():
    model = _FakeModel("x")
    backend = GigaamSttBackend("unused/path", model=model)
    assert await backend.transcribe(b"") == ""
    assert model.calls == []

    def broken(samples, *, sample_rate):
        raise RuntimeError("onnx boom")

    model.recognize = broken
    with pytest.raises(StageError) as err:
        await backend.transcribe(b"\x00\x00")
    assert err.value.stage == "stt"


def test_registered_offline_batch_provider_passes_config_to_onnx_asr(monkeypatch, tmp_path):
    loads = []
    fake = types.ModuleType("onnx_asr")
    fake.load_model = lambda name, path, *, quantization: loads.append((name, path, quantization))
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)

    prov = get_provider("stt", "gigaam")
    assert isinstance(prov, GigaamSttProvider)
    assert (prov.label, prov.uses_http_cloud, prov.supports_vocabulary) == ("GigaAM (offline)", False, False)
    path = str(tmp_path)
    prov.create(GigaamSttConfig(model_path=path), None)
    prov.create(GigaamSttConfig(model_path=path, quantization=""), None)  # empty = full precision
    assert loads == [("gigaam-v3-rnnt", path, "int8"), ("gigaam-v3-rnnt", path, None)]

    # A missing directory fails at build time instead of reaching onnx-asr's download.
    with pytest.raises(FileNotFoundError):
        prov.create(GigaamSttConfig(model_path=str(tmp_path / "absent")), None)
    assert len(loads) == 2


def test_model_path_options_list_sibling_model_dirs(tmp_path):
    for name in ("vosk-small", "Gigaam-v3", "gigaam-v2", ".cache"):
        (tmp_path / name).mkdir()
    for name in ("Gigaam-v3", "gigaam-v2", ".cache"):
        (tmp_path / name / "config.json").write_text("{}")
    (tmp_path / "voice.onnx").write_text("x")  # a file, not a model directory
    cfg = GigaamSttConfig(model_path=str(tmp_path / "Gigaam-v3") + "/")

    # vosk-small has no config.json: not an onnx-asr model directory.
    assert GigaamSttProvider().options("model_path", cfg, None) == [
        {"value": str(tmp_path / "gigaam-v2"), "label": "gigaam-v2"},
        {"value": str(tmp_path / "Gigaam-v3"), "label": "Gigaam-v3"},
    ]
    assert GigaamSttProvider().options("model", cfg, None) is None
