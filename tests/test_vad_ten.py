"""Unit tests for the TEN VAD stage plugin (session, backend, provider, config).

These drive TenVadSession directly against a FAKE engine injected through
engine_factory (no native library needed): hop framing across odd chunk sizes,
the flag-based speech decision, the end-pointing decisions against an
EndpointPolicy and the one-engine-per-utterance lifecycle. The real-library
integration tests at the bottom are skipped when the `ten-vad` package isn't
installed and pin the verified engine interface (TenVad(hop_size, threshold),
process(int16[256]) -> (prob, flag)).
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
from pydantic import ValidationError

import src.plugins  # noqa: F401  triggers @register on all providers
from src.plugins.base import get_provider
from src.plugins.vad.ten import (
    HOP_BYTES,
    HOP_MS,
    HOP_SAMPLES,
    TenVadBackend,
    TenVadConfig,
    TenVadSession,
)
from src.vad import EndpointPolicy, VadBackend, VadSession

HAVE_TEN_VAD = importlib.util.find_spec("ten_vad") is not None
PIPER_VOICE = "models/ru_RU-ruslan-medium.onnx"


class FakeTenVad:
    """TenVad double: process(samples) returns scripted (prob, flag) tuples in
    order, repeating the last one once the script runs out. Records every
    samples array passed in (copied), so framing tests can inspect exactly what
    the engine saw."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.frames = []

    def process(self, samples):
        self.frames.append(np.array(samples))
        if self._i < len(self._script):
            val = self._script[self._i]
            self._i += 1
        else:
            val = self._script[-1] if self._script else (0.0, 0)
        return val


# 16 ms hops: 3 silent hops end (48 ms), 2 speech hops arm (32 ms),
# 20 hops hard cap (320 ms), 10 hops with no speech finalize (160 ms).
POLICY = EndpointPolicy(
    silence_ms=48,
    min_speech_ms=32,
    max_utterance_ms=320,
    no_speech_timeout_ms=160,
)

HOP = b"\x00" * HOP_BYTES  # one 16 ms hop of silence-shaped PCM


def make_session(script, *, policy=POLICY):
    fake = FakeTenVad(script)
    session = TenVadSession(fake, policy)
    return session, fake


# --- framing -----------------------------------------------------------------

def test_feed_does_its_own_framing_across_odd_chunk_sizes():
    # Chunks are NOT 512-aligned: the session carries the remainder across feeds
    # and only whole 512-byte hops reach the engine. 300 + 300 bytes = 1 whole
    # hop (512 bytes) + an 88-byte remainder.
    session, fake = make_session([(0.0, 0)])
    assert session.feed(b"\x01" * 300) is None   # sub-hop: no process() yet
    assert fake.frames == []
    assert session.feed(b"\x01" * 300) is None   # 600 bytes -> one hop consumed
    assert len(fake.frames) == 1
    assert len(session._rem) == 600 - HOP_BYTES
    assert session.debug_state()["elapsed_ms"] == HOP_MS
    # Every process() call saw exactly one int16 hop of HOP_SAMPLES samples.
    samples = fake.frames[0]
    assert samples.shape == (HOP_SAMPLES,)
    assert samples.dtype == np.int16


def test_feed_sub_hop_chunks_accumulate_to_hops():
    # Chunks smaller than a hop produce no process() until 512 bytes accumulate.
    session, fake = make_session([(0.0, 0)])
    for _ in range(5):  # 5 * 100 = 500 bytes < 512
        assert session.feed(b"\x00" * 100) is None
    assert fake.frames == []
    session.feed(b"\x00" * 100)  # 600 bytes -> one hop consumed
    assert len(fake.frames) == 1


def test_multi_hop_chunk_processes_every_whole_hop():
    # One big chunk of 3.5 hops: exactly 3 process() calls, half a hop carried.
    session, fake = make_session([(0.0, 0)])
    session.feed(b"\x00" * (3 * HOP_BYTES + HOP_BYTES // 2))
    assert len(fake.frames) == 3
    assert len(session._rem) == HOP_BYTES // 2


def test_int16_samples_reach_engine_unscaled():
    # TEN consumes raw int16 PCM (no /32768 normalization like Silero).
    session, fake = make_session([(0.0, 0)])
    pcm = np.full(HOP_SAMPLES, -12345, dtype="<i2").tobytes()
    session.feed(pcm)
    assert np.all(fake.frames[0] == -12345)


# --- the flag IS the decision (engine-applied threshold) -------------------------

def test_speech_decision_uses_engine_flag_not_probability():
    # The native engine applies the configured threshold internally and returns
    # the flag; the session must trust the flag, NOT re-compare the probability:
    # a high prob with flag 0 is non-speech, a low prob with flag 1 is speech.
    session, _ = make_session([(0.9, 0), (0.9, 0)])
    session.feed(HOP * 2)
    assert session.debug_state()["speech_ms"] == 0
    session, _ = make_session([(0.1, 1), (0.1, 1)])
    session.feed(HOP * 2)
    assert session.debug_state()["speech_ms"] == 2 * HOP_MS


# --- end-pointing decisions ----------------------------------------------------

def test_endpoint_after_trailing_silence():
    # 2 speech hops (>= 32 ms min) then 3 silent hops (>= 48 ms of trailing
    # silence) -> endpoint.
    session, _ = make_session([(0.9, 1), (0.9, 1), (0.1, 0), (0.1, 0), (0.1, 0)])
    assert session.feed(HOP * 5) == "endpoint"
    state = session.debug_state()
    assert state["speech_detected"] is True
    assert state["silence_ms"] >= POLICY.silence_ms


def test_no_speech_timeout_with_all_silent_hops():
    # Flags never go high: the no-speech timeout finalizes after 10 hops (160 ms).
    session, _ = make_session([(0.1, 0)])
    assert session.feed(HOP * 9) is None
    assert session.feed(HOP) == "no_speech"


def test_maxlen_when_speech_never_ends():
    # Always speech: no trailing silence, only the hard cap finalizes.
    session, _ = make_session([(0.9, 1)])
    assert session.feed(HOP * 19) is None
    assert session.feed(HOP) == "maxlen"  # 20 hops = 320 ms = max_utterance_ms


def test_min_speech_gate_blocks_endpoint_for_short_blip():
    # A single 16 ms speech blip (< min_speech_ms=32) never arms speech_detected,
    # so trailing silence can NOT end-point; only the no-speech timeout fires.
    session, _ = make_session([(0.9, 1), (0.1, 0)])
    assert session.feed(HOP * 9) is None
    assert session.debug_state()["speech_detected"] is False
    assert session.feed(HOP) == "no_speech"


def test_debug_state_shape_and_prob_rounding():
    session, _ = make_session([(0.637, 0)])
    session.feed(HOP)
    assert session.debug_state() == {
        "speech_ms": 0, "silence_ms": 0, "elapsed_ms": 16,
        "speech_detected": False, "prob": 0.64,
    }


# --- backend / provider ---------------------------------------------------------

def test_backend_builds_one_engine_per_open():
    # The TenVad handle carries internal state -> a FRESH engine per utterance:
    # two open() calls must invoke the factory twice with independent results.
    engines = []

    def factory():
        engines.append(FakeTenVad([(0.9, 1)]))
        return engines[-1]

    backend = TenVadBackend(0.5, engine_factory=factory)
    s1 = backend.open(POLICY)
    s2 = backend.open(POLICY)
    assert isinstance(s1, VadSession) and isinstance(s2, VadSession)
    assert len(engines) == 2
    assert s1._engine is not s2._engine
    # Sessions are independent: state accumulated in one never leaks to the other.
    s1.feed(HOP * 3)
    assert s2.debug_state()["elapsed_ms"] == 0
    assert s1.debug_state()["elapsed_ms"] == 3 * HOP_MS
    assert engines[1].frames == []


def test_backend_ctor_fails_fast_when_package_missing(monkeypatch):
    # Production path (no injected engine_factory): the `ten_vad` import happens
    # in the constructor — like Vosk/Silero fail-fast create() — so reconfig
    # surfaces the error and keeps the old backend. Simulate the missing package
    # by poisoning sys.modules (None makes `import ten_vad` raise ImportError).
    monkeypatch.setitem(sys.modules, "ten_vad", None)
    with pytest.raises(ImportError, match="pip install ten-vad"):
        TenVadBackend(0.5)


def test_provider_registered_with_config_model():
    prov = get_provider("vad", "ten")
    assert prov.label == "TEN VAD (neural)"
    assert prov.ConfigModel is TenVadConfig
    assert prov.category == "vad" and prov.id == "ten"


# --- config ----------------------------------------------------------------------

def test_config_defaults():
    assert TenVadConfig().threshold == 0.5


def test_config_bounds_reject_out_of_range():
    with pytest.raises(ValidationError):
        TenVadConfig(threshold=0.96)
    with pytest.raises(ValidationError):
        TenVadConfig(threshold=0.04)


def test_config_schema_has_slider_widget():
    props = TenVadConfig.model_json_schema()["properties"]
    assert props["threshold"]["widget"] == "slider"
    assert props["threshold"]["title"] == "Speech threshold"


# --- integration with the real ten-vad native library ----------------------------

@pytest.mark.skipif(not HAVE_TEN_VAD, reason="ten-vad package not installed")
def test_real_engine_rejects_tone_and_hits_no_speech_timeout():
    # The real engine via the real create() path. Empirical note: at the default
    # threshold 0.5 TEN flags the ONSET of a loud pure tone as speech for ~290 ms
    # (probability peaks ~0.71) — unlike Silero. At threshold 0.8 the tone never
    # crosses, so a tone+silence run must finalize as "no_speech"; this also pins
    # that the configured threshold really reaches the native engine.
    prov = get_provider("vad", "ten")
    backend = prov.create(TenVadConfig(threshold=0.8), deps=None)
    session = backend.open(EndpointPolicy(
        silence_ms=500, min_speech_ms=96,
        max_utterance_ms=10_000, no_speech_timeout_ms=1_000,
    ))
    t = np.arange(16000, dtype=np.float32) / 16000.0  # 1 s of 440 Hz tone
    tone = (8000 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes()
    reason = session.feed(tone)
    if reason is None:
        reason = session.feed(b"\x00" * 16000)  # silence past the 1 s timeout
    assert reason == "no_speech"
    assert session.debug_state()["speech_detected"] is False


@pytest.mark.skipif(
    not (HAVE_TEN_VAD and os.path.exists(PIPER_VOICE)),
    reason="ten-vad package and/or piper voice not available",
)
def test_real_engine_endpoints_on_synthesized_speech():
    # Real speech end-pointing: synthesize a short Russian utterance with the
    # in-repo Piper voice (22050 Hz), linearly resample to 16 kHz, append a
    # second of silence and assert the TEN session fires "endpoint".
    import io
    import wave

    from piper import PiperVoice

    voice = PiperVoice.load(PIPER_VOICE, PIPER_VOICE + ".json")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav("Привет, как дела?", wf)
    with wave.open(io.BytesIO(buf.getvalue()), "rb") as rf:
        src_rate = rf.getframerate()
        pcm = np.frombuffer(rf.readframes(rf.getnframes()), dtype="<i2")
    # Plain linear interpolation is plenty for a VAD decision (no scipy dep).
    idx = np.arange(0, len(pcm), src_rate / 16000.0)
    speech = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float32)).astype("<i2")

    backend = TenVadBackend(0.5)
    session = backend.open(EndpointPolicy(
        silence_ms=500, min_speech_ms=96,
        max_utterance_ms=30_000, no_speech_timeout_ms=5_000,
    ))
    reason = session.feed(speech.tobytes())
    if reason is None:
        reason = session.feed(b"\x00" * 32000)  # 1 s of trailing silence
    assert reason == "endpoint"
    state = session.debug_state()
    assert state["speech_detected"] is True
    assert state["speech_ms"] >= 96
