"""Unit tests for the Silero VAD stage plugin (session, backend, provider, config).

These drive SileroVadSession directly against a FAKE onnxruntime session (no
model file needed): windowing across odd chunk sizes, the threshold hysteresis,
the LSTM state/context round-tripping and the end-pointing decisions against an
EndpointPolicy. The real-model integration tests at the bottom are skipped when
models/silero_vad.onnx is absent (CI) and exercise the verified model interface
(input [1, 576] = 64 context + 512 samples, state [2, 1, 128], sr int64).
"""

import os

import numpy as np
import pytest
from pydantic import ValidationError

import src.plugins  # noqa: F401  triggers @register on all providers
from src.plugins.base import get_provider
from src.plugins.vad.silero import (
    CONTEXT_SAMPLES,
    WINDOW_BYTES,
    WINDOW_MS,
    WINDOW_SAMPLES,
    SileroVadBackend,
    SileroVadConfig,
    SileroVadSession,
)
from src.vad import EndpointPolicy, VadBackend, VadSession

SILERO_MODEL = "models/silero_vad.onnx"
PIPER_VOICE = "models/ru_RU-ruslan-medium.onnx"


class FakeOrt:
    """onnxruntime.InferenceSession double: run(None, feeds) returns the model's
    outputs in declaration order — [output [1,1], stateN [2,1,128]] — exactly what
    the session code consumes. Scripts a sequence of probabilities (repeating the
    last one once the script runs out) and records every feeds dict (arrays
    copied, so later in-place reuse can't alias) plus every state it returned,
    so state round-tripping can be asserted call by call."""

    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0
        self.feeds = []
        self.states_returned = []

    def run(self, output_names, feeds):
        assert output_names is None
        self.feeds.append({k: np.array(v) for k, v in feeds.items()})
        if self._i < len(self._probs):
            p = self._probs[self._i]
            self._i += 1
        else:
            p = self._probs[-1] if self._probs else 0.0
        # A distinct, recognizable state per call (call index broadcast over the
        # state shape) lets tests check that call N+1 was fed call N's state.
        state = np.full((2, 1, 128), float(self._i), dtype=np.float32)
        self.states_returned.append(state)
        return [np.array([[p]], dtype=np.float32), state]


# 32 ms windows: 3 silent windows end (96 ms), 2 speech windows arm (64 ms),
# 20 windows hard cap (640 ms), 10 windows with no speech finalize (320 ms).
POLICY = EndpointPolicy(
    silence_ms=96,
    min_speech_ms=64,
    max_utterance_ms=640,
    no_speech_timeout_ms=320,
)

WINDOW = b"\x00" * WINDOW_BYTES  # one 32 ms window of silence-shaped PCM


def make_session(probs, *, threshold=0.5, neg_threshold=0.35, policy=POLICY):
    fake = FakeOrt(probs)
    session = SileroVadSession(fake, policy, threshold, neg_threshold)
    return session, fake


# --- framing -----------------------------------------------------------------

def test_feed_does_its_own_framing_across_odd_chunk_sizes():
    # Chunks are NOT 1024-aligned: the session carries the remainder across feeds
    # and only whole 1024-byte windows reach the model. 700 + 700 bytes = 1 whole
    # window (1024 bytes) + a 376-byte remainder.
    session, fake = make_session([0.0])
    assert session.feed(b"\x01" * 700) is None   # sub-window: no inference yet
    assert fake.feeds == []
    assert session.feed(b"\x01" * 700) is None   # 1400 bytes -> one window consumed
    assert len(fake.feeds) == 1
    assert len(session._rem) == 1400 - WINDOW_BYTES
    assert session.debug_state()["elapsed_ms"] == WINDOW_MS
    # Every model call saw exactly one context-prefixed window: [1, 576] float32.
    x = fake.feeds[0]["input"]
    assert x.shape == (1, CONTEXT_SAMPLES + WINDOW_SAMPLES)
    assert x.dtype == np.float32


def test_feed_sub_window_chunks_accumulate_to_windows():
    # Chunks smaller than a window produce no inference until 1024 bytes accumulate.
    session, fake = make_session([0.0])
    for _ in range(10):  # 10 * 100 = 1000 bytes < 1024
        assert session.feed(b"\x00" * 100) is None
    assert fake.feeds == []
    session.feed(b"\x00" * 100)  # 1100 bytes -> one window consumed
    assert len(fake.feeds) == 1


def test_input_normalization_and_context_carry():
    # int16 samples are normalized by /32768; the first window gets a zero
    # context prefix, the second window's prefix is the previous window's tail.
    session, fake = make_session([0.0])
    first = np.full(WINDOW_SAMPLES, 16384, dtype="<i2").tobytes()
    second = np.full(WINDOW_SAMPLES, -8192, dtype="<i2").tobytes()
    session.feed(first)
    session.feed(second)
    x1, x2 = fake.feeds[0]["input"][0], fake.feeds[1]["input"][0]
    assert np.all(x1[:CONTEXT_SAMPLES] == 0.0)               # zero context at start
    assert np.allclose(x1[CONTEXT_SAMPLES:], 16384 / 32768)  # /32768 normalization
    assert np.allclose(x2[:CONTEXT_SAMPLES], 16384 / 32768)  # tail of window 1
    assert np.allclose(x2[CONTEXT_SAMPLES:], -8192 / 32768)


def test_sr_input_is_int64_16000():
    session, fake = make_session([0.0])
    session.feed(WINDOW)
    sr = fake.feeds[0]["sr"]
    assert sr.dtype == np.int64
    assert int(sr) == 16000


# --- end-pointing decisions ----------------------------------------------------

def test_endpoint_after_trailing_silence():
    # 2 speech windows (>= 64 ms min) then 3 low-probability windows (>= 96 ms
    # of trailing silence) -> endpoint.
    session, _ = make_session([0.9, 0.9, 0.1, 0.1, 0.1])
    assert session.feed(WINDOW * 5) == "endpoint"
    state = session.debug_state()
    assert state["speech_detected"] is True
    assert state["silence_ms"] >= POLICY.silence_ms


def test_threshold_boundary_is_inclusive():
    # prob == threshold counts as speech (>=), just below does not.
    session, _ = make_session([0.5, 0.5])
    session.feed(WINDOW * 2)
    assert session.debug_state()["speech_ms"] == 2 * WINDOW_MS
    session, _ = make_session([0.499])
    session.feed(WINDOW)
    assert session.debug_state()["speech_ms"] == 0


def test_no_speech_timeout_with_all_low_probabilities():
    # Probabilities never reach the threshold: the no-speech timeout finalizes
    # after 10 windows (320 ms).
    session, _ = make_session([0.1])
    assert session.feed(WINDOW * 9) is None
    assert session.feed(WINDOW) == "no_speech"


def test_maxlen_when_speech_never_ends():
    # Always above threshold: no trailing silence, only the hard cap finalizes.
    session, _ = make_session([0.9])
    assert session.feed(WINDOW * 19) is None
    assert session.feed(WINDOW) == "maxlen"  # 20 windows = 640 ms = max_utterance_ms


# --- hysteresis -----------------------------------------------------------------

def test_hysteresis_keeps_speech_alive_between_thresholds():
    # threshold 0.5 / neg 0.35 over [0.6, 0.4, 0.4, 0.2]: windows 2-3 are still
    # speech (0.4 >= 0.35 while speaking), window 4 (0.2 < 0.35) ends speech.
    session, _ = make_session([0.6, 0.4, 0.4, 0.2])
    session.feed(WINDOW * 4)
    state = session.debug_state()
    assert state["speech_ms"] == 3 * WINDOW_MS
    assert state["silence_ms"] == 1 * WINDOW_MS
    assert state["speech_detected"] is True  # 3 windows >= min_speech_ms=64


def test_no_hysteresis_when_thresholds_equal():
    # Same probabilities with neg == threshold: 0.4 windows are NOT speech, so
    # only the first window counts (contrast with the hysteresis test above).
    session, _ = make_session([0.6, 0.4, 0.4, 0.2], neg_threshold=0.5)
    session.feed(WINDOW * 4)
    assert session.debug_state()["speech_ms"] == 1 * WINDOW_MS


def test_hysteresis_rearms_at_full_threshold_after_speech_ends():
    # Once speech has ended (prob < neg), an in-between probability (neg <= p <
    # threshold) must NOT restart speech: the start threshold applies again.
    session, _ = make_session([0.6, 0.2, 0.4])
    session.feed(WINDOW * 3)
    assert session.debug_state()["speech_ms"] == 1 * WINDOW_MS


# --- LSTM state round-tripping ---------------------------------------------------

def test_state_fed_to_next_call_is_state_returned_by_previous():
    session, fake = make_session([0.5, 0.5, 0.5])
    session.feed(WINDOW * 3)
    assert len(fake.feeds) == 3
    # Call 1 starts from zeros; call N+1 is fed exactly what call N returned.
    assert np.all(fake.feeds[0]["state"] == 0.0)
    assert np.array_equal(fake.feeds[1]["state"], fake.states_returned[0])
    assert np.array_equal(fake.feeds[2]["state"], fake.states_returned[1])
    assert fake.feeds[0]["state"].shape == (2, 1, 128)
    assert fake.feeds[0]["state"].dtype == np.float32


def test_new_session_starts_from_zero_state():
    fake = FakeOrt([0.5])
    backend = SileroVadBackend("unused", 0.5, 0.35, ort_session=fake)
    s1 = backend.open(POLICY)
    s1.feed(WINDOW * 2)
    assert not np.all(fake.feeds[-1]["state"] == 0.0)  # state advanced within s1
    s2 = backend.open(POLICY)
    s2.feed(WINDOW)
    # The new utterance's first call is back to a zero LSTM state and zero context.
    assert np.all(fake.feeds[-1]["state"] == 0.0)
    assert np.all(fake.feeds[-1]["input"][0][:CONTEXT_SAMPLES] == 0.0)


# --- backend / provider ---------------------------------------------------------

def test_backend_opens_fresh_independent_sessions():
    backend = SileroVadBackend("unused", 0.5, 0.35, ort_session=FakeOrt([0.9]))
    s1 = backend.open(POLICY)
    s2 = backend.open(POLICY)
    assert isinstance(s1, VadSession) and isinstance(s2, VadSession)
    assert s1 is not s2
    # Sessions are independent: state accumulated in one never leaks to the other.
    s1.feed(WINDOW * 3)
    assert s2.debug_state()["elapsed_ms"] == 0
    assert s1.debug_state()["elapsed_ms"] == 3 * WINDOW_MS


def test_backend_fails_fast_on_missing_model_file():
    # Production path (no injected ort_session): the InferenceSession load fails
    # in the constructor, like Vosk's Model(model_path) — not on first feed().
    with pytest.raises(Exception):
        SileroVadBackend("models/definitely-not-there.onnx", 0.5, 0.35)


def test_provider_registered_with_config_model():
    prov = get_provider("vad", "silero")
    assert prov.label == "Silero VAD (neural)"
    assert prov.ConfigModel is SileroVadConfig
    assert prov.category == "vad" and prov.id == "silero"


def test_debug_state_shape_and_prob_rounding():
    session, _ = make_session([0.637])
    session.feed(WINDOW)
    state = session.debug_state()
    assert state == {
        "speech_ms": 32, "silence_ms": 0, "elapsed_ms": 32,
        "speech_detected": False, "prob": 0.64,
    }


# --- config ----------------------------------------------------------------------

def test_config_defaults():
    cfg = SileroVadConfig()
    assert cfg.model_path == "models/silero_vad.onnx"
    assert cfg.threshold == 0.5
    assert cfg.neg_threshold == 0.35


def test_config_clamps_neg_threshold_to_threshold():
    # Misconfigured pair (sliders moved independently): neg is clamped, not rejected.
    cfg = SileroVadConfig(threshold=0.4, neg_threshold=0.8)
    assert cfg.neg_threshold == 0.4
    # A valid pair is left alone.
    cfg = SileroVadConfig(threshold=0.7, neg_threshold=0.3)
    assert cfg.neg_threshold == 0.3


def test_config_bounds_reject_out_of_range():
    with pytest.raises(ValidationError):
        SileroVadConfig(threshold=0.96)
    with pytest.raises(ValidationError):
        SileroVadConfig(neg_threshold=0.04)


def test_config_schema_has_slider_widgets():
    props = SileroVadConfig.model_json_schema()["properties"]
    assert props["threshold"]["widget"] == "slider"
    assert props["neg_threshold"]["widget"] == "slider"
    assert props["threshold"]["title"] == "Speech threshold"
    assert props["neg_threshold"]["title"] == "Speech-end threshold"


# --- integration with the real ONNX model (skipped when the file is absent) ------

@pytest.mark.skipif(not os.path.exists(SILERO_MODEL), reason="silero model not downloaded")
def test_real_model_rejects_tone_and_hits_no_speech_timeout():
    # The real model via the real create() path. A loud pure tone is exactly what
    # Silero (unlike WebRTC VAD) classifies as NON-speech, so a tone+silence run
    # must finalize as "no_speech" — this pins the verified model interface
    # (input/state/sr names, the 576-sample context window) end to end.
    prov = get_provider("vad", "silero")
    backend = prov.create(SileroVadConfig(), deps=None)
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
    not (os.path.exists(SILERO_MODEL) and os.path.exists(PIPER_VOICE)),
    reason="silero model and/or piper voice not downloaded",
)
def test_real_model_endpoints_on_synthesized_speech():
    # Real speech end-pointing: synthesize a short Russian utterance with the
    # in-repo Piper voice (22050 Hz), linearly resample to 16 kHz, append a
    # second of silence and assert the Silero session fires "endpoint".
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

    backend = SileroVadBackend(SILERO_MODEL, 0.5, 0.35)
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
