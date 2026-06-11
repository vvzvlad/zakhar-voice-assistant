"""Unit tests for the WebRTC VAD stage plugin (session, backend, provider, boost).

These drive WebRtcVadSession directly (much cleaner than via the pipeline):
framing across odd chunk sizes, the end-pointing decisions against an
EndpointPolicy, and the decision-only auto_gain boost.
"""

import os

import numpy as np
import pytest
from pydantic import ValidationError

import src.plugins  # noqa: F401  triggers @register on all providers
from src.plugins.base import get_provider, providers
from src.plugins.vad.webrtc import (
    _VAD_BOOST_FLOOR,
    _VAD_BOOST_TARGET,
    FRAME_BYTES,
    FRAME_MS,
    WebRtcVadBackend,
    WebRtcVadConfig,
    WebRtcVadSession,
    _vad_boost,
)
from src.vad import EndpointPolicy, VadBackend, VadSession


class FakeVad:
    """webrtcvad.Vad double: is_speech(frame, rate) returns scripted booleans in
    order, repeating the last value once the script runs out. Also records every
    frame passed in, so boost tests can inspect exactly what the decision saw."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.frames = []

    def is_speech(self, frame, rate):
        self.frames.append(frame)
        if self._i < len(self._script):
            val = self._script[self._i]
            self._i += 1
        else:
            val = self._script[-1] if self._script else False
        return val


POLICY = EndpointPolicy(
    silence_ms=100,           # 5 frames of trailing silence to end
    min_speech_ms=40,         # 2 frames of speech to arm
    max_utterance_ms=400,     # 20 frames hard cap
    no_speech_timeout_ms=200,  # 10 frames with no speech -> finalize
)

FRAME = b"\x00" * FRAME_BYTES  # one 20 ms frame of silence-shaped PCM


def make_session(script, *, auto_gain=False, policy=POLICY):
    fake = FakeVad(script)
    session = WebRtcVadSession(fake, policy, auto_gain=auto_gain)
    return session, fake


# --- framing -----------------------------------------------------------------

def test_feed_does_its_own_framing_across_odd_chunk_sizes():
    # Chunks are NOT 640-aligned: the session carries the remainder across feeds
    # and only whole 640-byte frames reach is_speech. 1000 + 1000 bytes = 3 whole
    # frames (1920 bytes) + an 80-byte remainder.
    session, fake = make_session([False])
    assert session.feed(b"\x01" * 1000) is None  # 1 frame consumed, 360 B left over
    assert len(fake.frames) == 1
    assert session.feed(b"\x01" * 1000) is None  # 2 more frames, 80 B left over
    assert len(fake.frames) == 3
    assert session.debug_state()["elapsed_ms"] == 3 * FRAME_MS
    # Every frame handed to the classifier was exactly one 640-byte frame.
    assert all(len(f) == FRAME_BYTES for f in fake.frames)


def test_feed_sub_frame_chunks_accumulate_to_frames():
    # Chunks smaller than a frame produce no decision until 640 bytes accumulate.
    session, fake = make_session([False])
    for _ in range(6):  # 6 * 100 = 600 bytes < 640
        assert session.feed(b"\x00" * 100) is None
    assert fake.frames == []
    session.feed(b"\x00" * 100)  # 700 bytes -> one frame consumed
    assert len(fake.frames) == 1


# --- end-pointing decisions ----------------------------------------------------

def test_endpoint_after_trailing_silence():
    # 3 speech frames (>= 40 ms min) then 5 silence frames (>= 100 ms) -> endpoint.
    session, _ = make_session([True, True, True, False, False, False, False, False])
    assert session.feed(FRAME * 8) == "endpoint"
    state = session.debug_state()
    assert state["speech_detected"] is True
    assert state["silence_ms"] >= POLICY.silence_ms


def test_min_speech_gate_blocks_endpoint_for_short_blip():
    # A single 20 ms speech blip (< min_speech_ms=40) never arms speech_detected,
    # so trailing silence can NOT end-point; the run only ends when the no-speech
    # timeout is reached.
    session, _ = make_session([True, False])
    # 9 frames total (1 speech + 8 silence = 180 ms elapsed): below the timeout.
    assert session.feed(FRAME * 9) is None
    assert session.debug_state()["speech_detected"] is False
    # One more frame crosses no_speech_timeout_ms=200 -> "no_speech", not "endpoint".
    assert session.feed(FRAME) == "no_speech"


def test_maxlen_when_speech_never_ends():
    # Always speech: no trailing silence, so only the max-length cap can finalize.
    session, _ = make_session([True])
    assert session.feed(FRAME * 19) is None
    assert session.feed(FRAME) == "maxlen"  # 20 frames = 400 ms = max_utterance_ms


def test_no_speech_timeout():
    # Never any speech: the no-speech timeout finalizes after 10 frames (200 ms).
    session, _ = make_session([False])
    assert session.feed(FRAME * 9) is None
    assert session.feed(FRAME) == "no_speech"


def test_silence_counter_resets_on_resumed_speech():
    # Speech, almost enough silence, then speech again: the trailing-silence counter
    # resets, so the utterance keeps going instead of end-pointing mid-phrase.
    script = [True, True] + [False] * 4 + [True] + [False] * 4
    session, _ = make_session(script)
    assert session.feed(FRAME * 11) is None  # silence never reached 5 frames in a row
    assert session.feed(FRAME) == "endpoint"  # the 5th trailing-silence frame ends it


def test_debug_state_shape():
    session, _ = make_session([True])
    session.feed(FRAME * 2)
    assert session.debug_state() == {
        "speech_ms": 40, "silence_ms": 0, "elapsed_ms": 40,
        "speech_detected": True, "peak": 0,
    }


# --- auto_gain (decision-only makeup gain) -------------------------------------

def _quiet_frame(value=20):
    return np.full(FRAME_BYTES // 2, value, dtype="<i2").tobytes()


def test_auto_gain_off_passes_frames_verbatim():
    session, fake = make_session([False], auto_gain=False)
    frame = _quiet_frame(8000)
    session.feed(frame)
    assert fake.frames == [frame]
    assert session.debug_state()["peak"] == 0  # no peak tracking when off


def test_auto_gain_tracks_running_peak():
    session, _ = make_session([False], auto_gain=True)
    session.feed(np.full(320, 8000, "<i2").tobytes())
    assert session.debug_state()["peak"] >= 8000


def test_auto_gain_does_not_boost_below_floor():
    # Until the running peak crosses _VAD_BOOST_FLOOR, pre-roll silence must NOT be
    # amplified into false speech: frames reach the classifier unchanged.
    session, fake = make_session([False], auto_gain=True)
    frame = _quiet_frame(_VAD_BOOST_FLOOR - 10)
    session.feed(frame)
    assert session.debug_state()["peak"] == _VAD_BOOST_FLOOR - 10
    assert fake.frames == [frame]  # below the floor -> identity


def test_auto_gain_boosts_decision_only_after_real_signal():
    # Once the peak exceeds the floor, subsequent frames are lifted toward the boost
    # target FOR THE DECISION ONLY (the original bytes are what the pipeline buffers).
    session, fake = make_session([False], auto_gain=True)
    peak = 3000
    loud = _quiet_frame(peak)
    session.feed(loud)
    quiet = _quiet_frame(100)
    session.feed(quiet)
    assert len(fake.frames) == 2
    # Both frames were boosted with the same utterance-level gain (target/peak).
    gain = _VAD_BOOST_TARGET / peak
    expected_quiet = np.clip(
        np.frombuffer(quiet, dtype="<i2").astype(np.float32) * gain, -32768, 32767
    ).astype("<i2").tobytes()
    assert fake.frames[1] == expected_quiet
    assert fake.frames[1] != quiet  # the decision saw a boosted frame...
    # ...but the session never mutates the fed bytes themselves (decision-only).


# --- _vad_boost (the boost primitive, moved verbatim from the pipeline) --------

def test_vad_boost_lifts_quiet_frame_toward_target():
    # A frame whose own samples are small, with a representative running utterance
    # peak (~3000), is scaled by min(target/peak, max_gain) so quiet speech reaches
    # WebRTC VAD's range. The gain is well below the max_gain cap here.
    peak = 3000
    gain = _VAD_BOOST_TARGET / peak
    samples = np.array([100, -200, 300, -50] * 80, dtype="<i2")  # 320 samples = 640 B
    frame = samples.tobytes()
    out = _vad_boost(frame, peak)
    out_samples = np.frombuffer(out, dtype="<i2")
    expected = np.clip(samples.astype(np.float32) * gain, -32768, 32767).astype("<i2")
    assert np.array_equal(out_samples, expected)
    # Sanity: the frame really was amplified (gain > 1).
    assert gain > 1.0
    assert np.max(np.abs(out_samples)) > np.max(np.abs(samples))


def test_vad_boost_below_floor_is_identity():
    # peak below _VAD_BOOST_FLOOR (30) -> treated as pre-roll silence -> frame unchanged.
    frame = (np.array([10, -20, 30, -40] * 80, dtype="<i2")).tobytes()
    assert _vad_boost(frame, 20) == frame


def test_vad_boost_caps_at_max_gain():
    # A peak just above the floor gives a target/peak ratio above the max_gain cap
    # (128); the cap must clamp it. peak=40 -> target/40 ≈ 145 > 128, so the gain is 128.
    peak = 40
    assert _VAD_BOOST_TARGET / peak > 128.0  # uncapped ratio really does exceed the cap
    samples = np.array([5, -6, 7, -8] * 80, dtype="<i2")  # small -> no clipping at 128x
    frame = samples.tobytes()
    out = np.frombuffer(_vad_boost(frame, peak), dtype="<i2")
    expected = np.clip(samples.astype(np.float32) * 128.0, -32768, 32767).astype("<i2")
    assert np.array_equal(out, expected)


def test_vad_boost_handles_empty_and_odd_length():
    # Empty frame -> unchanged. Odd-length (a stray trailing byte) -> still safe; the
    # whole-sample prefix is boosted and no exception is raised.
    assert _vad_boost(b"", 3000) == b""
    odd = (np.array([100, -200], dtype="<i2")).tobytes() + b"\x07"  # 5 bytes
    out = _vad_boost(odd, 3000)
    assert isinstance(out, bytes)
    # A single-byte frame has no whole int16 sample -> returned unchanged.
    assert _vad_boost(b"\x01", 3000) == b"\x01"


# --- backend / provider ---------------------------------------------------------

def test_backend_opens_fresh_session_per_run():
    backend = WebRtcVadBackend(vad_factory=lambda: FakeVad([False]))
    s1 = backend.open(POLICY)
    s2 = backend.open(POLICY)
    assert isinstance(s1, VadSession) and isinstance(s2, VadSession)
    assert s1 is not s2
    # Sessions are independent: state accumulated in one never leaks to the other.
    s1.feed(FRAME * 3)
    assert s2.debug_state()["elapsed_ms"] == 0


def test_backend_default_factory_builds_real_webrtcvad():
    import webrtcvad

    backend = WebRtcVadBackend(aggressiveness=3)
    session = backend.open(POLICY)
    assert isinstance(session._vad, webrtcvad.Vad)


def test_provider_registered_and_creates_configured_backend():
    prov = get_provider("vad", "webrtc")
    assert prov.label == "WebRTC VAD"
    assert prov.ConfigModel is WebRtcVadConfig
    backend = prov.create(WebRtcVadConfig(aggressiveness=1, auto_gain=True), deps=None)
    assert isinstance(backend, VadBackend)
    assert backend._aggressiveness == 1
    assert backend._auto_gain is True


def test_config_defaults_and_schema_extras():
    cfg = WebRtcVadConfig()
    assert cfg.aggressiveness == 2
    assert cfg.auto_gain is False
    # The panel renders the same labeled segment control as before the move: the
    # choices/poles/readout json_schema_extra block moved verbatim from core.vad.
    schema = WebRtcVadConfig.model_json_schema()["properties"]["aggressiveness"]
    assert [c["value"] for c in schema["choices"]] == [0, 1, 2, 3]
    assert [c["label"] for c in schema["choices"]] == [
        "Lenient", "Balanced", "Strict", "Strictest",
    ]
    assert schema["poles"] == ["waits longest", "cuts off soonest"]
    assert schema["readout"] is True


# --- finalize precedence (endpoint > maxlen > no_speech, per the feed docstring) --

def test_endpoint_wins_over_maxlen_when_both_cross_in_one_chunk():
    # One 20-frame chunk arms speech (2 frames) then trails 18 silence frames:
    # silence_ms=360 >= 100 (endpoint) AND elapsed_ms=400 >= 400 (maxlen) are BOTH
    # true after the chunk. The documented precedence picks "endpoint".
    session, _ = make_session([True, True] + [False] * 18)
    assert session.feed(FRAME * 20) == "endpoint"
    state = session.debug_state()
    assert state["elapsed_ms"] >= POLICY.max_utterance_ms  # maxlen really was crossed too


def test_maxlen_wins_over_no_speech_when_timeouts_coincide():
    # With no_speech_timeout_ms == max_utterance_ms and zero speech, both checks
    # become true on the same frame; the precedence order picks "maxlen".
    policy = EndpointPolicy(
        silence_ms=100, min_speech_ms=40,
        max_utterance_ms=200, no_speech_timeout_ms=200,
    )
    session, _ = make_session([False], policy=policy)
    assert session.feed(FRAME * 10) == "maxlen"  # 10 frames = 200 ms hits both caps


# --- _vad_boost: no-attenuation branch (gain <= 1) --------------------------------

def test_vad_boost_never_attenuates_loud_signal():
    # When the running peak already sits at or above the boost target, gain <= 1:
    # the frame must come back byte-identical (the boost only ever lifts, never cuts).
    frame = np.array([100, -200, 300, -400] * 80, dtype="<i2").tobytes()
    assert _vad_boost(frame, int(_VAD_BOOST_TARGET)) == frame   # gain == 1 exactly
    assert _vad_boost(frame, 8000) == frame                     # gain < 1


# --- framing with auto_gain over odd-length chunks ---------------------------------

def test_feed_auto_gain_odd_chunk_consumes_frame_and_carries_remainder():
    # A 641-byte chunk with auto_gain on: peak tracking ignores the torn trailing
    # byte, exactly one 640-byte frame reaches the classifier and the 1-byte
    # remainder is carried into the next feed.
    session, fake = make_session([False], auto_gain=True)
    assert session.feed(b"\x01" * 641) is None
    assert len(fake.frames) == 1
    assert len(session._rem) == 1
    # The remainder completes the next frame: 1 + 639 = 640 bytes -> one more frame.
    session.feed(b"\x01" * 639)
    assert len(fake.frames) == 2
    assert len(session._rem) == 0


# --- config bounds ------------------------------------------------------------------

def test_config_aggressiveness_out_of_bounds_raises():
    # WebRTC only accepts modes 0..3; the pydantic ge/le bounds must reject the rest.
    with pytest.raises(ValidationError):
        WebRtcVadConfig(aggressiveness=4)
    with pytest.raises(ValidationError):
        WebRtcVadConfig(aggressiveness=-1)


# --- integration with the real webrtcvad C library ----------------------------------

def test_real_webrtcvad_accepts_frames_and_hits_no_speech_timeout():
    # No fake factory: the default backend builds a real webrtcvad.Vad, pinning that
    # FRAME_BYTES/SAMPLE_RATE match the C library's accepted frame sizes (a wrong
    # frame size would raise inside is_speech). Zero frames are silence, so feed()
    # returns None until the no-speech timeout and "no_speech" exactly when crossed.
    backend = WebRtcVadBackend()
    policy = EndpointPolicy(
        silence_ms=10_000, min_speech_ms=40,
        max_utterance_ms=10_000, no_speech_timeout_ms=60,
    )
    session = backend.open(policy)
    silence = b"\x00" * FRAME_BYTES
    assert session.feed(silence) is None          # 20 ms
    assert session.feed(silence) is None          # 40 ms
    assert session.feed(silence) == "no_speech"   # 60 ms == the timeout


# --- generic stage contract over every registered vad provider ----------------------

@pytest.mark.parametrize("provider_id", sorted(providers("vad")))
def test_vad_provider_contract(provider_id):
    # Every "vad" provider must honor the stage contract: create() with its config
    # defaults yields a VadBackend, open() yields a VadSession, feed() returns None
    # or a known finalize reason, debug_state() returns a dict. Shape-only checks so
    # new providers pass without snapshotting their internals.
    prov = providers("vad")[provider_id]
    cfg = prov.ConfigModel()
    # Model-backed providers (e.g. silero) fail fast in create() when the model
    # file isn't downloaded; skip them rather than require models in CI.
    model_path = getattr(cfg, "model_path", None)
    if model_path and not os.path.exists(model_path):
        pytest.skip(f"model file not present: {model_path}")
    backend = prov.create(cfg, deps=None)
    assert isinstance(backend, VadBackend)
    session = backend.open(EndpointPolicy(
        silence_ms=100, min_speech_ms=40,
        max_utterance_ms=400, no_speech_timeout_ms=200,
    ))
    assert isinstance(session, VadSession)
    reason = session.feed(b"")
    assert reason is None or reason in {"endpoint", "maxlen", "no_speech"}
    assert isinstance(session.debug_state(), dict)
