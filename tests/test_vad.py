"""Unit tests for EndpointTracker, the engine-independent end-pointing state
machine in src/vad.py.

These drive update() directly with synthetic speech/non-speech windows: every
VAD session (WebRTC 20 ms, Silero 32 ms, TEN 16 ms) shares this exact decision
logic, so the tracker is tested on its own, independent of any audio engine.
The WebRTC session tests in tests/test_vad_webrtc.py remain the parity oracle
for the moved logic.
"""

from src.vad import EndpointPolicy, EndpointTracker

POLICY = EndpointPolicy(
    silence_ms=100,            # 100 ms of trailing silence to end
    min_speech_ms=40,          # 40 ms of speech to arm speech_detected
    max_utterance_ms=400,      # 400 ms hard cap
    no_speech_timeout_ms=200,  # 200 ms with no speech -> finalize
)


def drive(tracker, pattern, window_ms=20):
    """Feed a string of 's' (speech) / '.' (silence) windows; return the first
    non-None reason (and stop there, like a session would) or None."""
    for ch in pattern:
        reason = tracker.update(ch == "s", window_ms)
        if reason is not None:
            return reason
    return None


# --- endpoint ------------------------------------------------------------------

def test_endpoint_after_trailing_silence():
    # 60 ms speech (>= min_speech_ms) then 100 ms silence -> endpoint.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "sss" + "." * 5) == "endpoint"
    state = tracker.debug_state()
    assert state["speech_detected"] is True
    assert state["silence_ms"] >= POLICY.silence_ms


def test_endpoint_fires_exactly_at_threshold():
    # The 5th 20 ms silence window is the first to satisfy silence_ms >= 100.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "ss....") is None  # 80 ms trailing silence: not yet
    assert tracker.update(False, 20) == "endpoint"  # 100 ms: exactly the threshold


# --- min_speech gate -------------------------------------------------------------

def test_min_speech_gate_blocks_endpoint_for_short_blip():
    # A single 20 ms blip (< min_speech_ms=40) never arms speech_detected, so
    # trailing silence can NOT end-point; the run ends via the no-speech timeout.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "s" + "." * 8) is None  # 180 ms elapsed: below the timeout
    assert tracker.debug_state()["speech_detected"] is False
    assert tracker.update(False, 20) == "no_speech"  # 200 ms -> timeout, not endpoint


def test_min_speech_accumulates_across_nonconsecutive_windows():
    # speech_ms accumulates over the utterance: two separated 20 ms blips reach
    # min_speech_ms=40 and arm speech_detected just like 40 ms in a row.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "s.s") is None
    assert tracker.debug_state()["speech_detected"] is True


# --- maxlen ----------------------------------------------------------------------

def test_maxlen_when_speech_never_ends():
    # Continuous speech: no trailing silence, only the hard cap can finalize.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "s" * 19) is None
    assert tracker.update(True, 20) == "maxlen"  # 20 windows = 400 ms = the cap


# --- no_speech -------------------------------------------------------------------

def test_no_speech_timeout():
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "." * 9) is None
    assert tracker.update(False, 20) == "no_speech"  # 200 ms == the timeout


# --- silence reset ----------------------------------------------------------------

def test_silence_counter_resets_on_resumed_speech():
    # Speech, almost enough silence, then speech again: the trailing-silence
    # counter resets, so the utterance keeps going instead of ending mid-phrase.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "ss" + "." * 4 + "s" + "." * 4) is None
    assert tracker.debug_state()["silence_ms"] == 80
    assert tracker.update(False, 20) == "endpoint"  # the 5th trailing-silence window


# --- mixed window sizes -----------------------------------------------------------

def test_32ms_windows_reach_same_endpoint_decision_as_20ms():
    # Silero-style 32 ms windows: 2 speech windows (64 ms >= 40) arm the gate,
    # then 4 silence windows (128 ms >= 100) end-point — the same decision the
    # 20 ms-window engine reaches for equivalent durations.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "ss...", window_ms=32) is None  # 96 ms silence: not yet
    assert tracker.update(False, 32) == "endpoint"        # 128 ms crosses 100


def test_16ms_windows_reach_same_no_speech_decision_as_20ms():
    # TEN-style 16 ms hops: pure silence crosses no_speech_timeout_ms=200 on the
    # 13th window (208 ms), same reason a 20 ms engine returns on its 10th.
    tracker = EndpointTracker(POLICY)
    assert drive(tracker, "." * 12, window_ms=16) is None  # 192 ms: not yet
    assert tracker.update(False, 16) == "no_speech"        # 208 ms crosses 200


def test_window_size_only_changes_granularity_not_semantics():
    # maxlen by elapsed time: 400 ms of continuous speech finalizes as "maxlen"
    # regardless of how it is sliced (20 x 20 ms vs 12.5 -> 13 x 32 ms).
    t20 = EndpointTracker(POLICY)
    t32 = EndpointTracker(POLICY)
    assert drive(t20, "s" * 20) == "maxlen"
    assert drive(t32, "s" * 13, window_ms=32) == "maxlen"  # 416 ms >= 400


# --- precedence (endpoint > maxlen > no_speech) -------------------------------------

def test_endpoint_wins_over_maxlen_on_the_same_window():
    # Arm speech, then trail silence so that silence_ms and elapsed_ms cross
    # their thresholds on the SAME update; precedence picks "endpoint".
    policy = EndpointPolicy(
        silence_ms=100, min_speech_ms=40,
        max_utterance_ms=200, no_speech_timeout_ms=200,
    )
    tracker = EndpointTracker(policy)
    assert drive(tracker, "sssss....") is None       # 180 ms elapsed, 80 ms silence
    assert tracker.update(False, 20) == "endpoint"   # both cross at 200 ms elapsed


def test_maxlen_wins_over_no_speech_when_timeouts_coincide():
    policy = EndpointPolicy(
        silence_ms=100, min_speech_ms=40,
        max_utterance_ms=200, no_speech_timeout_ms=200,
    )
    tracker = EndpointTracker(policy)
    assert drive(tracker, "." * 10) == "maxlen"  # 200 ms hits both caps -> maxlen


# --- debug_state ---------------------------------------------------------------------

def test_debug_state_keys_and_values():
    tracker = EndpointTracker(POLICY)
    assert tracker.debug_state() == {
        "speech_ms": 0, "silence_ms": 0, "elapsed_ms": 0, "speech_detected": False,
    }
    drive(tracker, "ss.")
    assert tracker.debug_state() == {
        "speech_ms": 40, "silence_ms": 20, "elapsed_ms": 60, "speech_detected": True,
    }
