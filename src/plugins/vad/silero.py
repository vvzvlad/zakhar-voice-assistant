"""Silero VAD provider: neural speech/no-speech classifier behind the vad stage.

Runs the bare silero_vad.onnx model directly via onnxruntime (the pip package
`silero-vad` would pull torch+torchaudio — gigabytes; the bare ONNX model is
~2 MB and onnxruntime is already a project dependency through piper-tts).

Model interface (silero-vad v5/v6 onnx, 16 kHz path; verified against the real
file — feeding bare 512-sample windows WITHOUT the context prefix silently
yields ~0 probabilities even for clean speech):
  inputs:  "input"  float32 [batch, 576]    — 64 context samples (the tail of
           the previous window, zeros at utterance start) + 512 new samples (32 ms)
           "state"  float32 [2, batch, 128] — LSTM state, fed back every call
           "sr"     int64 scalar            — sample rate (16000)
  outputs: "output" float32 [batch, 1]      — speech probability 0..1
           "stateN" float32 [2, batch, 128] — next state
Audio is int16 PCM normalized to float32 by /32768.0. The generic end-pointing
thresholds stay in core.vad and arrive per run as an EndpointPolicy; this module
owns only the Silero specifics (windowing, the ONNX session, the probability
thresholds with hysteresis).
"""

import numpy as np
from pydantic import BaseModel, Field, model_validator

from src.plugins.base import Deps, Provider, register
from src.vad import SAMPLE_RATE, EndpointPolicy, EndpointTracker, VadBackend, VadSession

# Silero's 16 kHz path consumes exactly 512-sample windows = 32 ms = 1024 bytes
# of 16-bit mono PCM. Each model call additionally takes the last 64 samples of
# the PREVIOUS window as context (zeros at utterance start), so the actual
# "input" tensor is 64 + 512 = 576 samples wide.
WINDOW_SAMPLES = 512
WINDOW_BYTES = 1024  # 16-bit mono, 32 ms @ 16 kHz
WINDOW_MS = 32
CONTEXT_SAMPLES = 64


class SileroVadSession(VadSession):
    """One utterance's Silero VAD session.

    feed() consumes arbitrary-size PCM chunks: device chunks aren't 1024-aligned,
    so leftover bytes are kept in a remainder buffer and only whole 1024-byte
    windows reach the model. The session owns the Silero engine specifics (the
    windowing, the per-utterance LSTM state, the threshold hysteresis); the
    speech/silence counters and the endpoint / maxlen / no_speech decision live
    in the shared EndpointTracker, driven one 32 ms window at a time.

    feed() is synchronous on purpose: it is called on the event loop and one
    512-sample window infers in well under 1 ms on CPU.
    """

    def __init__(self, ort_session, policy: EndpointPolicy, threshold: float,
                 neg_threshold: float):
        self._ort = ort_session        # exposes run(None, feeds) -> [output, stateN]
        self._tracker = EndpointTracker(policy)
        self._threshold = threshold
        self._neg_threshold = neg_threshold
        self._rem = bytearray()        # leftover bytes between non-1024-aligned chunks
        # Fresh LSTM state per utterance; fed back to the model after every window.
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # 64-sample context: the tail of the previous window, zeros at start.
        # Required by the v5/v6 model — without it the probabilities collapse to ~0.
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._speaking = False         # hysteresis: are we inside a speech run?
        self._last_prob = 0.0          # last window's probability, for debug_state()

    def feed(self, chunk: bytes) -> str | None:
        # Consume whole 1024-byte windows only; keep the remainder for the next
        # chunk. Each window drives the shared tracker; the LAST window's verdict
        # is the chunk's verdict — identical to deciding after the whole chunk,
        # because update() evaluates the decision on the same counters an
        # end-of-chunk check would read. Known boundary: a chunk that completes
        # no window returns None without re-checking thresholds — equivalent for
        # any positive policy values (the panel enforces positive ms thresholds),
        # since window-less chunks leave the counters untouched.
        reason = None
        self._rem.extend(chunk)
        while len(self._rem) >= WINDOW_BYTES:
            window = bytes(self._rem[:WINDOW_BYTES])
            del self._rem[:WINDOW_BYTES]
            samples = (np.frombuffer(window, dtype="<i2").astype(np.float32)
                       / 32768.0)
            # Prepend the previous window's 64-sample tail (context) -> [1, 576].
            x = np.concatenate([self._context, samples]).reshape(
                1, CONTEXT_SAMPLES + WINDOW_SAMPLES
            )
            outputs = self._ort.run(
                None, {"input": x, "state": self._state, "sr": self._sr}
            )
            # Model output order is ["output", "stateN"]; be defensive about the
            # output shape ([batch, 1] today) by flattening before indexing.
            prob = float(np.asarray(outputs[0]).reshape(-1)[0])
            self._state = outputs[1]
            self._context = samples[-CONTEXT_SAMPLES:]
            self._last_prob = prob
            # Hysteresis: speech starts at `threshold` but only ends below
            # `neg_threshold`, so breathy/trailing sounds don't flicker the flag.
            if self._speaking:
                speech = prob >= self._neg_threshold
            else:
                speech = prob >= self._threshold
            self._speaking = speech
            reason = self._tracker.update(speech, WINDOW_MS)
        return reason

    def debug_state(self) -> dict:
        # The shared tracker's counters plus the engine extras (last probability),
        # keeping key parity with the pipeline's VAD-finalize debug log line.
        return {**self._tracker.debug_state(), "prob": round(self._last_prob, 2)}


class SileroVadBackend(VadBackend):
    """Stateless Silero session factory around one shared ONNX session.

    The InferenceSession is built ONCE in the constructor (fail fast if the
    model file is missing, like Vosk's Model(model_path)) and shared across
    sessions: onnxruntime's run() is thread-safe and stateless — all per-
    utterance state (the LSTM state, the hysteresis flag) lives in the
    SileroVadSession. `ort_session` is injectable so tests never need the
    real model file. Thread pools are pinned to 1 thread: the model is tiny,
    and extra threads would only contend with the asyncio app.
    """

    def __init__(self, model_path: str, threshold: float, neg_threshold: float,
                 *, ort_session=None):
        if ort_session is None:
            # Lazy import: onnxruntime is only required when the Silero backend
            # is actually selected at runtime (never in unit tests).
            import onnxruntime

            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            ort_session = onnxruntime.InferenceSession(
                model_path, sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        self._ort = ort_session
        self._threshold = threshold
        self._neg_threshold = neg_threshold

    def open(self, policy: EndpointPolicy) -> SileroVadSession:
        # Fresh LSTM state / tracker per utterance; the ONNX session is shared.
        return SileroVadSession(self._ort, policy, self._threshold, self._neg_threshold)


class SileroVadConfig(BaseModel):
    model_path: str = "models/silero_vad.onnx"
    threshold: float = Field(
        0.5, ge=0.05, le=0.95,
        title="Speech threshold",
        json_schema_extra={"widget": "slider"},
        description="Windows with speech probability at or above this count as speech. Lower = more sensitive (catches quiet speech, more false triggers from noise).",
    )
    neg_threshold: float = Field(
        0.35, ge=0.05, le=0.95,
        title="Speech-end threshold",
        json_schema_extra={"widget": "slider"},
        description="Hysteresis: once speech has started, it only ends when the probability drops below this. Must be at or below the speech threshold; prevents flicker on breathy/trailing sounds.",
    )

    @model_validator(mode="after")
    def _clamp_neg_threshold(self):
        # The hysteresis only makes sense with neg_threshold <= threshold; a
        # misconfigured pair (sliders moved independently) is clamped instead of
        # rejected so the panel never bricks the provider.
        self.neg_threshold = min(self.neg_threshold, self.threshold)
        return self


@register
class SileroVadProvider(Provider):
    category = "vad"
    id = "silero"
    label = "Silero VAD (neural)"
    ConfigModel = SileroVadConfig

    def create(self, cfg: SileroVadConfig, deps: Deps):
        # Needs nothing from Deps: the classifier is pure CPU over the PCM stream.
        # The heavy InferenceSession load happens here — fine, reconfig runs
        # create() off the event loop (same pattern as the Vosk model load).
        return SileroVadBackend(cfg.model_path, cfg.threshold, cfg.neg_threshold)
