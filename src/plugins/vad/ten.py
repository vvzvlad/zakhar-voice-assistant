"""TEN VAD provider: frame-level neural speech/no-speech classifier (16 ms hops).

Wraps the `ten-vad` pip package (Apache-2, TEN Framework): a ctypes binding over
a bundled native runtime — the wheel ships the .so/.dylib/.dll inside, so there
are NO extra Python dependencies beyond numpy. Upstream benchmarks claim faster
speech-to-silence transition detection than Silero, which is exactly what our
end-pointing cares about.

Engine interface (ten-vad 1.0.6.8, verified against the installed package):
  TenVad(hop_size=256, threshold=0.5) — the handle carries internal state, so
  ONE INSTANCE PER UTTERANCE (created in open(), like Silero's per-session LSTM
  state). process(frame) takes a numpy int16 array of exactly hop_size samples
  (256 = 16 ms @ 16 kHz) and returns (probability: float 0..1, flag: 0/1) with
  the threshold applied internally by the native library.
The generic end-pointing thresholds stay in core.vad and arrive per run as an
EndpointPolicy; this module owns only the TEN specifics (the 512-byte hop
framing, the per-utterance engine instance, the threshold).
"""

import numpy as np
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.vad import EndpointPolicy, EndpointTracker, VadBackend, VadSession

# TEN VAD consumes fixed 256-sample hops = 16 ms = 512 bytes of 16-bit mono PCM
# at the pipeline's 16 kHz stream rate.
HOP_SAMPLES = 256
HOP_BYTES = 512  # 16-bit mono, 16 ms @ 16 kHz
HOP_MS = 16


class TenVadSession(VadSession):
    """One utterance's TEN VAD session.

    feed() consumes arbitrary-size PCM chunks: device chunks aren't 512-aligned,
    so leftover bytes are kept in a remainder buffer and only whole 512-byte
    hops reach the engine. The session owns the TEN engine specifics (the hop
    framing and the per-utterance engine instance — the native handle carries
    internal state, so it is never shared across sessions); the speech/silence
    counters and the endpoint / maxlen / no_speech decision live in the shared
    EndpointTracker, driven one 16 ms hop at a time.

    feed() is synchronous on purpose: it is called on the event loop and one
    256-sample hop infers in well under 1 ms on CPU.
    """

    def __init__(self, engine, policy: EndpointPolicy):
        self._engine = engine          # exposes process(int16[256]) -> (prob, flag)
        self._tracker = EndpointTracker(policy)
        self._rem = bytearray()        # leftover bytes between non-512-aligned chunks
        self._last_prob = 0.0          # last hop's probability, for debug_state()

    def feed(self, chunk: bytes) -> str | None:
        # Consume whole 512-byte hops only; keep the remainder for the next
        # chunk. Each hop drives the shared tracker; the LAST hop's verdict is
        # the chunk's verdict — identical to deciding after the whole chunk,
        # because update() evaluates the decision on the same counters an
        # end-of-chunk check would read. Known boundary: a chunk that completes
        # no hop returns None without re-checking thresholds — equivalent for
        # any positive policy values (the panel enforces positive ms thresholds),
        # since hop-less chunks leave the counters untouched.
        reason = None
        self._rem.extend(chunk)
        while len(self._rem) >= HOP_BYTES:
            hop = bytes(self._rem[:HOP_BYTES])
            del self._rem[:HOP_BYTES]
            samples = np.frombuffer(hop, dtype="<i2")
            prob, flag = self._engine.process(samples)
            self._last_prob = float(prob)
            # The engine's flag IS the thresholded decision: the native library
            # was created with our configured threshold and applies it internally
            # (returning both prob and flag). Trusting the flag is more faithful
            # to the installed API than re-comparing prob ourselves — if upstream
            # ever adds internal onset/hangover logic to the flag, we inherit it.
            speech = bool(flag)
            reason = self._tracker.update(speech, HOP_MS)
        return reason

    def debug_state(self) -> dict:
        # The shared tracker's counters plus the engine extras (last probability),
        # keeping key parity with the pipeline's VAD-finalize debug log line.
        return {**self._tracker.debug_state(), "prob": round(self._last_prob, 2)}


class TenVadBackend(VadBackend):
    """Stateless TEN session factory; the engine itself is built PER SESSION.

    Unlike Silero's shared ONNX session, the TenVad handle carries internal
    state, so open() constructs a fresh engine per utterance (cheap: the native
    library is already loaded after the first construction). The `ten_vad`
    import happens in the constructor — fail fast like Vosk's Model(model_path)
    and Silero's InferenceSession load, so reconfig's create() surfaces a
    missing package immediately and keeps the old backend. `engine_factory` is
    injectable so tests never need the real native library.
    """

    def __init__(self, threshold: float, *, engine_factory=None):
        self._threshold = threshold
        if engine_factory is None:
            # Lazy import: the package is only required when the TEN backend is
            # actually created (never at module import — src/plugins/__init__.py
            # imports this module at boot on machines without the package).
            try:
                from ten_vad import TenVad
            except ImportError as e:
                raise ImportError(
                    "TEN VAD requires the `ten-vad` package "
                    "(pip install ten-vad); it bundles its native runtime."
                ) from e
            engine_factory = lambda: TenVad(  # noqa: E731
                hop_size=HOP_SAMPLES, threshold=self._threshold
            )
        self._engine_factory = engine_factory

    def open(self, policy: EndpointPolicy) -> TenVadSession:
        # Fresh engine (internal native state) + fresh tracker per utterance.
        return TenVadSession(self._engine_factory(), policy)


class TenVadConfig(BaseModel):
    threshold: float = Field(
        0.5, ge=0.05, le=0.95,
        title="Speech threshold",
        json_schema_extra={"widget": "slider"},
        description="Hops with speech probability at or above this count as speech. Lower = more sensitive.",
    )


@register
class TenVadProvider(Provider):
    category = "vad"
    id = "ten"
    label = "TEN VAD (neural)"
    ConfigModel = TenVadConfig

    def create(self, cfg: TenVadConfig, deps: Deps):
        # Needs nothing from Deps: the classifier is pure CPU over the PCM stream.
        # The ten_vad import (and thus the missing-package failure) happens here.
        return TenVadBackend(cfg.threshold)
