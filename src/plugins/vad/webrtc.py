"""WebRTC VAD provider: the speech/no-speech classifier behind the vad stage.

Owns everything WebRTC-specific that used to live inline in the pipeline: the
640-byte framing, the webrtcvad.Vad object and the per-utterance speech/silence
counters. The generic end-pointing thresholds stay in core.vad and arrive per
run as an EndpointPolicy. The decision-only makeup gain for the quiet
less-processed mic channel is the PIPELINE's concern now (core.vad.mic_auto_gain,
applied before any VAD backend sees the chunk), not this plugin's.
"""

import webrtcvad
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.vad import SAMPLE_RATE, EndpointPolicy, EndpointTracker, VadBackend, VadSession

# WebRTC VAD requires mono 16-bit PCM frames of exactly 10/20/30 ms at 16 kHz.
# We use 20 ms frames = 16000 * 2 * 20/1000 = 640 bytes.
FRAME_MS = 20
FRAME_BYTES = 640  # 16-bit mono, 20 ms @ 16 kHz


class WebRtcVadSession(VadSession):
    """One utterance's WebRTC VAD session.

    feed() consumes arbitrary-size PCM chunks: device chunks aren't 640-aligned,
    so leftover bytes are kept in a remainder buffer and only whole 640-byte
    frames reach webrtcvad. The session owns only the WebRTC engine specifics
    (framing, the Vad object); the speech/silence counters and the endpoint /
    maxlen / no_speech decision live in the shared EndpointTracker, driven one
    20 ms frame at a time.
    """

    def __init__(self, vad, policy: EndpointPolicy):
        self._vad = vad                # exposes is_speech(frame, rate) -> bool
        self._tracker = EndpointTracker(policy)
        self._rem = bytearray()        # leftover bytes between non-640-aligned chunks

    def feed(self, chunk: bytes) -> str | None:
        # Consume whole 640-byte frames only; keep the remainder for the next chunk.
        # Each frame drives the shared tracker; the LAST frame's verdict is the
        # chunk's verdict — identical to the original decide-after-the-whole-chunk
        # logic, because update() evaluates the decision on the same counters the
        # old end-of-chunk check read (so a mid-chunk near-endpoint that speech
        # later in the chunk cancels still keeps the session listening). Known
        # boundary: a chunk that completes no frame returns None without
        # re-checking thresholds — equivalent for any positive policy values
        # (the panel enforces positive ms thresholds), since frame-less chunks
        # leave the counters untouched.
        reason = None
        self._rem.extend(chunk)
        while len(self._rem) >= FRAME_BYTES:
            frame = bytes(self._rem[:FRAME_BYTES])
            del self._rem[:FRAME_BYTES]
            speech = self._vad.is_speech(frame, SAMPLE_RATE)
            reason = self._tracker.update(speech, FRAME_MS)
        return reason

    def debug_state(self) -> dict:
        return self._tracker.debug_state()


class WebRtcVadBackend(VadBackend):
    """Stateless WebRTC session factory.

    A fresh webrtcvad.Vad is constructed per session (cheap): the aggressiveness is
    baked into the Vad object at construction, so building it per run replaces the
    old on_start rebuild hack in the pipeline. `vad_factory` lets tests inject a
    fake Vad object exposing is_speech(frame, rate).
    """

    def __init__(self, *, aggressiveness: int = 2, vad_factory=None):
        self._aggressiveness = aggressiveness
        self._vad_factory = vad_factory

    def _make_vad(self):
        if self._vad_factory is not None:
            return self._vad_factory()
        return webrtcvad.Vad(self._aggressiveness)

    def open(self, policy: EndpointPolicy) -> WebRtcVadSession:
        return WebRtcVadSession(self._make_vad(), policy)


class WebRtcVadConfig(BaseModel):
    aggressiveness: int = Field(
        2, ge=0, le=3,
        title="Speech detection strictness",
        # Rendered as a labeled segment control (see SchemaForm.ScaleSeg): each level
        # gets a word label, the extremes get pole captions, and the numeric value is
        # shown small for debugging. The "stricter = cuts sooner" explanation lives in
        # those labels, so the description stays short.
        json_schema_extra={
            "choices": [
                {"value": 0, "label": "Lenient"},
                {"value": 1, "label": "Balanced"},
                {"value": 2, "label": "Strict"},
                {"value": 3, "label": "Strictest"},
            ],
            "poles": ["waits longest", "cuts off soonest"],
            "readout": True,
        },
        description="How strictly WebRTC VAD decides that speech has ended.",
    )


@register
class WebRtcVadProvider(Provider):
    category = "vad"
    id = "webrtc"
    label = "WebRTC VAD"
    ConfigModel = WebRtcVadConfig

    def create(self, cfg: WebRtcVadConfig, deps: Deps):
        # Needs nothing from Deps: the classifier is pure CPU over the PCM stream.
        return WebRtcVadBackend(aggressiveness=cfg.aggressiveness)
