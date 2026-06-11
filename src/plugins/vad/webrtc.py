"""WebRTC VAD provider: the speech/no-speech classifier behind the vad stage.

Owns everything WebRTC-specific that used to live inline in the pipeline: the
640-byte framing, the webrtcvad.Vad object, the per-utterance speech/silence
counters and the optional decision-only makeup gain (``auto_gain``) for the
quiet less-processed mic channel. The generic end-pointing thresholds stay in
core.vad and arrive per run as an EndpointPolicy.
"""

import numpy as np
import webrtcvad
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.vad import SAMPLE_RATE, EndpointPolicy, EndpointTracker, VadBackend, VadSession

# WebRTC VAD requires mono 16-bit PCM frames of exactly 10/20/30 ms at 16 kHz.
# We use 20 ms frames = 16000 * 2 * 20/1000 = 640 bytes.
FRAME_MS = 20
FRAME_BYTES = 640  # 16-bit mono, 20 ms @ 16 kHz

# Target peak (~-3 dBFS) the VAD makeup gain lifts quiet audio toward, and the floor
# below which the running utterance peak is treated as "no real signal yet" (so leading
# pre-roll silence is NOT amplified into false speech).
# Target int16 peak the boost lifts the quiet channel's SPEECH toward. Deliberately
# MODERATE (~-15 dBFS, not full-scale): a higher target needs a larger gain, which also
# amplifies the channel's (clean, quiet) trailing silence enough that WebRTC VAD reads
# it as speech and never end-points (the utterance runs to max-length). At -15 dBFS the
# quiet channel's speech (peak ~50) reaches ~5800 (clearly speech) while its silence
# (peak ~5) stays ~600 (clearly non-speech), so the pause is still detected → fast end-point.
_VAD_BOOST_TARGET = 5824.0    # 32767 * 10**(-15/20)
# int16 peak below which we treat the (very clean) less-processed channel as pre-roll
# silence and don't boost — keeps leading noise from being amplified into false speech.
# Must sit BELOW the real speech level of the quiet channel (measured ~50-80) and ABOVE
# its silence floor (~1-5), so the boost engages once the wake word is heard.
_VAD_BOOST_FLOOR = 30


def _vad_boost(frame: bytes, peak: int, max_gain: float = 128.0) -> bytes:
    """Lift a 16-bit mono PCM frame toward _VAD_BOOST_TARGET for the VAD decision only.

    `peak` is the running peak of the WHOLE utterance so far (not this frame), so the
    gain (target/peak) is the same for every frame once the loud wake word has set the
    peak — this preserves the speech-vs-silence energy ratio (silence stays detectable)
    while bringing the quiet less-processed channel into WebRTC VAD's range. Returns the
    frame unchanged until a real signal has been seen (peak < floor) or when no boost is
    needed (gain <= 1). Used ONLY for the is_speech() decision — never stored.
    """
    if peak < _VAD_BOOST_FLOOR:
        return frame
    gain = min(_VAD_BOOST_TARGET / peak, max_gain)
    if gain <= 1.0:
        return frame
    n = len(frame) - (len(frame) % 2)
    if n == 0:
        return frame
    s = np.frombuffer(frame[:n], dtype="<i2").astype(np.float32) * gain
    return np.clip(s, -32768, 32767).astype("<i2").tobytes()


class WebRtcVadSession(VadSession):
    """One utterance's WebRTC VAD session.

    feed() consumes arbitrary-size PCM chunks: device chunks aren't 640-aligned,
    so leftover bytes are kept in a remainder buffer and only whole 640-byte
    frames reach webrtcvad. The session owns only the WebRTC engine specifics
    (framing, the Vad object, the decision-only auto_gain boost and its peak
    tracking); the speech/silence counters and the endpoint / maxlen / no_speech
    decision live in the shared EndpointTracker, driven one 20 ms frame at a time.
    """

    def __init__(self, vad, policy: EndpointPolicy, *, auto_gain: bool = False):
        self._vad = vad                # exposes is_speech(frame, rate) -> bool
        self._tracker = EndpointTracker(policy)
        self._auto_gain = auto_gain
        self._rem = bytearray()        # leftover bytes between non-640-aligned chunks
        self._peak = 0                 # running peak of the utterance (for the frame boost)

    def feed(self, chunk: bytes) -> str | None:
        # When auto_gain is on, track the running peak of the whole utterance so the
        # frame boost below can lift this quiet channel into WebRTC VAD's range. The
        # pipeline's buffer/STT path is unaffected (it normalizes per utterance).
        if self._auto_gain and chunk:
            m = len(chunk) - (len(chunk) % 2)
            if m:
                self._peak = max(
                    self._peak,
                    int(np.abs(np.frombuffer(chunk[:m], dtype="<i2")).max()),
                )

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
            # Boost the frame for the speech/silence decision only (never stored)
            # so WebRTC VAD can detect the quiet less-processed channel.
            vframe = _vad_boost(frame, self._peak) if self._auto_gain else frame
            speech = self._vad.is_speech(vframe, SAMPLE_RATE)
            reason = self._tracker.update(speech, FRAME_MS)
        return reason

    def debug_state(self) -> dict:
        # The shared tracker's counters plus the engine extras (running peak),
        # keeping key parity with the pipeline's VAD-finalize debug log line.
        return {**self._tracker.debug_state(), "peak": self._peak}


class WebRtcVadBackend(VadBackend):
    """Stateless WebRTC session factory.

    A fresh webrtcvad.Vad is constructed per session (cheap): the aggressiveness is
    baked into the Vad object at construction, so building it per run replaces the
    old on_start rebuild hack in the pipeline. `vad_factory` lets tests inject a
    fake Vad object exposing is_speech(frame, rate).
    """

    def __init__(self, *, aggressiveness: int = 2, auto_gain: bool = False,
                 vad_factory=None):
        self._aggressiveness = aggressiveness
        self._auto_gain = auto_gain
        self._vad_factory = vad_factory

    def _make_vad(self):
        if self._vad_factory is not None:
            return self._vad_factory()
        return webrtcvad.Vad(self._aggressiveness)

    def open(self, policy: EndpointPolicy) -> WebRtcVadSession:
        return WebRtcVadSession(self._make_vad(), policy, auto_gain=self._auto_gain)


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
    auto_gain: bool = Field(
        False,
        title="VAD auto gain",
        description="Boost quiet frames for the speech/no-speech decision only (the stored audio is untouched). Lets WebRTC VAD end-point the quiet, less-processed mic channel.",
    )


@register
class WebRtcVadProvider(Provider):
    category = "vad"
    id = "webrtc"
    label = "WebRTC VAD"
    ConfigModel = WebRtcVadConfig

    def create(self, cfg: WebRtcVadConfig, deps: Deps):
        # Needs nothing from Deps: the classifier is pure CPU over the PCM stream.
        return WebRtcVadBackend(aggressiveness=cfg.aggressiveness, auto_gain=cfg.auto_gain)
