"""One voice run of a device: a wake's run or a follow-up window's.

The pipeline (src/pipeline.py) keeps the run it serves as a Run; everything that belongs
to one run — its audio, its VAD session, its STT sessions, how far its wake got through
the wake-word check and the cross-device arbitration — lives on it, so starting a run
is constructing one, and a run another wake replaces keeps its own state.
"""

from src.vad import SAMPLE_RATE

# A wake session's stream opens with this much pre-roll, ending where the device
# detected the wake word (docs/device-protocol.md, "Voice session").
PREROLL_MS = 2000
PREROLL_BYTES = PREROLL_MS * SAMPLE_RATE * 2 // 1000


class Run:
    """The state of one run, from its wake (or its follow-up window) to its end."""

    def __init__(self, cid, word: str, vad, vad_desc: str, *, followup=None, taken=False):
        # The conversation id (the device's session) the run's events are labelled with.
        self.cid = cid
        # The wake word; a follow-up window keeps the word of its session's wake.
        self.word = word
        # The run's VAD session (it owns all end-pointing state) and the identity of
        # the backend that opened it, for the finalize log line.
        self.vad = vad
        self.vad_desc = vad_desc
        # (name, recognized request, un-accented spoken reply) of the turn before a
        # follow-up window, for the follow-up check; None for a wake's run.
        self.followup = followup
        # The selected mic channel (what VAD and STT get), and the other channel, kept
        # only for the stored stereo diagnostic WAV.
        self.buffer = bytearray()
        self.buffer2 = bytearray()
        # Live streaming STT sessions of the chains whose STT starts at speech, by
        # chain name; the run's _run takes them over once it is claimed.
        self.stt_streams: dict = {}
        # The wake-word check: its verdict (None until it rules; True for a run it does
        # not check), its task, score, latency in ms, and whether it timed out or
        # failed (no verdict, the fail policy decided).
        self.verdict: bool | None = None
        self.verify = None
        self.score: float | None = None
        self.verify_ms = 0
        self.error = False
        # The cross-device arbitration: the token the run registered under, whether it
        # still waits for the verdict, and whether it lost.
        self.token = 0
        self.arbitrating = False
        self.lost = False
        # The device was shown `listening` for this run.
        self.taken = taken
        # The VAD's end of the utterance seen while the run was held; its finalize
        # waits for the hold to lift.
        self.endpoint: str | None = None
        # Claimed for its _run (or ended): later audio is dropped.
        self.finalized = False
        # The running utterance peak driving the decision-only VAD boost
        # (core.vad.mic_auto_gain).
        self.vad_peak = 0
        # Logging only: "receiving audio" once per run; the missing second channel
        # warned once per run.
        self.audio_logged = False
        self.mic_fallback_logged = False

    def claim(self) -> tuple[bytes, bytes] | None:
        """Atomically claim the run for its finalization and snapshot its audio.

        No await, so concurrent eager on_audio calls can't double-claim or lose audio.
        Returns (pcm, pcm2) — the selected-channel PCM to process and the other
        channel's, kept for the stored stereo WAV (empty when the device streamed a
        single channel) — or None when the run is already finalized."""
        if self.finalized:
            return None
        self.finalized = True
        pcm, pcm2 = bytes(self.buffer), bytes(self.buffer2)
        self.buffer.clear()
        self.buffer2.clear()
        return pcm, pcm2

    @property
    def preroll_pcm(self) -> bytes:
        """The pre-roll received so far (the selected mic channel)."""
        return bytes(self.buffer[:PREROLL_BYTES])
