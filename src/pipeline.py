"""Per-device voice pipeline: mic PCM -> recognition chains (STT -> LLM) -> TTS -> device playback.

One Pipeline instance per device name. It owns its lock and the run it serves (a Run,
src/voice_run.py: the run's audio buffer and all its other state); the brain and TTS
backend are stateless shared dependencies, and the device server injects the
transport hooks (send_event, play, has_speaker, has_follow_up).

Server-side VAD end-pointing
----------------------------
The device fires its wake word on-device and then streams mic PCM to us
continuously (silence included). It has no end-pointing of its own
(docs/device-protocol.md); the device's `stop`, the stop word, cancels the run through
cancel_run. To know when the user has
finished speaking we run the configured VAD stage backend over the incoming PCM
(one VadSession per run, see src/vad.py) and finalize the utterance once it
reports an endpoint (speech followed by enough trailing silence), a hard max
length, or a no-speech timeout. Every finalize funnels through a synchronous
Run.claim() (which atomically marks the run finalized and snapshots the audio)
followed by the async _run(), so a run is finalized exactly once even under
concurrent eager on_audio tasks.
"""

import asyncio
import os
import re
import time

import numpy as np
from loguru import logger

from src import chains, config_store, context, llm, telemetry
from src.audio_prep import (
    highpass,
    normalize_peak,
    pcm_to_wav_bytes,
    trim_start_pcm,
    vad_boost,
)
from src.pipeline_events import StageEvent
from src.llm_text import clean_llm_output
from src.prompt import build_system_prompt
from src.runs_store import live_row, new_record, summary_row
from src.followup import NoFollowup
from src.speech import SpeechMixin
from src.voice_run import PREROLL_BYTES, PREROLL_MS, Run
from src.wakeword import PassthroughVerifier
# SAMPLE_RATE now lives in the VAD stage contract module; re-exported here so the
# many existing `from src.pipeline import SAMPLE_RATE` users keep working.
from src.vad import SAMPLE_RATE, EndpointPolicy  # noqa: F401  (SAMPLE_RATE re-export)

# Generous hard memory cap (~60s of 16 kHz / 16-bit mono PCM). VAD should end the
# utterance long before this; the cap only guards against unbounded growth if VAD
# misbehaves, and also forces a finalize when reached.
HARD_CAP_BYTES = SAMPLE_RATE * 2 * 60

# The word the speaker's own stop model listens for while an answer plays. That model is
# off in the follow-up window and after a wake, so there the word reaches the server as a
# phrase; said alone (or right after the wake word) it ends the session the same way.
STOP_WORD = "стоп"


def _words(text: str) -> list[str]:
    """The lowercased words of `text`, punctuation dropped."""
    return re.findall(r"[a-zа-яё]+", text.lower())


def is_stop_phrase(text: str, wake_word: str | None) -> bool:
    """True when `text` is the stop word alone, optionally preceded by `wake_word`."""
    words = _words(text)
    name = _words(wake_word or "")
    if name and words[:len(name)] == name:
        words = words[len(name):]
    return words == [STOP_WORD]


def has_wake_word(text: str, wake_word: str) -> bool:
    """True when one of the first three words of `text` starts with `wake_word`: the
    STT often fuses the wake word with the next word («захаркоторый час»), and the
    phrase may open with a word or two the pre-roll caught before it."""
    name = wake_word.lower()
    return any(word.startswith(name) for word in _words(text)[:3])


class Pipeline(SpeechMixin):
    """Drives one voice run for a single speaker."""

    def __init__(self, name, runtime):
        self.name = name
        # The mutable Runtime holder: all config and backends are read THROUGH it
        # per request (no frozen copies), so reconfiguration takes effect live.
        self.rt = runtime
        self._lock = asyncio.Lock()

        # A wake is held as a proposal until the wake-word check and the cross-device
        # arbitration have ruled on it (on_wake); taken (_take), it becomes the current
        # run: the one the device's session belongs to — a wake's or a follow-up
        # window's, None before the first taken wake. The mic audio goes to the
        # proposal while there is one, else to the current run.
        self._proposed: Run | None = None
        self._current: Run | None = None
        # The token is bumped for every run registered with the cross-device wake
        # arbitration (src/wake_arbiter.py), so a verdict meant for an older run of
        # this device is told apart and ignored.
        self._arbitration_token = 0

        # The finalized run (STT -> LLM -> TTS) started from on_audio; a taken wake
        # cancels it (see _take).
        self._run_task: asyncio.Task | None = None
        # The runs' wake-word checks (_check), the run log records of runs that ended
        # held (_record_held) and the closing of a replaced run's STT sessions, kept
        # referenced until they finish.
        self._background: set[asyncio.Task] = set()

        # Transport hooks, injected by the device server: send_event(event, data)
        # (sync), play(mime, bytes | async chunks) (returns {"outcome", "dry",
        # "gap_ms"} once the device reports the stream, see Connection.play),
        # has_speaker (a device without one gets no TTS at all) and has_follow_up
        # (the device reopens its microphone on a follow-up `listening`). arbiter is
        # the cross-device WakeArbiter
        # (src/wake_arbiter.py) a wake run registers with and reports to — its
        # pre-roll is in, its verifier ruled, it stopped waiting (sync calls); None
        # runs every wake unarbitrated. take_session(cid) and end_wake(cid) (sync): the
        # run of the wake that proposed session `cid` is taken (the session starts, the
        # playback is dropped) or ends without being taken (the device is refused that
        # number; the running session goes on).
        self.send_event = None
        self.play = None
        self.has_speaker = False
        self.has_follow_up = False
        self.arbiter = None
        self.take_session = lambda cid: None
        self.end_wake = lambda cid: None

        self._init_speech()

        # Tail of the chained background tasks that broadcast live (in-progress) run
        # snapshots. Scheduling these off the run coroutine keeps a slow WS client from
        # backpressuring `self._lock`; `_run`'s outer finally awaits this tail (outside
        # the lock) so the finalized broadcast never overtakes a partial. None = idle.
        self._live_send_tail = None

    # Read-through convenience properties: external callers (the device server,
    # tests) and the run logic below all reach config/backends THROUGH the runtime,
    # so a reconfiguration (live field change or backend swap) takes effect without
    # rebuilding the pipeline.
    @staticmethod
    def _backend_desc(backend) -> str:
        """Human-readable 'provider/model' identity stamped by ConfigService.create();
        falls back to the class name for unstamped backends (tests/fakes)."""
        return getattr(backend, "backend_desc", None) or type(backend).__name__

    @property
    def core(self):
        return self.rt.core

    @property
    def wakeword_backend(self):
        return self.rt.wakeword_backend

    @property
    def stress_backend(self):
        return self.rt.stress_backend

    @property
    def hub(self):
        return self.rt.hub

    @property
    def runs_store(self):
        return self.rt.runs_store

    @property
    def prompt_store(self):
        return self.rt.prompt_store

    @property
    def run_events(self):
        return self.rt.run_events

    @property
    def _context_path(self):
        # Data dir is hardcoded in config_store; accessed as a module attribute (not a
        # bound copy) so tests can monkeypatch config_store.DATA_DIR.
        return os.path.join(config_store.DATA_DIR, f"context_{self.name}.txt")

    def _emit(self, event_type, data=None):
        """Emit a transport-neutral StageEvent with a flat dict[str, str] payload. The
        arbiter sees it too: it forwards a kept phrase's answer to the speakerless
        devices that lost that phrase."""
        payload = {str(k): str(v) for k, v in (data or {}).items()}
        if self.send_event is not None:
            self.send_event(event_type, payload)
        if self.arbiter is not None:
            self.arbiter.run_event(self, event_type, payload)

    def _spawn(self, coro) -> asyncio.Task:
        """Start `coro` as a background task kept referenced until it finishes."""
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def gates_wake(self, wake_word_phrase) -> bool:
        """Whether a wake with this phrase goes through the server-side wake-word gate:
        only a REAL wake-word run (non-empty phrase), and only when a real verifier is
        loaded — a None backend (warm-up not done) or a PassthroughVerifier (stage
        disabled) skips the gate entirely."""
        wb = self.rt.wakeword_backend
        return (
            bool(wake_word_phrase)
            and wb is not None
            and not isinstance(wb, PassthroughVerifier)
        )

    def on_wake(self, cid, word, over_answer: bool = False) -> None:
        """A wake of the device: propose its run (docs/device-protocol.md, "Voice session").

        Synchronous. The run registers with the cross-device arbitration at once. A
        wake over the device's own answer or open session (`over_answer`, which the
        check is told) may be the device's own voice: it is held — its audio buffered,
        nothing shown on the device, the current run going on untouched — until the
        quick wake-word check confirms it, when it is taken (_take), or rejects it, when
        it ends held (_end_held). Any other wake is taken at once and confirmed later by
        the wake word in its recognised phrase (_run). A taken run is released (_release)
        once the arbitration lets it through; one that loses ends held. A newer wake
        replaces a proposal silently."""
        if self._proposed is not None:
            self._drop(self._proposed)
        run = self._proposed = self._new_run(
            cid, word or "", preroll_ms=PREROLL_MS if word else 0,
            no_speech_timeout_ms=self.core.vad.no_speech_timeout_ms, over_answer=over_answer)
        if over_answer and self.gates_wake(word):
            logger.info(f"{self.name}: 🛡️ wake (cid={cid}) over the running answer waits "
                        f"for the wake-word check")
        else:
            # Not checked here (not over an answer, check disabled/None/passthrough, no
            # wake word): its verdict is in.
            run.verdict = True
        if word and self.arbiter is not None:
            self._arbitration_token += 1
            run.token = self._arbitration_token
            run.arbitrating = True
            # The arbiter may rule inside the call, on this run too: _advance below
            # finds a run that won (and is taken now) or lost.
            self.arbiter.register(self, run.token)
        self._advance(run)

    def _new_run(self, cid, word: str, *, preroll_ms: int, no_speech_timeout_ms: int,
                 **fields) -> Run:
        """A new run with its VAD session: a wake's (on_wake) and a follow-up window's
        (_arm_followup) start the same way. `fields` go to the Run."""
        # Open the per-run VAD session. The end-pointing POLICY (generic thresholds)
        # is read live off core.vad here, while the speech classifier itself is the
        # swappable vad stage backend. One session per run, so a hot-swapped backend
        # or a changed policy applies on the next run.
        vad_cfg = self.core.vad
        # Capture the identity from the SAME backend ref used to open the session,
        # so a hot-swap mid-run can't mislabel the finalize log line.
        vad_backend = self.rt.vad_backend
        vad = vad_backend.open(EndpointPolicy(
            silence_ms=vad_cfg.silence_ms,
            min_speech_ms=vad_cfg.min_speech_ms,
            max_utterance_ms=vad_cfg.max_utterance_ms,
            no_speech_timeout_ms=no_speech_timeout_ms,
            preroll_ms=preroll_ms,
        ))
        return Run(cid, word, vad, self._backend_desc(vad_backend), **fields)

    def _drop(self, run: Run) -> None:
        """Forget a proposal without a trace (a newer wake replaced it, its device
        vanished): its check is cancelled and it leaves the arbitration."""
        run.finalized = True
        if run.verify is not None:
            run.verify.cancel()
        self._leave_arbitration(run)

    def _advance(self, run: Run) -> None:
        """Move a wake's run on after anything it waits for changed: its check ruled,
        the arbiter ruled, its utterance ended while held.

        Synchronous, and called from inside the check's task and the arbiter's calls,
        so it never awaits and starts a run as its own task. Rejected: ended
        (_end_held). Lost: held until its utterance ends, then ended as a duplicate.
        Confirmed and not ruled out: taken (_take) — a run its own confirmation makes
        lose (the arbiter rules inside _check's run_changed) is never shown
        `listening`. Taken and let through by the arbitration: released (_release).
        """
        if run.finalized:
            return
        if run.verdict is False:
            self._end_held(run, "wakeword_error" if run.error else "wakeword_reject")
        elif run.lost:
            if run.endpoint is not None:
                self._end_held(run, "duplicate")
        elif run.verdict:
            if not run.taken:
                self._take(run)
            if not run.arbitrating and not run.released:
                self._release(run)

    def _take(self, run: Run) -> None:
        """Take a confirmed wake: it becomes the current run and its session starts.

        Fully synchronous, so nothing of the older run can run in between: the older
        run is cancelled (cancel_run; never the proposal's check, which may be the
        task this runs in), the device server starts the wake's session
        (take_session: `play_stop` before anything else of it) and the device is shown
        `listening` at once — the arbitration may hold the run for seconds more, and
        the device shows nothing of this wake until `listening`. A taken run that later loses keeps
        it (docs/device-protocol.md, the decisions)."""
        run.taken = True
        self._proposed = None
        old = self._current
        if self._run_task is not None and not self._run_task.done():
            logger.info(f"{self.name}: new wake cancels the running run")
        # Also ends a follow-up window the previous answer opened.
        self.cancel_run()
        self._current = run
        # The streaming STT sessions of a claimed older run whose cancelled run task
        # never started (cancel_run closes those of a run never claimed); a running
        # _run owns (and closes) its own.
        if old is not None and old.stt_streams:
            streams, old.stt_streams = old.stt_streams, {}
            self._spawn(self._close_stt_streams(streams))
        self.take_session(run.cid)
        logger.info(f"{self.name}: ▶️ run started (cid={run.cid})")
        self._emit(StageEvent.RUN_START, {})
        self._emit(StageEvent.STT_START, {})
        # Speech is already in-stream (pre-roll captured the command); the matching
        # STT_VAD_END (thinking) is emitted in _run once VAD finalizes the utterance.
        self._emit(StageEvent.STT_VAD_START, {})

    def _release(self, run: Run) -> None:
        """Let a taken run's audio through to STT, once per run: the arbitration let it
        through, or it has none (a follow-up window).

        Synchronously opens its STT streams and feeds each the audio held so far once
        — no await in between, so a concurrent on_audio can't double-feed or lose a
        chunk. If the utterance already ended while held (Run.endpoint), the run is
        claimed here as well and its _run started as the run task."""
        run.released = True
        self._open_stt_streams(run)
        # Feed the whole backlog buffered during the hold once, so streaming STT sees
        # the same audio the batch path would.
        if run.buffer:
            backlog = bytes(run.buffer)
            for stream in run.stt_streams.values():
                stream.feed(backlog)
        if run.endpoint is not None:
            pcm, pcm2 = run.claim()
            self._run_task = asyncio.create_task(self._run(run, run.endpoint, pcm, pcm2=pcm2))

    def _end_held(self, run: Run, reason: str) -> None:
        """End a held run that will not be released: its check rejected it
        (wakeword_reject, or wakeword_error when the check timed out or failed and its
        policy rejects), or it lost the arbitration and its utterance ended (duplicate).

        Synchronous: the run is claimed and leaves the arbitration. A taken one ends
        its own session (an empty STT_END and RUN_END, the device's `state idle`). One
        never taken sends nothing through send_event — its RUN_END would end the
        session of the answer the current run may be playing — and the device server
        ends its wake instead (end_wake). The run log record follows as its own task
        (_record_held), outside the run lock, which an answering run holds for its
        whole playback."""
        pcm, pcm2 = run.claim()
        self._leave_arbitration(run)
        if reason == "duplicate":
            logger.info(f"{self.name}: 🚫 wake dropped as a duplicate of another device's wake")
        else:
            logger.info(f"{self.name}: 🚫 wake word rejected by verifier "
                        f"(reason={reason}, score={run.score})")
        if run.taken:
            self._emit(StageEvent.STT_END, {"text": ""})
            self._emit(StageEvent.RUN_END, {})
        else:
            self._proposed = None
            self.end_wake(run.cid)
        self._spawn(self._record_held(run, reason, pcm, pcm2))

    def _arm_followup(self, answered: Run, request: str, reply: str) -> None:
        """Keep the session open after its answer played out and listen again without
        a wake word (docs/device-protocol.md, "Follow-up"). A new run, taken and
        released at once: it cancels nothing (it is armed inside the answering run),
        keeps the session wake's word for the follow-up check, has no pre-roll and no
        wake-word check, and the VAD ends it with no_speech when speech does not start
        within the stage's window."""
        window_ms = self.rt.followup_backend.window_ms
        run = self._current = self._new_run(
            answered.cid, answered.word, preroll_ms=0, no_speech_timeout_ms=window_ms,
            followup=(answered.word, request, reply), taken=True)
        run.verdict = True
        logger.info(f"{self.name}: 👂 listening for a follow-up ({window_ms} ms)")
        self._emit(StageEvent.STT_VAD_START, {})
        self._release(run)

    def cancel_run(self) -> None:
        """Cancel the current run: its run task and its ack/filler plays. A run still
        listening (a phrase, a follow-up window) is abandoned: audio that still
        arrives is dropped, so it can never end its phrase and answer later, and its
        streaming STT sessions are closed. A proposal is not touched (its check may be
        what is taking it). Synchronous; the cancelled run emits no RUN_END (its stages
        catch Exception, which a cancellation is not). The run stops waiting for the
        arbitration, so the arbiter skips it."""
        # Cancellation of an already-finished task is a no-op. Not awaited: each
        # cancelled task unwinds on its own (the ack/filler sets self-clean via their
        # done-callbacks); snapshot the sets first since those callbacks mutate them.
        for task in (self._run_task, *self._ack_tasks, *self._filler_tasks):
            if task is not None:
                task.cancel()
        self._run_task = None
        run = self._current
        if run is None:
            return
        if not run.finalized:
            # A claimed run is finalized already, and its _run closes its own streams.
            run.finalized = True
            if run.stt_streams:
                streams, run.stt_streams = run.stt_streams, {}
                self._spawn(self._close_stt_streams(streams))
        self._leave_arbitration(run)

    def _leave_arbitration(self, run: Run) -> None:
        """Stop waiting for the arbitration verdict (the run was cancelled, ended held,
        or its device vanished): the arbiter re-checks the group without this run."""
        if run.arbitrating:
            run.arbitrating = False
            self.arbiter.run_changed(self, run.token)

    def _open_stt_streams(self, run: Run) -> None:
        """Open the live streaming STT sessions for this run (synchronous): one per
        chain whose STT starts at speech (chains.starts_at_speech), so on_audio can
        feed them chunks while speech is still arriving (VAD still owns end-pointing;
        on finalize the chain runner calls finish()).

        A batch-only backend (no open_stream(), or one returning None) gets no
        session — NOT an error — and is transcribed in one shot at VAD end; only a
        genuine open failure warns and falls back to that batch route. Called by
        _release only, for a wake's run and a follow-up window alike.
        """
        run.stt_streams = {}
        for chain in chains.starts_at_speech(self.rt, self.core.chains):
            try:
                stream = chains.open_stream(self.rt, chain)
            except Exception as e:  # noqa: BLE001 - never let STT setup break the run; batch-fallback
                logger.warning(
                    f"{self.name}: streaming STT open failed for chain {chain.name!r}, "
                    f"using batch: {e}"
                )
                continue
            if stream is not None:
                run.stt_streams[chain.name] = stream

    async def on_audio(self, data: bytes, data2=None) -> None:
        """Accumulate mic PCM and run VAD end-pointing on the configured channel.

        The Voice PE streams two mic channels: channel 0 (`data`) is the more-
        processed XMOS AGC output, channel 1 (`data2`) the less-processed XMOS
        noise-suppression output (cleaner but quieter). Only the channel is selected
        HERE, per chunk, by `core.vad.mic_channel` (read live off `self.core`, so panel
        changes apply on the next utterance — no restart). The optional high-pass +
        peak-normalization conditioning (`core.vad.mic_highpass` / `mic_normalize`) is
        applied LATER, once to the WHOLE utterance, in `_run` before STT — never per
        chunk (per-chunk normalization would destroy dynamics and amplify silence).
        The NON-selected channel is buffered separately (Run.buffer2) only so the
        stored diagnostic WAV can carry both channels in stereo — it never reaches
        VAD or STT.

        The device streams continuously and never signals end-of-speech, so we
        detect it here: VAD over the PCM finalizes the utterance once speech is
        followed by enough trailing silence (or a max length / no-speech timeout).

        The audio goes to the proposed wake while there is one (on_wake), else to the
        current run.

        Never suspends: the whole update is synchronous and a finalized run is
        started as its own task (_run_task), so the device server's read loop can
        await this per frame without ever waiting on a run.
        """
        run = self._proposed or self._current
        if run is None or run.finalized:
            return  # No run, or late audio for an already-finalized one.

        # Select the mic channel for the WHOLE pipeline (VAD + STT) from
        # config (0 = processed, 1 = less-processed/quieter). Channel 1 uses the
        # device's second stream when present; if it's missing, fall back to channel 0
        # and warn once per run. Read live off self.core, so panel changes take effect
        # on the next utterance. Any high-pass / normalize conditioning happens later,
        # once on the whole utterance in _run (never per chunk).
        other = None  # the non-selected channel, kept only for the stored diagnostic WAV
        if self.core.vad.mic_channel == 1:
            if data2:
                data, other = data2, data
            elif not run.mic_fallback_logged:
                run.mic_fallback_logged = True
                logger.debug(
                    f"{self.name}: mic_channel=1 but device sent no second channel; "
                    f"using channel 0"
                )
        else:
            other = data2  # may be None when the device streams a single channel

        # Held (not released): the wake-word check or the cross-device arbitration
        # hasn't ruled yet, or the run lost the arbitration. Keep accumulating the
        # buffer (channel-select above picks Run.buffer/buffer2) but DON'T feed the STT
        # streams (there are none yet) and DON'T finalize the utterance — if VAD
        # end-points, stash the reason and stop feeding VAD; the finalize is deferred
        # until the release replays it, and a lost run ends at it (_advance). While the
        # check has yet to rule, start it EXACTLY once, when enough pre-roll is
        # buffered (window_bytes) or the utterance already ended. While the
        # arbitration is pending, tell the arbiter once when the whole pre-roll is in.
        # This branch is fully SYNCHRONOUS (every await lives inside the check's task)
        # so the eager-on_audio atomicity invariant holds.
        if not run.released:
            if not run.audio_logged:
                run.audio_logged = True
                logger.info(f"{self.name}: 🎤 receiving audio (held)...")
            buffered = len(run.buffer)
            run.buffer.extend(data)
            if other:
                run.buffer2.extend(other)
            elif run.buffer2:
                run.buffer2.extend(b"\x00" * len(data))
            if len(run.buffer2) >= HARD_CAP_BYTES:
                del run.buffer2[HARD_CAP_BYTES:]
            if len(run.buffer) >= HARD_CAP_BYTES:
                del run.buffer[HARD_CAP_BYTES:]
            # Track the end-point while held, but DON'T finalize here: stash the
            # reason and stop feeding VAD (the utterance end is replayed on release).
            if run.endpoint is None and run.vad is not None:
                run.endpoint = run.vad.feed(data)
                if run.endpoint is not None:
                    self._advance(run)
            wb = self.rt.wakeword_backend
            if run.verdict is None and run.verify is None and not run.finalized and wb is not None:
                window_bytes = wb.window_ms * SAMPLE_RATE * 2 // 1000
                if len(run.buffer) >= window_bytes or run.endpoint is not None:
                    # Snapshot the pre-roll head SYNCHRONOUSLY; the task then awaits.
                    run.verify = self._spawn(self._check(run, bytes(run.buffer[:window_bytes])))
            # Last: the buffer just reached the whole pre-roll. The arbiter may rule
            # inside the call, on this run too, so nothing follows it.
            if run.arbitrating and buffered < PREROLL_BYTES <= len(run.buffer):
                self.arbiter.preroll_in(self, run.token)
            return

        # Log once per run on the first chunk; the speaker streams many chunks.
        if not run.audio_logged:
            run.audio_logged = True
            logger.info(f"{self.name}: 🎤 receiving audio...")

        # The full utterance audio (everything streamed, silence included) is what
        # we send to STT.
        run.buffer.extend(data)
        # Feed the live streaming STT sessions (if any) the same selected-channel bytes
        # that go into the buffer. feed() is synchronous and never raises.
        for stream in run.stt_streams.values():
            stream.feed(data)
        if other:
            run.buffer2.extend(other)
        elif run.buffer2:
            # Keep channels time-aligned if the second stream drops out mid-run.
            run.buffer2.extend(b"\x00" * len(data))
        if len(run.buffer2) >= HARD_CAP_BYTES:
            del run.buffer2[HARD_CAP_BYTES:]
        reason = None
        if len(run.buffer) >= HARD_CAP_BYTES:
            del run.buffer[HARD_CAP_BYTES:]
            reason = "maxlen"
        elif run.vad is not None:
            # Feed the SELECTED channel — the same bytes that went into the buffer —
            # to the per-run VAD session. The session does its own framing/counters
            # and returns a finalize reason ("endpoint" | "maxlen" | "no_speech") or
            # None to keep listening. feed() is SYNCHRONOUS, so the claim logic's
            # atomicity guarantees hold (no await before the claim below). When
            # HARD_CAP already set reason="maxlen" above, the session is NOT fed: we're
            # about to finalize this chunk anyway, and the buffer was truncated, so
            # there's no point feeding (possibly truncated) bytes into the VAD counters.
            vdata = data
            if self.core.vad.mic_auto_gain:
                # Track the running utterance peak and lift the chunk for the
                # speech/no-speech decision only — the buffer/STT path above keeps
                # the raw bytes (per-utterance normalization happens pre-STT in _run).
                m = len(data) - (len(data) % 2)
                if m:
                    run.vad_peak = max(
                        run.vad_peak,
                        int(np.abs(np.frombuffer(data[:m], dtype="<i2")).max()),
                    )
                vdata = vad_boost(data, run.vad_peak)
            reason = run.vad.feed(vdata)

        if reason is not None:
            state = run.vad.debug_state() if run.vad else {}
            logger.debug(
                f"{self.name}: VAD finalize reason={reason} peak={run.vad_peak} "
                + " ".join(f"{k}={v}" for k, v in state.items())
            )

        # All synchronous state is updated above; claim last. Run.claim() is atomic (no
        # await), so a run is finalized exactly once and no audio is lost.
        if reason is not None:
            claimed = run.claim()
            if claimed is not None:
                self._run_task = asyncio.create_task(
                    self._run(run, reason, claimed[0], pcm2=claimed[1]))

    async def verify_wake(self, pcm: bytes, over_answer: bool = False
                          ) -> tuple[bool, float | None, int, bool]:
        """Run the second-stage wake-word verifier on a pre-roll head: (accepted,
        score, t_ms, error). verify() never raises, but we still bound it with the
        backend's timeout and treat a timeout/None as the configured fail policy
        (fail_open accepts, fail_closed rejects); `error` tells that case (no verdict,
        score None) from a genuine verdict. `over_answer`: a wake held over the
        device's own answer (WakewordVerifier.verify)."""
        t0 = time.perf_counter()
        wb = self.rt.wakeword_backend
        try:
            verdict = await asyncio.wait_for(wb.verify(pcm, over_answer=over_answer),
                                             timeout=wb.timeout_ms / 1000)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - timeout/error -> fail policy
            verdict = None
        t_ms = int((time.perf_counter() - t0) * 1000)
        score = verdict.score if verdict is not None else None
        # A timeout / internal error has no verdict -> apply the fail policy.
        accepted = verdict.accepted if verdict is not None else wb.fail_open
        if verdict is None:
            logger.warning(
                f"{self.name}: 🛡️ wake-word verify timed out/errored "
                f"(fail_open={wb.fail_open}) -> {'accept' if accepted else 'reject'} "
                f"({t_ms} ms)"
            )
        elif accepted:
            logger.info(f"{self.name}: ✅ wake word confirmed (score={score}, {t_ms} ms)")
        else:
            logger.info(f"{self.name}: 🚫 wake word rejected (score={score}, {t_ms} ms)")
        return accepted, score, t_ms, verdict is None

    async def _check(self, run: Run, pcm: bytes) -> None:
        """The wake-word check of a held run on its pre-roll head, then the run moves on
        (_advance).

        Spawned exactly once per checked run (from on_audio). All the awaits live here,
        so the on_audio branch that schedules this stays synchronous and the
        eager-on_audio atomicity invariant holds. A confirmation goes to the arbiter
        first — it may rule inside the call, on this run too — so the run is taken only
        if it did not lose by it (docs/device-protocol.md, the decisions)."""
        if self.rt.wakeword_backend is None:
            return  # backend vanished.
        verdict = await self.verify_wake(pcm, over_answer=run.over_answer)
        if run.finalized:
            return  # the run was dropped while we were verifying.
        run.verdict, run.score, run.verify_ms, run.error = verdict
        if run.verdict and run.arbitrating:
            self.arbiter.run_changed(self, run.token)
        self._advance(run)

    def _run_of(self, token: int) -> Run | None:
        """The run (the proposal or the current one) registered with the arbitration
        under `token`."""
        return next((r for r in (self._proposed, self._current)
                     if r is not None and r.token == token), None)

    @property
    def preroll_pcm(self) -> bytes:
        """The pre-roll of the device's latest wake (the proposal, else the current
        run) received so far (the selected mic channel), for the arbiter's
        signal-to-noise comparison."""
        run = self._proposed or self._current
        return run.preroll_pcm if run is not None else b""

    @property
    def verifier_pending(self) -> bool:
        """Whether the wake-word verifier has yet to rule on the device's latest wake;
        the arbiter waits for it, except that a run without a speaker loses as it is
        once a ready candidate has one."""
        run = self._proposed or self._current
        return run is not None and run.verdict is None

    def awaits_arbitration(self, token: int) -> bool:
        """Whether the run registered under `token` still waits for the arbitration
        verdict: it is still this device's proposal or current run, not finalized,
        still held for it."""
        run = self._run_of(token)
        return run is not None and run.arbitrating and not run.finalized

    def arbitration_verdict(self, token: int, won: bool) -> None:
        """The cross-device arbiter's verdict on the run registered under `token`.

        Synchronous, and possibly called inside a call this or another pipeline made
        to the arbiter, or from one of its timers — the module docstring of
        src/wake_arbiter.py lists them. A verdict the run no longer awaits is ignored.
        Lost: the run stays held — its device streams the rest of the phrase on,
        keeping the `listening` a taken run was already shown — until its VAD
        end-point, when it ends as a duplicate; a run not taken by then is never taken.
        Won: the run is released once its wake-word check confirms it (_advance).
        """
        if not self.awaits_arbitration(token):
            return
        run = self._run_of(token)
        run.arbitrating = False
        run.lost = not won
        self._advance(run)

    async def _record_held(self, run: Run, reason: str, pcm: bytes, pcm2: bytes) -> None:
        """The run log record of a run _end_held ended: `rejected`, with its reason, its
        check's latency and score and its stored audio, no ack chime and no capture
        file. A duplicate's phrase is recognized by the first enabled chain's STT for
        the run log only — the device never gets that transcript — unless the VAD found
        no speech in it: the STT would make up words for the noise."""
        t0 = time.perf_counter()
        pcm, pcm2 = self._condition_audio(pcm, pcm2, trim=True)
        record = self._new_record(reason, run, pcm, result="rejected")
        if reason == "duplicate" and run.endpoint != "no_speech":
            stt_t = time.perf_counter()
            try:
                first = chains.enabled_chains(self.core.chains)[0]
                record["stt_text"] = await chains.transcribe(self.rt, first, pcm)
            except Exception as e:  # noqa: BLE001 - the run is recorded without it
                logger.error(f"{self.name}: duplicate STT failed: {e}")
            record["t_stt"] = int((time.perf_counter() - stt_t) * 1000)
        await self._broadcast_run(await self._store_run(record, t0, pcm, pcm2))

    def _emit_live(self, record) -> None:
        """Schedule a best-effort broadcast of the current in-progress run snapshot.

        The snapshot is built SYNCHRONOUSLY here (so a later stage mutating `record`
        can't alter an already-scheduled message), then sent from a background task
        chained after this pipeline's previous live send. Scheduling instead of
        awaiting keeps a slow/stuck WS client from backpressuring the per-device
        `self._lock` that `_run` holds across the whole run — the same reason the
        finalized broadcast is deferred outside the lock. Chaining preserves per-stage
        delivery order; `_run`'s outer `finally` drains the chain (outside the lock)
        before the finalized broadcast so the final never overtakes a partial.
        """
        if self.run_events is None:
            return
        row = live_row(record)  # snapshot now; `record` keeps mutating across stages
        prev = self._live_send_tail

        async def _send():
            # Preserve order: wait for the previous live send first. `prev` swallows
            # its own errors, so awaiting it never raises here.
            if prev is not None:
                try:
                    await prev
                except Exception:  # noqa: BLE001 - defensive; _send never re-raises
                    pass
            try:
                await self.run_events.broadcast({"type": "run", "run": row})
            except Exception as e:  # noqa: BLE001 - live updates must never break a run
                logger.debug(f"{self.name}: live run update failed: {e}")

        self._live_send_tail = asyncio.create_task(_send())

    async def _close_stt_streams(self, streams: dict) -> None:
        """Close the given streaming STT sessions (keyed by chain name); a failure is
        logged, never raised."""
        for stream in streams.values():
            try:
                await stream.aclose()
            except Exception as e:  # noqa: BLE001 - cleanup must never break the run
                logger.debug(f"{self.name}: streaming STT close failed: {e}")

    def _llm_request(self, text: str) -> llm.LlmRequest:
        """A handler's input for `text`: the assembled system prompt (a SQLite read
        of the active profile), the context history and the originating device."""
        return llm.LlmRequest(
            system_prompt=build_system_prompt(self.core, self.prompt_store),
            history=context.load_context(
                self._context_path,
                max_turns=self.core.context.max_turns,
                ttl_seconds=self.core.context.ttl_seconds,
            ),
            user_text=text,
            device=self.name,
        )

    @staticmethod
    def _record_chain_run(record: dict, run) -> None:
        """Fill the run record from what the recognition chains produced."""
        record["chain"] = run.chain
        record["attempts"] = run.attempts
        record["t_stt"] = run.t_stt
        record["t_llm"] = run.t_llm
        record["stt_text"] = run.text
        if run.reply is not None:
            record["llm_text"] = run.reply
        if run.result is not None:
            record["model"] = run.result.model
            record["tokens"] = run.result.tokens
            record["rounds"] = run.result.rounds
            record["request"] = run.result.request_debug
        else:
            # A failed handler attaches what it accumulated before the failure.
            record["model"] = run.partial.get("model")
            record["tokens"] = run.partial.get("tokens")
            record["rounds"] = run.partial.get("rounds") or []
            record["request"] = run.partial.get("request")
        if run.answered:
            # Did the handler actually run any tool this run?
            record["result"] = "tool" if run.result.tool_used else "ok"
        elif run.error is not None:
            record["result"] = "error"
            record["error_stage"] = run.error.stage.upper()
            record["error_text"] = str(run.error)
        elif run.reply is not None:
            # Every handler that heard the request declined: its reply is the answer.
            record["result"] = "ok"

    async def on_disconnect(self) -> None:
        """Device dropped its connection. A proposed wake is forgotten without a
        record, its check cancelled so a late verdict can't act on a dead run. A current
        run that was never finalized (the speaker vanished mid-utterance) is abandoned:
        it takes no more audio, so frames after a later wake can't end its phrase and
        run the old command, and its streaming STT sessions are closed so the gRPC
        streams/channels don't linger. A FINALIZED run owns the sessions and closes them
        in _run's finally, so skip it here to avoid racing finish() against aclose()."""
        proposed, self._proposed = self._proposed, None
        if proposed is not None:
            self._drop(proposed)
        run = self._current
        if run is None:
            return
        # A vanished device no longer waits for the arbitration: the arbiter skips it.
        self._leave_arbitration(run)
        if not run.finalized:
            run.finalized = True
            streams, run.stt_streams = run.stt_streams, {}
            await self._close_stt_streams(streams)

    def _end_silently(self, record: dict, text: str, result: str, reason: str,
                      note: str) -> None:
        """End a run without a sound once its transcript is in: a stop phrase, a wake
        without the wake word in its transcript, a rejected follow-up phrase. No handler
        ran, no transcript goes to the device (an empty STT_END), nothing joins the
        context; the run log still shows what was heard (`text`) with `result` and
        `reason`. `note` is the log line."""
        record.update(stt_text=text, result=result, reason=reason)
        logger.info(f"{self.name}: {note}")
        self._emit(StageEvent.STT_END, {"text": ""})
        self._emit(StageEvent.RUN_END, {})

    async def _run(self, run: Run, reason, pcm, pcm2: bytes = b"") -> None:
        """Run the recognition chains -> TTS -> events on the already-claimed audio, once.

        The caller claims `run` via Run.claim() (which sets its finalized flag,
        snapshots and clears the buffer) and passes the snapshotted pcm here. The lock
        only provides defensive serialization; the claim already guarantees single
        entry. RUN_END is always sent, unless the played answer opens a follow-up
        window (_arm_followup). The run takes over the streaming STT sessions of the
        claimed utterance, so a cancelled run's cleanup never touches the next one.
        """
        stt_streams, run.stt_streams = run.stt_streams, {}
        conversation_id = run.cid
        # The last turn when this utterance was heard in a follow-up window, else None.
        followup = run.followup
        # A wake's phrase the quick check did not hold (on_wake): with the wake-word
        # check on, its wake is confirmed by the wake word in its transcript
        # (has_wake_word, in _gate below). The same switch as the quick check, so with
        # it off a wake on a model of another word (Okay Nabu, Hey Jarvis) still passes.
        text_checked = (followup is None and not run.over_answer
                        and self.gates_wake(run.word))
        pending_run = None
        try:
            async with self._lock:
                # Total-run timer (logging only); started after the claim/lock.
                t0 = time.perf_counter()
                logger.info(
                    f"{self.name}: 🎙️ captured {len(pcm)} bytes "
                    f"(~{len(pcm) / (SAMPLE_RATE * 2):.1f}s), reason={reason}, "
                    f"vad={run.vad_desc}"
                )
                if not pcm:
                    # Truly-empty audio: nothing to transcribe and nothing to record.
                    logger.info(f"{self.name}: empty audio, ending run")
                    await self._close_stt_streams(stt_streams)
                    self._emit(StageEvent.RUN_END, {})
                    return

                # End-of-phrase ack ("блям"): we've just finalized the utterance, so an
                # unchecked phrase gets a short confirmation chime NOW as immediate "got
                # it" feedback, before STT/LLM/TTS run. Fire-and-forget (scheduled, never
                # awaited) so it never delays STT; tracked in a set so it isn't GC'd
                # mid-playback. The ack, a filler and the reply are separate play()
                # streams, queued in call order on the device.
                #
                # A wake word with no command after it (no_speech) ends the run
                # silently — no ack chime — so a bare wake word never beeps (a wake the
                # quick check rejected never gets here: _end_held). A text-checked
                # wake's phrase gets its chime only once the wake word is found in its
                # transcript, a follow-up phrase only once the follow-up check accepts
                # it (_gate).
                if reason != "no_speech" and followup is None and not text_checked:
                    self._schedule_ack()

                # A follow-up phrase has no lead-in: its audio starts at the
                # `listening`, before the phrase. The non-empty guard above still holds
                # afterwards (trim_start_pcm never returns empty audio).
                pcm, pcm2 = self._condition_audio(pcm, pcm2, trim=followup is None)

                # Observability record for this run (_new_record). Accumulated across
                # all terminal paths below and inserted exactly once in the finally;
                # the rest of the timings fill in as stages complete.
                # The answer's stored speech (speak_reply), stored after the run is
                # inserted, mirroring the utterance-audio store. Stays None unless the
                # answer played, so a failed/empty TTS stores nothing.
                tts_audio = None
                tts_mime = None
                # False only for a follow-up window that heard no speech: no request,
                # so no run in the log.
                keep_record = True
                record = self._new_record(reason, run, pcm)
                try:
                    # VAD found no speech in the whole window — this is silence/noise,
                    # not an utterance. Skip STT entirely: Whisper hallucinates stray
                    # phrases on non-speech audio (e.g. "Продолжение следует..."), which
                    # would then pollute the run log or flow into the LLM/TTS. Balance the
                    # STT_START emitted in _take with an empty STT_END, end the run, and
                    # let the finally below record it as an empty run (result="empty",
                    # stt_text="", t_stt=0 — all left at their defaults).
                    if reason == "no_speech":
                        logger.info(
                            f"{self.name}: 😶 no speech detected by VAD, skipping STT"
                        )
                        # Discard any streaming sessions WITHOUT reading their results —
                        # we never want a transcript for silence.
                        await self._close_stt_streams(stt_streams)
                        stt_streams = {}
                        keep_record = followup is None
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit(StageEvent.RUN_END, {})
                        return
                    # Utterance ended and there IS speech to process: tell the device to
                    # enter the "thinking" LED phase now. It stays in thinking through STT
                    # and the LLM until TTS_START switches it to "replying" — this is the
                    # indicator shown to the user between their question and the answer.
                    self._emit(StageEvent.STT_VAD_END, {})
                    filler_fired = False  # at most one early filler per run

                    async def _speak_filler(text: str, tool_names: list[str]) -> None:
                        # Policy lives here (llm.py is policy-free): speak at most once per
                        # run, and only for a SLOW tool (so instant smart-home actions don't
                        # double-talk). The slow KNOWLEDGE comes from the tool source via
                        # hub.is_slow (each source declares whether its tools are slow).
                        # Schedule synthesis+play as a fire-and-forget task
                        # so the slow tool is NOT delayed by filler TTS. Synchronous
                        # gate/dedup first (no await before we decide), then schedule.
                        nonlocal filler_fired
                        if filler_fired or not self.has_speaker:
                            return
                        if not any(self.hub.is_slow(n) for n in tool_names):
                            return  # fast action: the final reply alone is enough
                        spoken = clean_llm_output(text)  # tag cleanup only; engine post-processing happens inside the TTS backend
                        if not spoken:
                            return
                        filler_fired = True
                        record["filler_text"] = spoken
                        record["t_filler"] = int((time.perf_counter() - t0) * 1000)
                        self._schedule_filler(spoken)

                    def _on_text(text: str, t_stt: int) -> None:
                        # The first non-empty transcript of any chain: shown on the device
                        # (STT_END) and in the panel (live STT partial), then the device
                        # enters the intent phase. Fires at most once per run.
                        record["stt_text"] = text
                        record["t_stt"] = t_stt
                        self._emit(StageEvent.STT_END, {"text": text})
                        self._emit_live(record)
                        self._emit(StageEvent.INTENT_START, {})

                    # A phrase heard in a follow-up window: the follow-up check of its
                    # first transcript, asked before anything reaches the device. An
                    # accepted phrase gets the end-of-phrase chime now and runs like any
                    # phrase; a rejected one ends the run silently (below). The check
                    # lands in the run's attempts.
                    followup_check = None

                    async def _followup_gate(text: str) -> bool:
                        nonlocal followup_check
                        gate = self.rt.followup_backend
                        t = time.perf_counter()
                        verdict = await gate.check(*followup, text)
                        if verdict.accepted:
                            outcome = "accepted"
                            self._schedule_ack()
                        else:
                            outcome = "rejected" if verdict.probability is not None else "error"
                        followup_check = {
                            "kind": "followup", "provider": self._backend_desc(gate),
                            "text": text, "outcome": outcome, "probability": verdict.probability,
                            "t": int((time.perf_counter() - t) * 1000),
                        }
                        return verdict.accepted

                    # The stop word as the whole phrase (is_stop_phrase): no follow-up
                    # check, no handler — the run ends silently below.
                    stop_text = None
                    # A text-checked wake's transcript without the wake word: the wake
                    # was false, and the run ends silently below. With it, the phrase
                    # gets the end-of-phrase chime now and runs like any phrase.
                    wake_reject_text = None

                    async def _gate(text: str) -> bool:
                        nonlocal stop_text, wake_reject_text
                        if is_stop_phrase(text, run.word):
                            stop_text = text
                            return False
                        if text_checked:
                            if not has_wake_word(text, run.word):
                                wake_reject_text = text
                                return False
                            self._schedule_ack()
                        return followup is None or await _followup_gate(text)

                    # The recognition chains: each enabled chain's STT then handler, in
                    # order, until one answers. Streaming sessions opened at speech were
                    # fed in on_audio; the runner finishes them. A handler failure is
                    # mapped to a configured spoken phrase — the raw API error text never
                    # reaches the user's ears.
                    chain_run = await chains.run(
                        self.rt,
                        device=self.name,
                        request=self._llm_request,
                        pcm=pcm,
                        streams=dict(stt_streams),
                        on_text=_on_text,
                        on_filler=_speak_filler,
                        gate=_gate,
                    )
                    self._record_chain_run(record, chain_run)
                    if stop_text is not None:
                        self._end_silently(record, stop_text, "ok", "stop",
                                           f"🛑 stop phrase {stop_text!r}, session ended")
                        return
                    if wake_reject_text is not None:
                        # A false wake: dropped without a sound, like a rejected follow-up.
                        self._end_silently(record, wake_reject_text, "rejected",
                                           "wakeword_reject",
                                           f"🚫 no wake word in {wake_reject_text!r}, "
                                           f"wake dropped")
                        return
                    if followup_check is not None:
                        record["attempts"].insert(0, followup_check)
                        if followup_check["outcome"] != "accepted":
                            # Not a continuation of the dialog, or the check failed: the
                            # phrase is dropped without a sound.
                            drop_reason = ("followup_reject"
                                           if followup_check["outcome"] == "rejected"
                                           else "followup_error")
                            p = followup_check["probability"]
                            self._end_silently(
                                record, followup_check["text"], "rejected", drop_reason,
                                f"🚫 follow-up dropped ({drop_reason}, "
                                f"probability={'—' if p is None else f'{p:.0%}'})")
                            return
                    if chain_run.reply is None:
                        # No chain heard anything (or every STT failed): nothing is spoken.
                        # Balance the STT_START emitted in _take with an empty STT_END
                        # and end the run; the finally records it ("empty" or an STT error).
                        logger.info(f"{self.name}: no chain heard a request, ending run")
                        self._emit(StageEvent.STT_END, {"text": ""})
                        self._emit_live(record)
                        self._emit(StageEvent.RUN_END, {})
                        return
                    reply = chain_run.reply
                    # Live LLM partial: record now carries llm_text, model, tokens,
                    # rounds, t_llm, the chain attempts and the updated result.
                    self._emit_live(record)

                    # Only an answered turn is a conversation turn: don't pollute the
                    # context history with a spoken fallback or decline phrase.
                    if chain_run.answered:
                        try:
                            context.append_context(
                                self._context_path,
                                chain_run.text,
                                reply,
                                max_turns=self.core.context.max_turns,
                                ttl_seconds=self.core.context.ttl_seconds,
                            )
                        except Exception as e:
                            # Context persistence failure must not break the run.
                            logger.error(f"{self.name}: context append failed: {e}")

                    self._emit(StageEvent.INTENT_END, {"conversation_id": conversation_id})

                    played = None  # what the play hook returned for the answer stream
                    try:
                        played, tts_audio, tts_mime = await self.speak_reply(
                            reply, record, answer=True)
                    except Exception:  # noqa: BLE001 - marked on the record and logged by speak_reply
                        pass  # the run still ends cleanly

                    logger.info(
                        f"{self.name}: ✅ run complete in {time.perf_counter() - t0:.2f}s"
                    )
                    # The answer played out (not dropped by a new wake or play_stop) on
                    # a device that reopens its microphone, with the stage on and no
                    # proactive speech queued behind it: the session stays open for a
                    # follow-up phrase instead of ending.
                    fb = self.rt.followup_backend
                    if (played is not None and played["outcome"] == "played"
                            and self.has_follow_up and not self._speaking
                            and fb is not None and not isinstance(fb, NoFollowup)):
                        self._arm_followup(run, chain_run.text, chain_run.reply)
                    else:
                        self._emit(StageEvent.RUN_END, {})
                except Exception as e:
                    record["result"] = "error"
                    record["error_stage"] = "pipeline"
                    record["error_text"] = str(e)
                    logger.exception(f"{self.name}: pipeline run failed: {e}")
                    self._emit(
                        StageEvent.ERROR,
                        {"code": "server_error", "message": str(e)},
                    )
                    self._emit(StageEvent.RUN_END, {})
                finally:
                    # Single cleanup point for the streaming STT sessions covering every
                    # path out of the inner try (no_speech, STT error, empty/non-empty
                    # success, later-stage exceptions). finish() already drains/closes
                    # the gRPC call, so this is a no-op there; for the error/early-exit
                    # paths and for parallel sessions whose result went unused it
                    # actually closes the session.
                    await self._close_stt_streams(stt_streams)
                    # Record the run on every non-empty-pcm path (empty-STT return,
                    # success, TTS-fail, exception) but a silent follow-up window.
                    # Defer the live broadcast until the lock is released so a slow
                    # WebSocket consumer can't backpressure the next run on this speaker.
                    if keep_record:
                        pending_run = await self._store_run(
                            record, t0, pcm, pcm2, tts_audio, tts_mime
                        )
        finally:
            # Outside the lock: first drain any in-flight live stage broadcasts, then
            # push the finalized run. The live partials are scheduled as background
            # tasks (see _emit_live) so a slow WS client can't backpressure the
            # per-device lock held during the run; draining the chain HERE — outside
            # the lock — both guarantees the finalized broadcast never overtakes a
            # partial and keeps a stuck client from delaying the next run on this
            # speaker. This `finally` (not merely after the `async with`) also ensures
            # the finalized broadcast still fires on the early-return paths inside the
            # lock (no_speech / empty-transcription set `pending_run` then `return`).
            # Fully isolated — a broadcast failure or slow client must never affect the
            # run or the lock.
            # The run owns its STT sessions from the claim on, so a cancel (a new wake)
            # that lands while it waits for the lock, before the inner try, closes them
            # here; aclose() is idempotent, so the inner close is harmless.
            await self._close_stt_streams(stt_streams)
            tail = self._live_send_tail
            self._live_send_tail = None
            if tail is not None:
                try:
                    await tail
                except Exception as e:  # noqa: BLE001 - draining must never break the run
                    logger.debug(f"{self.name}: live run update drain failed: {e}")
            await self._broadcast_run(pending_run)

    def _condition_audio(self, pcm: bytes, pcm2: bytes, trim: bool) -> tuple[bytes, bytes]:
        """The captured sample as STT gets it and the run log stores it: the lead-in
        trimmed off (when `trim`) and the selected channel conditioned.

        Trim the configured lead-in (wake-word tail / button-press click) off the
        start of the captured sample. The trim is applied ONCE, so every downstream
        consumer in this run uses the trimmed sample: STT, the t_vad metric, the
        capture-session WAV and the stored diagnostic audio. Read live off core.vad
        so it hot-applies. trim_start_pcm never returns empty audio.

        Optional pre-STT conditioning of the FULL utterance (read live off core.vad,
        so the toggles hot-apply). High-pass first to strip DC/rumble, then
        peak-normalize so the quieter less-processed channel reaches a consistent
        level without clipping. Applied ONCE to the whole sample (never per chunk),
        so the capture WAV, the stored diagnostic audio and STT all get the exact
        audio Whisper sees. `pcm2` (the other channel) is deliberately kept raw, so
        the stored stereo WAV allows comparing conditioned vs raw channels.
        NOTE: mic_normalize gates ONLY this pre-STT normalization; the decision-only
        VAD boost applied per chunk in on_audio is gated by `core.vad.mic_auto_gain`.
        """
        trim_ms = self.core.vad.trim_start_ms
        if trim_ms > 0 and trim:
            trimmed = trim_start_pcm(pcm, trim_ms)
            if len(trimmed) != len(pcm):
                logger.info(
                    f"{self.name}: ✂️ trimmed {len(pcm) - len(trimmed)} bytes "
                    f"(~{trim_ms} ms) off sample start"
                )
            pcm = trimmed
            # Trim the other (non-selected) channel by the same amount so
            # the two channels of the stored stereo WAV stay time-aligned.
            if pcm2:
                pcm2 = trim_start_pcm(pcm2, trim_ms)
        if self.core.vad.mic_highpass:
            pcm = highpass(pcm)
        if self.core.vad.mic_normalize:
            pcm = normalize_peak(pcm)
        return pcm, pcm2

    def _new_record(self, reason, run: Run | None = None, pcm: bytes = b"", **fields) -> dict:
        """A pipeline run's record before its stages fill it in (new_record with the
        stage fields at their not-run values); `fields` set the rest. A voice run's
        (`run`) also carries its second-stage wake-word gate: the verify latency and
        score, 0/None for an ungated (non-wake / disabled / passthrough) run; a text
        request has neither key.

        t_vad is the captured-utterance duration in ms (the "VAD capture" segment
        shown in the UI waterfall), derived from the PCM length — NOT the CPU time
        spent running the VAD. audio_ms is intentionally left None for now: the
        reply's playback duration is not measured yet."""
        if run is not None:
            fields = {"t_wakeword": run.verify_ms, "wakeword_score": run.score, **fields}
        return new_record(self.name, reason, **{
            "t_vad": int(len(pcm) / (SAMPLE_RATE * 2) * 1000),
            "t_stt": 0, "t_llm": 0, "t_stress": 0, "t_tts": 0,
            "stress_text": "",
            "filler_text": "", "t_filler": None,
            "model": None, "tokens": None,
            "audio_ms": None, "audio_bytes": None, "audio_fmt": None,
            "rounds": [],
            "request": None,
            # The recognition chain that served the request.
            "chain": None,
            **fields,
        })

    async def _store_run(self, record: dict, t0: float, pcm: bytes = b"", pcm2: bytes = b"",
                         tts_audio=None, tts_mime=None) -> dict | None:
        """Insert the run record, publish it to the telemetry (whether or not the run
        log is on, with its run log id when it has one) and store its audio; returns the
        summary row to broadcast (_broadcast_run), or None. A recording failure must never
        break the run or swallow RUN_END, so it is fully wrapped."""
        record["t_total"] = int((time.perf_counter() - t0) * 1000)
        if self.runs_store is None:
            telemetry.publish_run(self.rt, record)
            return None
        try:
            run_id = await asyncio.to_thread(self.runs_store.insert, record)
        except Exception as e:
            logger.error(f"{self.name}: run record failed: {e}")
            telemetry.publish_run(self.rt, record)
            return None
        telemetry.publish_run(self.rt, record, run_id)
        # Store the finalized utterance audio in a rolling window of
        # the last runs.audio_keep, so it can be downloaded/played from
        # the log to diagnose mis-triggers (e.g. a wake-word tail
        # reaching STT). When the device streamed both mic channels the
        # WAV is STEREO: left = the pipeline/STT channel (trimmed +
        # conditioned — exactly what STT received), right = the other
        # raw channel (same trim, no conditioning); mono when the
        # device streams a single channel. Best-effort: a storage
        # failure must never break the run or swallow the broadcast.
        stored_audio = False
        if self.core.runs.store_audio and pcm:
            try:
                wav = pcm_to_wav_bytes(pcm, pcm2)
                await asyncio.to_thread(
                    self.runs_store.put_audio,
                    run_id, wav, self.core.runs.audio_keep,
                )
                stored_audio = True
            except Exception as e:
                logger.error(
                    f"{self.name}: utterance audio store failed: {e}"
                )
        # Store the generated TTS reply audio (native backend format) in the same
        # rolling window as the utterance audio (runs.audio_keep) so operators can play
        # back exactly what the assistant spoke — shown right after the Stress text in
        # the log detail view. Best-effort: a storage failure must never break the run.
        # tts_audio is set only when synthesis produced bytes (see speak_reply).
        if self.core.runs.store_audio and tts_audio:
            try:
                await asyncio.to_thread(
                    self.runs_store.put_tts_audio,
                    run_id, tts_audio, tts_mime, self.core.runs.audio_keep,
                )
            except Exception as e:
                logger.error(f"{self.name}: TTS audio store failed: {e}")
        if self.run_events is None:
            return None
        return summary_row(record, run_id, has_audio=stored_audio)

    async def _broadcast_run(self, row: dict | None) -> None:
        """Push a finalized run row to the run-log clients; a failure is logged."""
        if row is None:
            return
        try:
            await self.run_events.broadcast({"type": "run", "run": row})
        except Exception as e:
            logger.error(f"{self.name}: run broadcast failed: {e}")

    async def run_text(self, text: str, speak: bool = True) -> dict:
        """Public entry for TEXT requests (agent MCP `ask`): a full handler turn
        through the recognition chains (with tools and conversation context),
        optionally spoken on the speaker.

        The chains of a voice run entered at the handler, its reply spoken by the same
        speak_reply: no audio, no VAD/STT and no stage events of its own (there is no
        voice session for a text run, so they would reach the device out of band; the
        speech first ends an open follow-up window, see _close_followup). Serialized
        with voice runs via the same per-device lock. The run is recorded to the runs
        store with reason="text" and the request in stt_text, so the panel run log
        shows it in the usual column.

        Returns {"reply", "result", "error_stage", "error_text"}.
        """
        pending_run = None
        reply = ""
        try:
            async with self._lock:
                t0 = time.perf_counter()
                # A pipeline run's record: audio-stage fields stay at their text-run
                # values (t_vad/t_stt = 0, audio_* = None).
                record = self._new_record("text", stt_text=text)
                try:
                    # The recognition chains, entered at the handler step (no STT): each
                    # enabled chain's handler in order until one answers. No on_filler:
                    # a text run has no listener waiting in silence, so no early filler
                    # line is spoken.
                    run = await chains.run(
                        self.rt, device=self.name, request=self._llm_request, text=text,
                    )
                    self._record_chain_run(record, run)
                    reply = run.reply

                    # Only an answered turn is a conversation turn: don't pollute the
                    # context history with a spoken fallback or decline phrase.
                    if run.answered:
                        try:
                            context.append_context(
                                self._context_path,
                                text,
                                reply,
                                max_turns=self.core.context.max_turns,
                                ttl_seconds=self.core.context.ttl_seconds,
                            )
                        except Exception as e:
                            # Context persistence failure must not break the run.
                            logger.error(f"{self.name}: context append failed: {e}")

                    # Only the spoken text is accented: the value RETURNED to the caller
                    # stays the plain `reply`, which is also returned when its speech
                    # fails.
                    if speak and reply:
                        try:
                            await self.speak_reply(reply, record)
                        except Exception:  # noqa: BLE001 - marked on the record and logged by speak_reply
                            pass
                except Exception as e:
                    # Mirror `_run`'s catch-all: an unexpected failure is recorded
                    # honestly and never propagates to the MCP caller.
                    record["result"] = "error"
                    record["error_stage"] = "pipeline"
                    record["error_text"] = str(e)
                    logger.exception(f"{self.name}: text run failed: {e}")
                finally:
                    # Record the run; the broadcast is deferred until the lock is
                    # released (same reason as `_run`).
                    pending_run = await self._store_run(record, t0)
        finally:
            # Outside the lock: best-effort push of the finalized run to live
            # panel subscribers; only fires when the insert succeeded.
            await self._broadcast_run(pending_run)
        return {
            "reply": reply,
            "result": record["result"],
            "error_stage": record["error_stage"],
            "error_text": record["error_text"],
        }

    async def _close_followup(self) -> None:
        """End a follow-up window that has heard no phrase yet the way silence ends it:
        the device gets `state idle` and turns its microphone off. Every speech calls it
        first (_voice), or proactive speech would be heard in the window as a phrase; an
        answer finds none open (its own finalized run is the current one)."""
        run = self._current
        if run is None or run.followup is None or run.finalized:
            return
        # Synchronously, before any await: on_audio drops whatever still arrives.
        run.finalized = True
        streams, run.stt_streams = run.stt_streams, {}
        logger.info(f"{self.name}: follow-up window closed for proactive speech")
        self._emit(StageEvent.STT_END, {"text": ""})
        self._emit(StageEvent.RUN_END, {})
        await self._close_stt_streams(streams)
