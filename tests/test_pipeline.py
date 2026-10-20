import asyncio
import io
import wave

import numpy as np
import pytest

from src.audio_prep import (
    VAD_BOOST_FLOOR,
    VAD_BOOST_TARGET,
    highpass,
    normalize_peak,
    pcm_to_wav_bytes,
    trim_start_pcm,
)
from src.core_config import (
    AckConfig,
    ChainConfig,
    ContextConfig,
    CoreConfig,
    PromptConfig,
)
from src.llm import LlmResult
from src.pipeline import SAMPLE_RATE, Pipeline
from src.llm_text import clean_llm_output
from src.pipeline_events import StageEvent
from src.plugins.llm.base import LlmConfig
from src.plugins.vad.webrtc import WebRtcVadBackend
from src.prompt_store import PromptStore
from src.runs_store import _LIST_COLS
from src.runtime import Runtime
from src.stage_errors import StageError
from src.tts import OwnedChunks, TtsBackend, pcm_mime

class FakeSvc:
    """Tiny ConfigService stand-in for the Runtime: exposes a live CoreConfig via
    `.core`, every handler's config via `.instance_config("llm", id)`, its closed
    vocabulary via `.vocabulary(id)` and the enabled TTS variants via
    `.tts_variants()` (the mutable `variants` list). The pipeline reads them through
    the Runtime, so mutating them here changes them live."""

    def __init__(self, core, llm_cfg, vocabulary=(), tts_variants=("fake",)):
        self._core = core
        self._llm = llm_cfg
        self._vocabulary = list(vocabulary)
        self.variants = list(tts_variants)

    @property
    def core(self):
        return self._core

    def instance_config(self, _category, _provider_id):
        return self._llm

    def vocabulary(self, _handler_id):
        return list(self._vocabulary)

    def tts_variants(self):
        return list(self.variants)


class FakeSttBackend:
    """STT double: returns scripted text regardless of the PCM passed in."""

    def __init__(self, text="распознанный текст"):
        self.text = text

    async def transcribe(self, pcm):
        return self.text


# Subclasses the real TtsBackend contract so the ABC's buffered single-chunk
# synthesize_stream default adapter applies — existing tests run unchanged
# through the new streaming TTS path.
class FakeTtsBackend(TtsBackend):
    def __init__(self, mime="audio/mpeg", audio=b"MP3"):
        self.mime = mime
        self.audio = audio

    async def synthesize(self, text, lang="ru"):
        return (self.mime, self.audio)


class FakePlayer:
    """The device's play hook: drains each clip (bytes or async chunks) and records
    (audio, mime), so tests can assert on what reached the device. Returns `stats`,
    as Connection.play does."""

    def __init__(self):
        self.calls = []
        self.stats = {"outcome": "played", "dry": 0, "gap_ms": 0}

    async def __call__(self, mime, audio):
        if not isinstance(audio, (bytes, bytearray)):
            audio = b"".join([c async for c in audio])
        self.calls.append((bytes(audio), mime))
        return self.stats


class FakeRunsStore:
    """In-memory RunsStore double: records inserted run dicts for assertions.

    Synchronous insert(), so it works as the target of asyncio.to_thread().
    """

    def __init__(self):
        self.records = []
        self.audio_calls = []  # (run_id, wav, keep) per put_audio
        self.audio = {}        # run_id -> wav
        self.tts_audio_calls = []  # (run_id, audio, mime, keep) per put_tts_audio
        self.tts_audio = {}        # run_id -> (audio, mime)

    def insert(self, rec):
        self.records.append(rec)
        return len(self.records)

    def put_audio(self, run_id, wav, keep):
        self.audio_calls.append((run_id, wav, keep))
        self.audio[run_id] = wav

    def get_audio(self, run_id):
        return self.audio.get(run_id)

    def put_tts_audio(self, run_id, audio, mime, keep):
        self.tts_audio_calls.append((run_id, audio, mime, keep))
        self.tts_audio[run_id] = (audio, mime)

    def get_tts_audio(self, run_id):
        return self.tts_audio.get(run_id)


class FakeRunEvents:
    """Captures broadcast payloads for assertions."""
    def __init__(self):
        self.broadcasts = []
    async def broadcast(self, payload):
        self.broadcasts.append(payload)


class FakeToolHub:
    """ToolHub double for the filler policy: is_slow() is membership in a fixed set
    of slow tool names (mirrors the real hub, where the owning source declares it)."""

    def __init__(self, slow_tools=()):
        self.slow_tools = set(slow_tools)

    def is_slow(self, name):
        return name in self.slow_tools


def make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="распознанный текст",
                  stt_backend=None, tts_backend=None, runs_store=None,
                  run_events=None, ack=False, hub=None, stress_backend=None,
                  tts_backends=None):
    # `tts_backend` is the single TTS variant "fake"; `tts_backends` ({id: backend})
    # makes every entry an enabled variant, in dict order.
    if tts_backends is None:
        tts_backends = {"fake": tts_backend or FakeTtsBackend()}
    # The data dir is hardcoded in config_store; the pipeline reads it as a module
    # attribute, so isolate per-test context files by monkeypatching DATA_DIR to
    # tmp_path BEFORE the pipeline (and its _context_path) is built.
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    # Test phrases start right at the wake: with the real 2 s pre-roll, whose speech
    # is the wake word and does not count, every wake-word test would have to stream
    # 2 s of audio before its phrase. test_bare_wake_word_ends_without_stt restores it.
    monkeypatch.setattr("src.pipeline.PREROLL_MS", 0)
    # The end-of-phrase ack chime is ON by default in CoreConfig, but it is one more
    # play() call; default it OFF here so tests see only the plays they assert on,
    # and let ack tests opt in (ack=True).
    # The pipeline now assembles the system prompt itself (build_system_prompt in
    # _run reads the active PromptStore profile), so seed a tmp store from a tmp
    # legacy prompt file to keep tests hermetic.
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("PROMPT BODY <<<<<TDW>>>>>", encoding="utf-8")
    prompt_store = PromptStore(str(tmp_path / "prompts.db"), seed_path=str(prompt_path))
    core = CoreConfig(
        context=ContextConfig(),
        ack=AckConfig(enabled=ack),
        prompt=PromptConfig(system_prompt_path=str(prompt_path)),
        # One recognition chain over the "fake" STT and handler below.
        chains=[ChainConfig(name="main", stt="fake", handler="fake")],
    )
    rt = Runtime(
        FakeSvc(core, LlmConfig(), tts_variants=list(tts_backends)),
        # Real WebRTC backend by default: tests that need scripted frame-level
        # speech/silence decisions swap it via install_fake_vad() before on_wake.
        vad_backend=WebRtcVadBackend(),
        stt_backends={"fake": stt_backend or FakeSttBackend(stt_text)},
        llm_backends={"fake": object()},
        # Accent stage backend: None by default so every existing test runs with the
        # stage skipped; the accent test injects a fake accentizer.
        stress_backend=stress_backend,
        tts_backends=tts_backends,
        # Default hub double: the filler tests' slow tools are slow, everything
        # else (set_light, ...) is fast — the same shape the real ToolHub exposes.
        hub=hub or FakeToolHub(slow_tools={"search_events", "google", "get_current_weather"}),
        runs_store=runs_store,
        run_events=run_events,
        prompt_store=prompt_store,
    )
    pipeline = Pipeline(name, rt)
    events = []
    pipeline.send_event = lambda et, data: events.append((et, data))
    pipeline.play = FakePlayer()
    pipeline.has_speaker = True
    return pipeline, events


class FakeVad:
    """VAD double: is_speech(frame, rate) returns scripted booleans in order.

    Once the script runs out, it repeats the last value forever (so tests can
    feed "speech then silence" with a short script, or "always X" with one item).
    """

    def __init__(self, script):
        self._script = list(script)
        self._i = 0

    def is_speech(self, frame, rate):
        if self._i < len(self._script):
            val = self._script[self._i]
            self._i += 1
        else:
            val = self._script[-1] if self._script else False
        return val


def install_fake_vad(pipeline, script):
    """Swap in a real WebRtcVadBackend whose webrtcvad.Vad object is a scripted
    FakeVad, preserving frame-level control over the speech/silence decisions.
    The pipeline opens its session from rt.vad_backend in on_wake, so this must
    run BEFORE on_wake (it does in every test)."""
    pipeline.rt.vad_backend = WebRtcVadBackend(vad_factory=lambda: FakeVad(script))


def set_small_vad_thresholds(pipeline):
    """Shrink VAD thresholds so end-pointing fires after only a few 20 ms frames.

    The pipeline reads these live off rt.core.vad, so we mutate the CoreConfig the
    runtime hands out (CoreConfig is a pydantic model: attribute assignment works)."""
    vad = pipeline.rt.core.vad
    vad.min_speech_ms = 40         # 2 frames of speech to arm
    vad.silence_ms = 100           # 5 frames of trailing silence to end
    vad.max_utterance_ms = 400     # 20 frames hard cap
    vad.no_speech_timeout_ms = 200  # 10 frames with no speech -> finalize


def make_llm_result(reply="ответ", **overrides):
    """Build a canned LlmResult with neutral observability defaults."""
    fields = {
        "reply": reply,
        "model": None,
        "tokens": None,
        "rounds": [],
        "request_debug": {},
        "tool_used": False,
    }
    fields.update(overrides)
    return LlmResult(**fields)


def patch_llm(monkeypatch, reply="ответ", error=None, result=None):
    """Stub the whole LLM stage. STT is injected as a fake backend, not patched.

    Replaces llm.LlmStage with a fake whose respond() returns a canned LlmResult
    (built from `reply`, or `result` verbatim when given). When `error` is given
    (a StageError), respond() raises it instead."""

    class FakeLlmStage:
        def __init__(self, backend, hub, cfg):
            pass

        async def respond(self, req, *, on_filler=None, has_next=False):
            if error is not None:
                raise error
            return result if result is not None else make_llm_result(reply)

    monkeypatch.setattr("src.llm.LlmStage", FakeLlmStage)


def types_of(events):
    return [et for et, _ in events]


def assert_all_str(events):
    for _, data in events:
        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


async def settle(pipeline):
    """Wait until nothing the pipeline started on its own is left running: the run a
    finalizing on_audio started as its own task and the wake-word checks."""
    while True:
        tasks = {t for t in (pipeline._run_task, *pipeline._background)
                 if t is not None and not t.done()}
        if not tasks:
            return
        done, _ = await asyncio.wait(tasks)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()


# A phrase as the device streams it, in whole 20 ms VAD frames: two frames of speech
# (the min_speech_ms of set_small_vad_thresholds), then the five frames of trailing
# silence (its silence_ms) the server's VAD end-points it on, at its last frame.
PHRASE = b"\x01\x02" * 320 * 7
PHRASE_VAD = [True, True, False]  # FakeVad repeats its last value


def install_phrase_vad(pipeline):
    """Script the VAD of every run to end-point a PHRASE at its last frame. Before
    on_wake: a run opens its VAD session there (a follow-up window as it opens)."""
    set_small_vad_thresholds(pipeline)
    install_fake_vad(pipeline, PHRASE_VAD)


async def say(pipeline, pcm2=None):
    """One run ended the way production ends it: a start, then the PHRASE (and `pcm2`,
    the device's second mic channel, if any), which the server's VAD end-points; waits
    for the run that starts."""
    install_phrase_vad(pipeline)
    pipeline.on_wake("cid", None)
    await pipeline.on_audio(PHRASE, pcm2)
    assert pipeline._run_task is not None  # the VAD end-pointed the PHRASE
    await settle(pipeline)


async def test_happy_path(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")

    await say(pipeline)

    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END,
        StageEvent.STT_END,
        StageEvent.INTENT_START,
        StageEvent.INTENT_END,
        StageEvent.TTS_START,
        StageEvent.RUN_END,
    ]

    data = dict(events)
    assert data[StageEvent.STT_END] == {"text": "включи свет"}
    assert data[StageEvent.TTS_START] == {"text": "готово"}
    # The synthesized clip reached the device's play hook with the backend's mime.
    assert pipeline.play.calls == [(b"MP3", "audio/mpeg")]
    assert_all_str(events)


async def test_empty_stt(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="")

    await say(pipeline)

    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]
    data = dict(events)
    assert data[StageEvent.STT_END] == {"text": ""}
    assert_all_str(events)


# The Whisper hallucination filter is the groq STT brick's concern now; its unit
# tests live in tests/test_stt.py against GroqSttBackend directly (the backend
# returns "" for a hallucinated transcript, which the pipeline handles via the
# existing empty-transcription path — covered by the empty-STT tests above).

# --- decision-only VAD boost (core.vad.mic_auto_gain) ---------------------------
# The pipeline applies the boost to every chunk BEFORE it reaches the VAD session
# (any engine), while the buffer/STT path keeps the raw bytes. The vad_boost
# primitive itself is unit-tested in tests/test_audio_prep.py.


class RecordingVadSession:
    """VadSession double: records every chunk fed to it and never finalizes."""

    def __init__(self):
        self.fed = []

    def feed(self, chunk):
        self.fed.append(chunk)
        return None

    def debug_state(self):
        return {}


class RecordingVadBackend:
    """VadBackend double: open() hands out a fresh recording session per run and
    keeps them all, so tests can inspect exactly what the pipeline fed."""

    def __init__(self):
        self.sessions = []

    def open(self, policy):
        session = RecordingVadSession()
        self.sessions.append(session)
        return session


def _const_chunk(value, samples=320):
    """One chunk of 16-bit mono PCM with every sample set to `value`."""
    return np.full(samples, value, dtype="<i2").tobytes()


def _boosted(chunk, peak):
    """The bytes vad_boost produces for `chunk` at running utterance peak `peak`."""
    gain = min(VAD_BOOST_TARGET / peak, 128.0)
    return np.clip(
        np.frombuffer(chunk, dtype="<i2").astype(np.float32) * gain, -32768, 32767
    ).astype("<i2").tobytes()


async def test_mic_auto_gain_boosts_vad_feed_but_buffers_raw(tmp_path, monkeypatch):
    # With mic_auto_gain on, once a loud chunk has raised the running peak above
    # the floor, the VAD session receives BOOSTED bytes (scaled by
    # min(target/peak, 128)) while the run's buffer keeps the raw chunk bytes.
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.rt.core.vad.mic_auto_gain = True
    backend = RecordingVadBackend()
    pipeline.rt.vad_backend = backend

    pipeline.on_wake("cid", None)
    peak = 3000
    loud = _const_chunk(peak)
    quiet = _const_chunk(100)
    await pipeline.on_audio(loud)
    await pipeline.on_audio(quiet)

    [session] = backend.sessions
    # The peak is tracked from the chunk itself BEFORE boosting, so even the
    # first loud chunk is already lifted with the utterance-level gain.
    assert session.fed == [_boosted(loud, peak), _boosted(quiet, peak)]
    assert session.fed[1] != quiet  # the decision really saw boosted bytes...
    assert pipeline.preroll_pcm == loud + quiet  # ...while the buffer kept raw


async def test_mic_auto_gain_off_feeds_verbatim_and_tracks_no_peak(tmp_path, monkeypatch):
    # Default (mic_auto_gain=False): the session receives the chunk verbatim and
    # the pipeline does no peak tracking at all.
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    backend = RecordingVadBackend()
    pipeline.rt.vad_backend = backend

    pipeline.on_wake("cid", None)
    loud = _const_chunk(3000)
    await pipeline.on_audio(loud)

    [session] = backend.sessions
    assert session.fed == [loud]


async def test_mic_auto_gain_preroll_below_floor_fed_verbatim(tmp_path, monkeypatch):
    # While the running peak is still below VAD_BOOST_FLOOR, pre-roll silence is
    # NOT amplified: chunks reach the session unchanged (boost identity).
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.rt.core.vad.mic_auto_gain = True
    backend = RecordingVadBackend()
    pipeline.rt.vad_backend = backend

    pipeline.on_wake("cid", None)
    preroll = _const_chunk(VAD_BOOST_FLOOR - 20)
    await pipeline.on_audio(preroll)

    [session] = backend.sessions
    assert session.fed == [preroll]


async def test_mic_auto_gain_peak_resets_between_runs(tmp_path, monkeypatch):
    # The running peak is per utterance: a loud first run must not leak its peak
    # into the next run — a quiet (below-floor) chunk in run 2 is fed verbatim.
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.rt.core.vad.mic_auto_gain = True
    backend = RecordingVadBackend()
    pipeline.rt.vad_backend = backend

    pipeline.on_wake("cid1", None)
    loud = _const_chunk(3000)
    await pipeline.on_audio(loud)
    assert backend.sessions[0].fed == [_boosted(loud, 3000)]

    pipeline.on_wake("cid2", None)
    quiet = _const_chunk(VAD_BOOST_FLOOR - 20)  # would be boosted at peak=3000
    await pipeline.on_audio(quiet)

    assert backend.sessions[-1].fed == [quiet]


async def test_pipelines_are_independent(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    a, _ = make_pipeline(tmp_path, monkeypatch, name="a")
    b, _ = make_pipeline(tmp_path, monkeypatch, name="b")

    a.on_wake("cid", None)
    await a.on_audio(b"\x01\x02" * 100)

    # Pushing audio only into "a" must not touch "b".
    assert b.preroll_pcm == b""
    assert a._context_path != b._context_path
    assert a._context_path.endswith("context_a.txt")
    assert b._context_path.endswith("context_b.txt")


# A 640-byte (one 20 ms frame) chunk of silence-shaped PCM. The injected FakeVad
# decides speech/silence regardless of the bytes; one chunk == one VAD frame.
FRAME = b"\x00" * 640

FULL_SEQUENCE = [
    StageEvent.RUN_START,
    StageEvent.STT_START,
    StageEvent.STT_VAD_START,
    StageEvent.STT_VAD_END,
    StageEvent.STT_END,
    StageEvent.INTENT_START,
    StageEvent.INTENT_END,
    StageEvent.TTS_START,
    StageEvent.RUN_END,
]


async def test_vad_endpoint_finalize(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    set_small_vad_thresholds(pipeline)
    # 3 speech frames (>= 40 ms min) then 6 silence frames (>= 100 ms) -> endpoint.
    install_fake_vad(pipeline, [True, True, True, False, False, False, False, False, False])

    pipeline.on_wake("cid", None)
    # 9 frames in one chunk so all are processed in a single on_audio call.
    await pipeline.on_audio(FRAME * 9)
    await settle(pipeline)

    assert types_of(events) == FULL_SEQUENCE
    assert_all_str(events)


async def test_bare_wake_word_ends_without_stt(tmp_path, monkeypatch):
    # A wake-word run whose only speech is the wake word in the 2 s pre-roll, then
    # silence: it ends as no_speech no_speech_timeout_ms after the pre-roll and the
    # wake word alone never reaches STT (Whisper would make up a phrase from it), and
    # no end-of-phrase chime plays.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, ack=True)
    monkeypatch.setattr("src.pipeline.PREROLL_MS", 2000)  # the real pre-roll
    set_small_vad_thresholds(pipeline)
    pipeline.rt.core.vad.max_utterance_ms = 15000
    install_fake_vad(pipeline, [True] * 5 + [False] * 200)
    stt_calls = []
    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy_transcribe

    pipeline.on_wake("cid", "Захар")
    await pipeline.on_audio(FRAME * 109)  # 2000 ms pre-roll + 180 ms
    await settle(pipeline)
    assert StageEvent.RUN_END not in types_of(events)
    await pipeline.on_audio(FRAME * 1)    # 200 ms after the pre-roll
    await settle(pipeline)
    await _drain_ack_tasks(pipeline)
    assert StageEvent.RUN_END in types_of(events)
    assert stt_calls == []
    assert wav_plays(pipeline) == []


async def test_vad_maxlen_finalize(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="ок")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="команда")
    set_small_vad_thresholds(pipeline)
    # Always speech: no trailing silence, so only the max-length cap can finalize.
    install_fake_vad(pipeline, [True])

    pipeline.on_wake("cid", None)
    # max_utterance_ms=400 -> 20 frames; feed 21 to cross the cap.
    await pipeline.on_audio(FRAME * 21)
    await settle(pipeline)

    assert StageEvent.RUN_END in types_of(events)
    # Full happy path because STT returned text.
    assert types_of(events) == FULL_SEQUENCE


async def test_vad_no_speech_finalize(tmp_path, monkeypatch):
    # No speech detected by VAD -> STT is SKIPPED entirely (so Whisper can't
    # hallucinate on non-speech audio); STT_END is emitted empty and the run ends
    # without intent/TTS.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="   ")
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    install_fake_vad(pipeline, [False])

    # Spy on transcribe to assert it is never called on a no-speech run.
    stt_calls = []
    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy_transcribe

    pipeline.on_wake("cid", None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)
    await settle(pipeline)

    # STT must never run on a no-speech utterance.
    assert stt_calls == []
    # STT_START (from on_wake) is balanced by an empty STT_END; no intent/TTS.
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]
    data = dict(events)
    assert data[StageEvent.STT_END] == {"text": ""}
    assert_all_str(events)


async def test_no_speech_run_is_recorded_as_empty(tmp_path, monkeypatch):
    # A no-speech finalization skips STT but still records the run as empty
    # (result="empty", stt_text="", t_stt=0, reason="no_speech").
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="   ", runs_store=store)
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    install_fake_vad(pipeline, [False])

    stt_calls = []
    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy_transcribe

    pipeline.on_wake("cid", None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)
    await settle(pipeline)

    # STT never ran, but the run is recorded once as an empty run.
    assert stt_calls == []
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "empty"
    assert rec["stt_text"] == ""
    assert rec["t_stt"] == 0
    assert rec["reason"] == "no_speech"


async def test_run_broadcast_on_no_speech(tmp_path, monkeypatch):
    # A no_speech finalization returns early INSIDE the lock, but the run must still be
    # pushed to live admin-panel WS subscribers, not only recorded.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="   ", runs_store=store, run_events=hub,
    )
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    install_fake_vad(pipeline, [False])

    pipeline.on_wake("cid", None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)
    await settle(pipeline)

    # The empty run is recorded once AND broadcast once to live subscribers.
    assert len(store.records) == 1
    assert store.records[0]["reason"] == "no_speech"
    assert len(hub.broadcasts) == 1
    assert hub.broadcasts[0]["type"] == "run"
    assert hub.broadcasts[0]["run"]["result"] == "empty"


async def test_vad_backend_hot_swap_applies_on_next_run(tmp_path, monkeypatch):
    # The pipeline opens one VadSession per run from rt.vad_backend, so swapping the
    # backend on the runtime (what the reconfigurator does on a vad.* change) takes
    # effect on the NEXT on_wake without rebuilding the pipeline. This replaces the
    # old "on_start rebuilds webrtcvad.Vad when aggressiveness changed" hack.
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)

    class RecordingBackend(WebRtcVadBackend):
        def __init__(self):
            super().__init__(vad_factory=lambda: FakeVad([False]))
            self.opened = 0

        def open(self, policy):
            self.opened += 1
            return super().open(policy)

    first, second = RecordingBackend(), RecordingBackend()
    pipeline.rt.vad_backend = first
    pipeline.on_wake("cid", None)
    assert (first.opened, second.opened) == (1, 0)

    # Hot-swap: the next run opens its session from the NEW backend.
    pipeline.rt.vad_backend = second
    pipeline.on_wake("cid", None)
    assert (first.opened, second.opened) == (1, 1)


async def test_finalize_once_race(tmp_path, monkeypatch):
    # Two chunks that each end the phrase arrive before the run task gets to run: the
    # run is claimed once, so one run happens and the second chunk is dropped.
    patch_llm(monkeypatch, reply="здравствуйте")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="привет")
    install_phrase_vad(pipeline)

    pipeline.on_wake("cid", None)
    await pipeline.on_audio(PHRASE)
    first = pipeline._run_task
    await pipeline.on_audio(PHRASE)
    assert pipeline._run_task is first
    await settle(pipeline)
    assert types_of(events) == FULL_SEQUENCE


async def test_history_flows_across_runs(tmp_path, monkeypatch):
    import json

    from src import context

    # Capturing fake: records (user_text, history) per request and returns a
    # scripted reply.
    seen = []  # list of (user_text, history)
    replies = {"первый вопрос": "первый ответ", "второй вопрос": "второй ответ"}

    class FakeLlmStage:
        def __init__(self, backend, hub, cfg):
            pass

        async def respond(self, req, *, on_filler=None, has_next=False):
            seen.append((req.user_text, req.history))
            return make_llm_result(replies[req.user_text])

    monkeypatch.setattr("src.llm.LlmStage", FakeLlmStage)

    pipeline, _ = make_pipeline(tmp_path, monkeypatch, name="hist", stt_text="первый вопрос")

    # Run 1.
    await say(pipeline)

    # Run 1 saw no prior history.
    assert seen[0][0] == "первый вопрос"
    assert not seen[0][1]  # [] or None
    # After run 1 the context file holds the first exchange as JSONL.
    saved = context.load_context(pipeline._context_path)
    assert saved == [
        {"role": "user", "content": "первый вопрос"},
        {"role": "assistant", "content": "первый ответ"},
    ]
    for ln in [
        ln
        for ln in open(pipeline._context_path, encoding="utf-8").read().splitlines()
        if ln
    ]:
        json.loads(ln)

    # Run 2 on the same pipeline instance with a different utterance: on_wake starts
    # a new run.
    pipeline.rt.stt_backends["fake"].text = "второй вопрос"
    await say(pipeline)

    # Run 2 received the first exchange as history.
    assert seen[1][0] == "второй вопрос"
    assert seen[1][1] == [
        {"role": "user", "content": "первый вопрос"},
        {"role": "assistant", "content": "первый ответ"},
    ]


async def test_run_recorded_on_happy_path(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    # Exactly one run recorded.
    assert len(store.records) == 1
    rec = store.records[0]
    # Stubbed LLM left no tool rounds -> "ok"; full transcript + reply captured.
    assert rec["result"] == "ok"
    assert rec["stt_text"] == "включи свет"
    assert rec["llm_text"] == "готово"
    # Stage timings present (t_total set in the finally; t_vad from audio length).
    assert rec["t_total"] >= 0
    assert rec["t_vad"] > 0
    assert "t_stt" in rec and "t_llm" in rec and "t_tts" in rec
    assert rec["audio_bytes"] == len(b"MP3")
    assert rec["audio_fmt"] == "mpeg"
    assert rec["error_stage"] is None


async def test_tts_audio_stored_on_happy_path(tmp_path, monkeypatch):
    # A normal voice run stores the GENERATED TTS reply audio (native backend
    # format + mime) keyed by the new run_id: the clip streams to the play hook
    # and the tee keeps a copy for the store.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет",
        tts_backend=FakeTtsBackend(mime="audio/mpeg", audio=b"MP3"),
        runs_store=store,
    )

    await say(pipeline)

    # Stored exactly once, keyed by the first run_id, with the configured keep.
    assert len(store.tts_audio_calls) == 1
    run_id, audio, mime, keep = store.tts_audio_calls[0]
    assert run_id == 1
    assert keep == pipeline.rt.core.runs.audio_keep == 100
    assert store.get_tts_audio(run_id) == (b"MP3", "audio/mpeg")


async def test_tts_audio_raw_pcm_stored_as_wav(tmp_path, monkeypatch):
    # A raw-PCM backend (Yandex / Fish / Piper streaming): the stored TTS audio is
    # the native clip in a WAV container at the backend's rate, so the panel can
    # play it; the record names the native format.
    patch_llm(monkeypatch, reply="готово")
    pcm = b"\x01\x00\x02\x00" * 80
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет",
        tts_backend=FakeTtsBackend(mime="audio/pcm;rate=24000", audio=pcm),
        runs_store=store,
    )

    await say(pipeline)

    audio, mime = store.get_tts_audio(1)
    assert mime == "audio/wav"
    with wave.open(io.BytesIO(audio), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (24000, 1, 2)
        assert w.readframes(w.getnframes()) == pcm
    assert pipeline.play.calls == [(pcm, "audio/pcm;rate=24000")]
    assert store.records[0]["audio_fmt"] == "pcm"


async def test_run_accentizes_tts_text_but_keeps_llm_text_original(tmp_path, monkeypatch):
    # The accent stage runs between LLM and TTS: TTS must synthesize the ACCENTED
    # text, while the recorded llm_text stays the original (mark-free) reply and
    # t_stress is set.
    class CapturingTtsBackend(TtsBackend):
        def __init__(self):
            self.texts = []

        async def synthesize(self, text, lang="ru"):
            self.texts.append(text)
            return ("audio/mpeg", b"MP3")

    class FakeAccentizer:
        async def accentize(self, text):
            return text + " [ACC]"

    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    tts = CapturingTtsBackend()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет",
        tts_backend=tts, runs_store=store, stress_backend=FakeAccentizer(),
    )

    await say(pipeline)

    # (a) TTS synthesized the accented text.
    assert tts.texts == ["готово [ACC]"]
    # The TTS_START event carries the accented text too.
    assert dict(events)[StageEvent.TTS_START] == {"text": "готово [ACC]"}
    # (b) and (c): the recorded run keeps the original reply and records t_stress.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["llm_text"] == "готово"
    assert rec["t_stress"] >= 0
    # (d) the accent-stage OUTPUT (text sent to TTS) is stored since it changed.
    assert rec["stress_text"] == "готово [ACC]"


async def test_run_accent_noop_leaves_stress_text_empty(tmp_path, monkeypatch):
    # When the accentizer returns the SAME text (nothing to stress / disabled-like
    # behavior), stress_text stays empty so the panel doesn't show a no-op section.
    class CapturingTtsBackend(TtsBackend):
        def __init__(self):
            self.texts = []

        async def synthesize(self, text, lang="ru"):
            self.texts.append(text)
            return ("audio/mpeg", b"MP3")

    class NoopAccentizer:
        async def accentize(self, text):
            return text  # unchanged

    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    tts = CapturingTtsBackend()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет",
        tts_backend=tts, runs_store=store, stress_backend=NoopAccentizer(),
    )

    await say(pipeline)

    assert tts.texts == ["готово"]
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["llm_text"] == "готово"
    assert rec["stress_text"] == ""


async def test_run_records_llm_result_observability_fields(tmp_path, monkeypatch):
    # model/tokens/rounds/request now land in the run record via the returned
    # LlmResult (no shared trace dict), and tool_used drives result="tool".
    rounds = [
        {"round": 1, "note": "tool call", "tokens": 30, "content": "",
         "calls": [{"name": "set_light", "args": {}, "result": "ok"}]},
        {"round": 2, "note": "final answer", "tokens": 12, "content": "готово",
         "calls": []},
    ]
    request_debug = {
        "system_prompt": "P", "context": [], "user_text": "включи свет", "tools": [],
    }
    patch_llm(monkeypatch, result=make_llm_result(
        "готово", model="m-final", tokens=42, rounds=rounds,
        request_debug=request_debug, tool_used=True,
    ))
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    rec = store.records[0]
    assert rec["model"] == "m-final"
    assert rec["tokens"] == 42
    assert rec["rounds"] == rounds
    assert rec["request"] == request_debug
    assert rec["result"] == "tool"
    assert rec["llm_text"] == "готово"


async def test_llm_error_records_partial_observability(tmp_path, monkeypatch):
    # On StageError the stage attaches the partial data accumulated before the
    # failure; the pipeline must record it (same fields the old partial trace
    # dict carried) alongside the error classification.
    rounds = [{"round": 1, "note": "tool call", "tokens": 30, "content": "",
               "calls": [{"name": "set_light", "args": {}, "result": "ok"}]}]
    request_debug = {"system_prompt": "P", "context": [], "user_text": "q", "tools": []}
    err = StageError("llm", "boom mid-loop")
    err.partial = {
        "model": "m-tool", "tokens": 30, "rounds": rounds, "request": request_debug,
    }
    patch_llm(monkeypatch, error=err)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "LLM"
    assert rec["error_text"] == "boom mid-loop"
    assert rec["model"] == "m-tool"
    assert rec["tokens"] == 30
    assert rec["rounds"] == rounds
    assert rec["request"] == request_debug


async def test_utterance_audio_stored_on_happy_path(tmp_path, monkeypatch):
    # A normal run stores its finalized utterance audio exactly once: keyed by the
    # new run_id, as a WAV (RIFF header), with keep == the configured audio_keep.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    assert len(store.audio_calls) == 1
    run_id, wav, keep = store.audio_calls[0]
    # The store returns id 1 for the first insert; audio is keyed by that run_id.
    assert run_id == 1
    assert wav.startswith(b"RIFF")
    # Default RunsConfig.audio_keep is 100.
    assert keep == pipeline.rt.core.runs.audio_keep == 100


async def test_utterance_audio_stored_on_no_speech_run(tmp_path, monkeypatch):
    # A no-speech finalize skips STT but STILL stores the utterance audio (that is
    # exactly what the operator wants to hear); the run is recorded as empty.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="   ", runs_store=store)
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    install_fake_vad(pipeline, [False])

    pipeline.on_wake("cid", None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)
    await settle(pipeline)

    # Audio stored even though STT was skipped.
    assert len(store.audio_calls) == 1
    assert store.audio_calls[0][1].startswith(b"RIFF")
    # The recorded run is an empty (no_speech) run.
    assert len(store.records) == 1
    assert store.records[0]["result"] == "empty"


async def test_utterance_audio_not_stored_when_disabled(tmp_path, monkeypatch):
    # With store_audio disabled, a normal run records but stores NO audio.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)
    pipeline.rt.core.runs.store_audio = False

    await say(pipeline)

    assert len(store.records) == 1
    assert store.audio_calls == []


async def test_run_broadcast_on_happy_path(tmp_path, monkeypatch):
    # With both a store and a run-events hub, a happy-path run now streams live
    # partials BEFORE the finalized broadcast. The LAST broadcast is the finalized
    # summary, carrying the same shape /api/runs returns.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await say(pipeline)

    # More than one broadcast: live partials precede the single finalized one.
    assert len(hub.broadcasts) > 1
    payload = hub.broadcasts[-1]
    assert payload["type"] == "run"
    # The store returns id 1 for the first insert; the summary echoes it.
    assert payload["run"]["id"] == 1
    assert payload["run"]["result"] in ("ok", "tool")
    # The broadcast run dict is the summary shape (_LIST_COLS) plus the has_audio flag.
    assert set(payload["run"].keys()) == set(_LIST_COLS) | {"has_audio"}
    # This happy-path run stored its utterance audio, so the flag is set.
    assert payload["run"]["has_audio"] == 1
    # The finalized row is NOT a live partial.
    assert not payload["run"].get("live")


async def test_run_streams_live_stage_updates(tmp_path, monkeypatch):
    # The pipeline streams ordered live partials (STT then LLM) before the
    # finalized broadcast, so the admin panel fills the row in stage by stage.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await say(pipeline)

    assert len(hub.broadcasts) >= 3
    # First broadcast: the live STT partial — recognized text present, no LLM yet,
    # no DB id yet.
    first = hub.broadcasts[0]
    assert first["run"]["live"]
    assert first["run"]["id"] is None
    assert first["run"]["stt_text"] == "включи свет"
    assert not first["run"].get("llm_text")
    # Some live partial before the final carries the LLM reply.
    assert any(
        b["run"].get("live") and b["run"].get("llm_text") == "готово"
        for b in hub.broadcasts
    )
    # Last broadcast: finalized (real id, not live).
    assert hub.broadcasts[-1]["run"]["id"] == 1
    assert not hub.broadcasts[-1]["run"].get("live")


async def test_no_live_partial_on_no_speech(tmp_path, monkeypatch):
    # A no_speech finalization skips STT entirely (returns before the live STT
    # partial), so it emits NO live partial — only the single finalized empty run.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="   ", runs_store=store, run_events=hub,
    )
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    install_fake_vad(pipeline, [False])

    pipeline.on_wake("cid", None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)
    await settle(pipeline)

    assert len(hub.broadcasts) == 1
    assert not hub.broadcasts[0]["run"].get("live")
    assert hub.broadcasts[0]["run"]["result"] == "empty"


async def test_no_broadcast_without_hub(tmp_path, monkeypatch):
    # run_events=None must not error and must record the run normally.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    assert len(store.records) == 1


async def test_run_recorded_on_empty_stt(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="", runs_store=store)

    await say(pipeline)

    # Empty STT short-circuits but still records a run with result "empty".
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "empty"
    assert rec["stt_text"] == ""
    assert rec["llm_text"] == ""
    assert rec["t_total"] >= 0


async def test_run_broadcast_on_empty_stt(tmp_path, monkeypatch):
    # An empty transcription returns early INSIDE the lock, but the run must still be
    # pushed to live admin-panel WS subscribers, not only recorded.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="", runs_store=store, run_events=hub,
    )

    await say(pipeline)

    # The empty run is recorded once AND pushed to live subscribers. The empty STT
    # still emits a live partial, so the FINAL (finalized) broadcast is the last one.
    assert len(store.records) == 1
    assert len(hub.broadcasts) >= 1
    payload = hub.broadcasts[-1]
    assert payload["type"] == "run"
    assert payload["run"]["result"] == "empty"
    assert not payload["run"].get("live")


# Subclasses TtsBackend so the streaming path's default synthesize_stream
# adapter raises the same error BEFORE the iterator is returned (the existing
# pre-URL TTS error path).
class RaisingTtsBackend(TtsBackend):
    """TTS double that always fails, to exercise the TTS except branch."""

    async def synthesize(self, text, lang="ru"):
        raise RuntimeError("tts boom")


async def test_llm_error_then_tts_fail_keeps_error_stage_llm(tmp_path, monkeypatch):
    # When the LLM raises a StageError the run is classified as an LLM error
    # but still continues into TTS with the configured fallback phrase. If TTS
    # then fails, the TTS except must NOT clobber the already-set LLM root cause.
    patch_llm(monkeypatch, error=StageError("llm", "модель недоступна"))
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch,
        stt_text="включи свет",
        tts_backend=RaisingTtsBackend(),
        runs_store=store,
    )

    await say(pipeline)

    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    # LLM stage/text preserved despite the later TTS failure.
    assert rec["error_stage"] == "LLM"
    assert rec["error_text"] == "модель недоступна"


async def test_llm_stage_error_speaks_reply_error(tmp_path, monkeypatch):
    # A generic LLM StageError: the run is recorded as an LLM error, but the user
    # hears the configured reply_error phrase (never the raw API error text).
    patch_llm(monkeypatch, error=StageError("llm", "boom from API"))
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store,
    )

    await say(pipeline)

    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END,
        StageEvent.STT_END,
        StageEvent.INTENT_START,
        StageEvent.INTENT_END,
        StageEvent.TTS_START,
        StageEvent.RUN_END,
    ]
    data = dict(events)
    assert data[StageEvent.TTS_START] == {"text": LlmConfig().reply_error}
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "LLM"
    assert rec["error_text"] == "boom from API"
    # The spoken fallback is what was sent to TTS, recorded as llm_text.
    assert rec["llm_text"] == LlmConfig().reply_error


async def test_llm_rate_limit_speaks_reply_rate_limit(tmp_path, monkeypatch):
    # kind="rate_limit" maps to the dedicated rate-limit phrase.
    patch_llm(monkeypatch, error=StageError("llm", "rate limited", kind="rate_limit"))
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="привет", runs_store=store,
    )

    await say(pipeline)

    data = dict(events)
    assert data[StageEvent.TTS_START] == {"text": LlmConfig().reply_rate_limit}
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "LLM"
    assert rec["error_text"] == "rate limited"


async def test_llm_stage_error_does_not_append_context(tmp_path, monkeypatch):
    # A failed LLM turn is not a conversation turn: nothing is appended to the
    # context history (the next run must not see the fallback phrase as a reply).
    from src import context

    appended = []
    monkeypatch.setattr(
        context, "append_context",
        lambda *a, **k: appended.append(a),
    )
    patch_llm(monkeypatch, error=StageError("llm", "boom"))
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="привет")

    await say(pipeline)

    assert appended == []


async def test_stt_stage_error_records_error_and_stays_silent(tmp_path, monkeypatch):
    # An STT failure ends the run silently (no LLM/TTS), but the record now says
    # error/STT instead of lying with result="empty". RaisingSttBackend (defined
    # below for the top-level-exception test) is reused with a StageError here.
    patch_llm(monkeypatch, reply="не должно быть произнесено")
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch,
        stt_backend=RaisingSttBackend(StageError("stt", "groq down")),
        runs_store=store,
    )

    await say(pipeline)

    # STT_END balances STT_START; RUN_END closes the run; no TTS events at all.
    assert types_of(events) == [
        StageEvent.RUN_START,
        StageEvent.STT_START,
        StageEvent.STT_VAD_START,
        StageEvent.STT_VAD_END,
        StageEvent.STT_END,
        StageEvent.RUN_END,
    ]
    assert dict(events)[StageEvent.STT_END] == {"text": ""}
    assert pipeline.play.calls == []  # nothing synthesized/served
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "STT"
    assert rec["error_text"] == "groq down"


# --- manual "record X seconds" capture-only mode -----------------------------

def _read_wav_bytes(wav_bytes):
    """Parse in-memory WAV bytes -> (nchannels, sampwidth, framerate, nframes, pcm)."""
    import io
    import wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return (
            w.getnchannels(), w.getsampwidth(), w.getframerate(),
            w.getnframes(), w.readframes(w.getnframes()),
        )


async def test_normal_run_hard_cap_truncates_at_60s(tmp_path, monkeypatch):
    # A NORMAL (non-capture) run is still truncated at the 60 s HARD_CAP_BYTES — the
    # duration-based capture cap must NOT leak into normal runs.
    # Feed > 60 s of PCM in one chunk; the run finalizes ("maxlen") with the buffer
    # capped at exactly HARD_CAP_BYTES.
    from src.pipeline import HARD_CAP_BYTES, SAMPLE_RATE

    captured = {}
    patch_llm(monkeypatch, reply="ок")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="команда")
    # This test asserts the HARD_CAP length reaches STT; disable the lead-in trim so
    # it is not coupled to trim_start_ms.
    pipeline.rt.core.vad.trim_start_ms = 0
    install_fake_vad(pipeline, [True])  # always speech: only the cap can finalize

    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy(pcm):
        captured["len"] = len(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy

    pipeline.on_wake("cid", None)
    # 65 s worth of PCM in a single chunk -> HARD_CAP fires, buffer trimmed to 60 s.
    await pipeline.on_audio(b"\x07\x08" * (SAMPLE_RATE * 65))
    await settle(pipeline)

    assert StageEvent.RUN_END in types_of(events)
    # The transcribed PCM was capped at exactly the 60 s HARD_CAP_BYTES.
    assert captured["len"] == HARD_CAP_BYTES


async def test_tts_fail_without_prior_error_sets_tts_stage(tmp_path, monkeypatch):
    # Sanity check the guard's other branch: a TTS failure with no earlier error
    # still records error_stage="TTS".
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch,
        stt_text="включи свет",
        tts_backend=RaisingTtsBackend(),
        runs_store=store,
    )

    await say(pipeline)

    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "TTS"
    assert rec["error_text"] == "tts boom"


# --- _trim_start_pcm (lead-in trim before STT) -----------------------------------

def test_trim_start_pcm_trims_normal_buffer():
    # 10 ms @ 16 kHz / 16-bit = 320 bytes off the front of a 1000-byte buffer.
    pcm = bytes(range(256)) * 4  # 1024 bytes; slice the first 1000 for a known size
    pcm = pcm[:1000]
    out = trim_start_pcm(pcm, 10)
    assert out == pcm[320:]
    assert len(out) == 680


def test_trim_start_pcm_zero_returns_unchanged():
    pcm = b"\x01\x02" * 100
    out = trim_start_pcm(pcm, 0)
    assert out == pcm


def test_trim_start_pcm_trim_exceeds_buffer_returns_unchanged():
    # 200 ms -> 6400 bytes, far larger than the 100-byte buffer: keep it intact so
    # we never hand empty audio to STT.
    pcm = b"\x01\x02" * 50  # 100 bytes
    out = trim_start_pcm(pcm, 200)
    assert out == pcm


def test_trim_start_pcm_negative_returns_unchanged():
    pcm = b"\x01\x02" * 100
    out = trim_start_pcm(pcm, -5)
    assert out == pcm


def test_trim_start_pcm_200ms_byte_count():
    # Sanity: 200 ms maps to exactly 6400 bytes when the buffer is long enough.
    pcm = b"\x00" * 10000
    out = trim_start_pcm(pcm, 200)
    assert len(pcm) - len(out) == 6400
    assert out == pcm[6400:]


async def test_stt_receives_trimmed_pcm(tmp_path, monkeypatch):
    # The STT backend must receive the captured sample minus the configured lead-in.
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    # 10 ms -> 320 bytes trimmed; the VAD end-point claims the whole buffer.
    pipeline.rt.core.vad.trim_start_ms = 10

    stt_calls = []
    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy_transcribe

    await say(pipeline)

    assert len(stt_calls) == 1
    assert stt_calls[0] == trim_start_pcm(PHRASE, 10)
    assert len(stt_calls[0]) == 4480 - 320 == 4160


async def test_stt_receives_full_pcm_when_trim_exceeds_short_utterance(
    tmp_path, monkeypatch
):
    # Default trim (200 ms = 6400 bytes) vs a short utterance (4480 bytes): the guard
    # returns the PCM unchanged end-to-end, so STT still gets the full sample.
    patch_llm(monkeypatch)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    # Leave trim_start_ms at its default (200).
    assert pipeline.rt.core.vad.trim_start_ms == 200

    stt_calls = []
    orig_transcribe = pipeline.rt.stt_backends["fake"].transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.rt.stt_backends["fake"].transcribe = spy_transcribe

    await say(pipeline)

    assert len(stt_calls) == 1
    assert stt_calls[0] == PHRASE
    assert len(stt_calls[0]) == 4480


async def test_trim_applies_to_record_and_stored_audio(tmp_path, monkeypatch):
    # The lead-in trim is applied to the sample ITSELF, once, so it affects every
    # consumer in the run — not just STT. Here we prove the run record's t_vad and the
    # stored diagnostic audio both reflect the trimmed (not the full) sample.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store,
    )
    # 10 ms -> 320 bytes trimmed.
    pipeline.rt.core.vad.trim_start_ms = 10

    await say(pipeline)

    # The run record's t_vad reflects the trimmed-length duration, not the full 4480 B.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["t_vad"] == int((4480 - 320) / (SAMPLE_RATE * 2) * 1000)

    # The stored diagnostic audio is the WAV built from the trimmed pcm. The first
    # insert returns run_id 1.
    assert store.get_audio(1) == pcm_to_wav_bytes(trim_start_pcm(PHRASE, 10))


# --- failure isolation: the run must always finalize cleanly ---------------------


class RaisingSttBackend:
    """STT double whose transcribe() always raises, to hit the _run top-level
    exception handler (the pipeline catch-all around STT->LLM->TTS)."""

    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("boom")

    async def transcribe(self, pcm):
        raise self._exc


async def test_run_top_level_exception_is_recorded_and_run_ends(tmp_path, monkeypatch):
    # An unexpected error inside the STT->LLM->TTS block (here: STT raises) is caught
    # by the pipeline's top-level handler: it emits a VOICE_ASSISTANT_ERROR
    # (code="server_error", message carrying the error text) AND a RUN_END, and the
    # run is still recorded once as an error attributed to the "pipeline" stage.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)
    pipeline.rt.stt_backends["fake"] = RaisingSttBackend(RuntimeError("boom"))

    await say(pipeline)

    types = types_of(events)
    # The error event is emitted before RUN_END, and RUN_END still fires.
    assert StageEvent.ERROR in types
    assert StageEvent.RUN_END in types
    assert types.index(StageEvent.ERROR) < types.index(StageEvent.RUN_END)
    data = dict(events)
    assert data[StageEvent.ERROR]["code"] == "server_error"
    assert "boom" in data[StageEvent.ERROR]["message"]
    assert_all_str(events)

    # The run is recorded once as a pipeline-stage error carrying the message.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "pipeline"
    assert "boom" in rec["error_text"]


class InsertRaisingRunsStore(FakeRunsStore):
    """RunsStore double whose insert() always raises: a recording failure must
    never break the run or leak out of it."""

    def insert(self, rec):
        raise RuntimeError("insert boom")


async def test_run_record_insert_failure_is_isolated(tmp_path, monkeypatch):
    # If runs_store.insert() raises, the run must still complete the full happy path
    # (all events incl. RUN_END) and the run must not propagate the error. The run_id
    # is never obtained, so no FINALIZED broadcast happens — only the live partials.
    patch_llm(monkeypatch, reply="готово")
    store = InsertRaisingRunsStore()
    hub = FakeRunEvents()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await say(pipeline)  # must not raise

    # Full happy path still emitted, ending in RUN_END.
    assert types_of(events) == FULL_SEQUENCE
    # Live partials still stream, but with no run_id there is no finalized broadcast:
    # every payload the hub saw is a live partial (id is None / live flagged).
    assert all(b["run"].get("live") for b in hub.broadcasts)
    assert not any(b["run"].get("id") is not None for b in hub.broadcasts)


class PutAudioRaisingRunsStore(FakeRunsStore):
    """RunsStore double: insert() succeeds, put_audio() raises. A utterance-audio
    store failure must not break the run or the broadcast."""

    def put_audio(self, run_id, wav, keep):
        raise RuntimeError("put_audio boom")


async def test_put_audio_failure_is_isolated(tmp_path, monkeypatch):
    # insert() succeeds but put_audio() raises: the run is still recorded exactly
    # once, RUN_END is emitted, and the broadcast still fires — but with has_audio
    # falsy because the audio store failed.
    patch_llm(monkeypatch, reply="готово")
    store = PutAudioRaisingRunsStore()
    hub = FakeRunEvents()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await say(pipeline)  # must not raise

    assert StageEvent.RUN_END in types_of(events)
    # Recorded exactly once despite the audio-store failure.
    assert len(store.records) == 1
    # The finalized broadcast (last one, after the live partials) still fires, but
    # has_audio is falsy (the audio never stored).
    payload = hub.broadcasts[-1]
    assert not payload["run"]["has_audio"]
    assert not payload["run"].get("live")


class BroadcastRaisingRunEvents(FakeRunEvents):
    """run-events hub whose broadcast() always raises: a slow/failing WS consumer
    must never affect the run or leak out of it."""

    async def broadcast(self, payload):
        self.broadcasts.append(payload)
        raise RuntimeError("broadcast boom")


async def test_broadcast_failure_is_isolated(tmp_path, monkeypatch):
    # A failing broadcast (slow/broken WS consumer) must not break the run: the run is
    # still recorded exactly once and the run ends without raising.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    hub = BroadcastRaisingRunEvents()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await say(pipeline)  # must not raise

    assert StageEvent.RUN_END in types_of(events)
    assert len(store.records) == 1
    # Broadcasts (live partials + the finalized one) were attempted and each raised,
    # but none broke the run.
    assert len(hub.broadcasts) >= 1


# --- early "filler" line (slow-tool placeholder, played ahead of the reply) -------


class EchoTts(TtsBackend):
    """TTS double whose 'audio' is the UTF-8 text, so the played clips tell apart."""

    def __init__(self):
        self.calls = []  # (text, lang) per synthesize

    async def synthesize(self, text, lang="ru"):
        self.calls.append((text, lang))
        return ("audio/x-test", text.encode())


def played_texts(pipeline):
    """Texts of the clips played so far through the fake play hook (EchoTts)."""
    return [audio.decode() for audio, _ in pipeline.play.calls]


async def test_filler_fires_when_any_tool_in_round_is_slow(tmp_path, monkeypatch):
    # The pipeline asks the HUB which tools are slow (the owning source declares
    # it; no name guessing here). A round mixing a fast and a slow tool still
    # fires the filler — any slow tool in the batch makes the user wait.
    patch_llm_with_filler(
        monkeypatch, tool_names=["set_light", "search_events"],
        content="Секунду…", reply="готово",
    )
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="свет и календарь",
                                tts_backend=EchoTts())

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert sorted(played_texts(pipeline)) == sorted([clean_llm_output("Секунду…"), "готово"])


# --- pipeline.speak() (public proactive-speech entry) -------------------------

async def test_speak_synthesizes_and_plays(tmp_path, monkeypatch):
    # speak() (the filler, external callers): synthesize via the current tts stage
    # and hand the clip to the device's play hook.
    tts = EchoTts()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, tts_backend=tts)

    await pipeline.speak("привет")

    assert tts.calls == [("привет", "ru")]
    assert pipeline.play.calls == [("привет".encode(), "audio/x-test")]


async def test_speak_without_speaker_synthesizes_nothing(tmp_path, monkeypatch):
    # A device without a speaker gets no TTS at all.
    tts = EchoTts()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, tts_backend=tts)
    pipeline.has_speaker = False

    await pipeline.speak("привет")

    assert tts.calls == []
    assert pipeline.play.calls == []


async def test_speak_propagates_synthesis_errors(tmp_path, monkeypatch):
    # speak() raises on failure — isolation is the CALLER's job (the filler wraps
    # it in try/except).
    class _BoomTts:
        async def synthesize(self, text, lang="ru"):
            raise RuntimeError("tts down")

    pipeline, _ = make_pipeline(tmp_path, monkeypatch, tts_backend=_BoomTts())
    with pytest.raises(RuntimeError, match="tts down"):
        await pipeline.speak("hi")
    assert pipeline.play.calls == []


def patch_llm_with_filler(monkeypatch, *, tool_names, content, reply="готово", times=1):
    """Stub the LLM stage with a fake that invokes the passed on_filler callback
    `times` times (with the same content + tool_names) before returning `reply`.

    Lets a test drive the pipeline's _speak_filler policy through the real _run
    wiring without a live LLM backend."""

    class FakeLlmStage:
        def __init__(self, backend, hub, cfg):
            pass

        async def respond(self, req, *, on_filler=None, has_next=False):
            if on_filler is not None:
                for _ in range(times):
                    await on_filler(content, tool_names)
            return make_llm_result(reply)

    monkeypatch.setattr("src.llm.LlmStage", FakeLlmStage)


async def _drain_filler_tasks(pipeline):
    """Await any fire-and-forget filler tasks so a test can assert on their side
    effects (the run does NOT await them itself)."""
    import asyncio
    if pipeline._filler_tasks:
        await asyncio.gather(*list(pipeline._filler_tasks))


async def test_filler_played_for_slow_tool(tmp_path, monkeypatch):
    # A slow tool round carrying spoken content -> the filler is synthesized and
    # played; the run record carries filler_text / t_filler.
    patch_llm_with_filler(
        monkeypatch, tool_names=["search_events"], content="Щас гляну…", reply="готово",
    )
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="что у меня в календаре", runs_store=store,
        tts_backend=EchoTts(),
    )

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert clean_llm_output("Щас гляну…") in played_texts(pipeline)
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["filler_text"] == clean_llm_output("Щас гляну…")
    assert rec["t_filler"] is not None and rec["t_filler"] >= 0


async def test_no_filler_for_fast_tool(tmp_path, monkeypatch):
    # A fast smart-home tool (set_light) -> only the reply is played, no filler recorded.
    patch_llm_with_filler(
        monkeypatch, tool_names=["set_light"], content="Ну изволь, включил.", reply="готово",
    )
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, tts_backend=EchoTts(),
    )

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert played_texts(pipeline) == ["готово"]
    assert store.records[0]["filler_text"] == ""
    assert store.records[0]["t_filler"] is None


async def test_filler_dedup_at_most_once_per_run(tmp_path, monkeypatch):
    # Two callback invocations in one run -> the filler plays once (speak-at-most-once).
    patch_llm_with_filler(
        monkeypatch, tool_names=["google"], content="Щас погуглю…", reply="готово", times=2,
    )
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="погугли погоду",
                                tts_backend=EchoTts())

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert played_texts(pipeline).count(clean_llm_output("Щас погуглю…")) == 1


async def test_no_filler_without_speaker(tmp_path, monkeypatch):
    # A device without a speaker: the slow-tool filler is a no-op (nothing scheduled,
    # nothing recorded) and the reply is not synthesized either.
    patch_llm_with_filler(
        monkeypatch, tool_names=["search_events"], content="Щас гляну…", reply="готово",
    )
    store = FakeRunsStore()
    tts = EchoTts()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="что в календаре", runs_store=store, tts_backend=tts,
    )
    pipeline.has_speaker = False

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert pipeline._filler_tasks == set()
    assert store.records[0]["filler_text"] == ""
    assert store.records[0]["t_filler"] is None
    assert tts.calls == []
    assert pipeline.play.calls == []
    # The reply text still reaches the device.
    assert dict(events)[StageEvent.TTS_START] == {"text": "готово"}


async def test_no_filler_when_content_blank_after_processing(tmp_path, monkeypatch):
    # Content that reduces to empty after clean_llm_output (whitespace / stripped
    # tags) -> no filler even for a slow tool.
    patch_llm_with_filler(
        monkeypatch, tool_names=["search_events"], content="   ", reply="готово",
    )
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="что в календаре", runs_store=store,
        tts_backend=EchoTts(),
    )

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    assert played_texts(pipeline) == ["готово"]
    assert store.records[0]["filler_text"] == ""


# --- end-of-phrase "ack" chime (server-side «блям» at VAD finalize) ---------------


async def _drain_ack_tasks(pipeline):
    """Await any fire-and-forget end-of-phrase ack tasks so a test can assert on their
    side effects (the run does NOT await them itself)."""
    import asyncio
    if pipeline._ack_tasks:
        await asyncio.gather(*list(pipeline._ack_tasks))


def wav_plays(pipeline):
    """The WAV clips played so far (the chime; the fake TTS reply is not WAV)."""
    return [audio for audio, mime in pipeline.play.calls if mime == "audio/wav"]


async def test_ack_scheduled_on_phrase_end(tmp_path, monkeypatch):
    # End-of-phrase ack enabled + a device speaker -> the synthesized chime (WAV) is
    # played once, alongside the reply.
    patch_llm(monkeypatch, reply="готово")
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", ack=True)

    await say(pipeline)
    await _drain_ack_tasks(pipeline)

    [chime] = wav_plays(pipeline)
    with wave.open(io.BytesIO(chime), "rb") as w:
        assert w.getnframes() > 0


async def test_ack_not_scheduled_when_disabled(tmp_path, monkeypatch):
    # ack.enabled = False -> no ack task scheduled, no chime played.
    patch_llm(monkeypatch, reply="готово")
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", ack=True)
    pipeline.rt.core.ack.enabled = False

    await say(pipeline)
    await _drain_ack_tasks(pipeline)

    assert pipeline._ack_tasks == set()
    assert wav_plays(pipeline) == []


async def test_ack_not_scheduled_without_speaker(tmp_path, monkeypatch):
    # A device without a speaker never gets playback, the chime included.
    patch_llm(monkeypatch, reply="готово")
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", ack=True)
    pipeline.has_speaker = False

    await say(pipeline)
    await _drain_ack_tasks(pipeline)

    assert pipeline._ack_tasks == set()
    assert pipeline.play.calls == []


async def test_ack_uses_configured_sound_file_when_present(tmp_path, monkeypatch):
    # A configured WAV sound_path that exists on disk is played verbatim (the
    # delivery boundary converts it for the device).
    patch_llm(monkeypatch, reply="готово")
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", ack=True)
    blyam = tmp_path / "blyam.wav"
    with wave.open(str(blyam), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(8000)
        wf.writeframes(b"\x00\x00" * 256)  # a few frames of silence
    pipeline.rt.core.ack.sound_path = str(blyam)

    await say(pipeline)
    await _drain_ack_tasks(pipeline)

    assert wav_plays(pipeline) == [blyam.read_bytes()]


async def test_ack_is_fire_and_forget_does_not_delay_finalize(tmp_path, monkeypatch):
    # The end-of-phrase ack must be truly fire-and-forget: even if its play is STUCK
    # (the device never reports it played), the run's finalize path must STILL
    # complete without waiting for the ack. If the run awaited the ack, the calls
    # below would hang and hit the timeout.
    import asyncio

    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", ack=True)

    stuck = asyncio.Event()  # never set -> the chime's play blocks forever
    chime_plays = []
    reply_play = pipeline.play

    async def play(mime, audio):
        if mime == "audio/wav":
            chime_plays.append(audio)
            await stuck.wait()
        else:
            await reply_play(mime, audio)

    pipeline.play = play

    await asyncio.wait_for(say(pipeline), timeout=5.0)

    # The run finalized fully (RUN_END emitted, full happy path) without the ack.
    assert types_of(events) == FULL_SEQUENCE

    # Let the scheduled (still-stuck) ack task reach its blocked play call.
    await asyncio.sleep(0)
    assert len(chime_plays) == 1
    # Still pending, proving the run did not await it; cancel so nothing leaks.
    assert pipeline._ack_tasks
    for task in list(pipeline._ack_tasks):
        task.cancel()


# --- mic channel selection + pre-STT conditioning (core.vad.mic_channel / mic_normalize / mic_highpass) ---

def test_normalize_peak_boosts_quiet_signal_to_target():
    # A quiet signal (peak 1000) is scaled up so its peak reaches the -3 dBFS target
    # (32767 * 10**(-3/20) ≈ 23197), within a loose tolerance.
    pcm = np.array([1000, -1000], dtype="<i2").tobytes()
    out = np.frombuffer(normalize_peak(pcm), "<i2")
    target = 32767.0 * (10.0 ** (-3.0 / 20.0))
    assert abs(int(np.max(np.abs(out))) - target) < 200


def test_normalize_peak_scales_down_loud_signal():
    # A near-full-scale signal must be scaled DOWN to the target peak, not left alone.
    pcm = np.array([32000, -32000], dtype="<i2").tobytes()
    out = np.frombuffer(normalize_peak(pcm), "<i2")
    assert int(np.max(np.abs(out))) < 32000


def test_normalize_peak_silence_is_identity():
    pcm = np.zeros(8, dtype="<i2").tobytes()
    assert normalize_peak(pcm) == pcm


def test_normalize_peak_empty_safe():
    assert normalize_peak(b"") == b""


def test_normalize_peak_max_gain_caps_boost():
    # A very quiet signal (peak 10) can't be boosted past input * max_gain (30).
    pcm = np.array([10, -10], dtype="<i2").tobytes()
    out = np.frombuffer(normalize_peak(pcm), "<i2")
    assert int(np.max(np.abs(out))) <= 10 * 30


def test_highpass_drives_dc_to_zero():
    # A pure DC offset (constant signal) has its mean driven to ~0 after the high-pass.
    pcm = np.full(2048, 5000, dtype="<i2").tobytes()
    out = np.frombuffer(highpass(pcm), "<i2").astype(np.float32)
    assert abs(float(np.mean(out))) < 500  # much smaller than the 5000 DC offset


def test_highpass_preserves_midband_tone():
    # A 1000 Hz tone (well above the 80 Hz cutoff) keeps most of its energy.
    t = np.arange(2048) / SAMPLE_RATE
    sig = (10000 * np.sin(2 * np.pi * 1000 * t)).astype("<i2")
    pcm = sig.tobytes()
    rms_before = float(np.sqrt(np.mean(sig.astype(np.float32) ** 2)))
    out = np.frombuffer(highpass(pcm), "<i2").astype(np.float32)
    rms_after = float(np.sqrt(np.mean(out ** 2)))
    assert rms_after > rms_before * 0.7  # most of the energy survives


def test_highpass_empty_safe():
    assert highpass(b"") == b""


async def test_mic_channel1_uses_second_stream(tmp_path, monkeypatch):
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.core.vad.mic_channel = 1  # live config: take the less-processed channel
    pipeline.on_wake("cid", None)
    await pipeline.on_audio(b"\x00\x00", b"\x11\x11")  # ch0 zeros, ch1 = 0x1111
    assert pipeline.preroll_pcm == b"\x11\x11"


async def test_mic_channel0_uses_first_stream(tmp_path, monkeypatch):
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.core.vad.mic_channel = 0
    pipeline.on_wake("cid", None)
    await pipeline.on_audio(b"\x22\x22", b"\x11\x11")
    assert pipeline.preroll_pcm == b"\x22\x22"


async def test_mic_channel1_falls_back_to_first_when_no_second_stream(tmp_path, monkeypatch):
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    pipeline.core.vad.mic_channel = 1
    pipeline.on_wake("cid", None)
    await pipeline.on_audio(b"\x22\x22", None)  # device sent only one channel
    assert pipeline.preroll_pcm == b"\x22\x22"


# --- stereo diagnostic WAV (stored run audio carries both mic channels) ---

def _parse_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnchannels(), w.getnframes(), w.readframes(w.getnframes())


def test_pcm_to_wav_bytes_mono_without_second_channel():
    # No second arg -> the WAV stays mono, exactly as before (manual capture path).
    pcm = np.array([1, -2, 3, -4], dtype="<i2").tobytes()
    nch, nframes, frames = _parse_wav(pcm_to_wav_bytes(pcm))
    assert nch == 1
    assert nframes == 4
    assert frames == pcm


def test_pcm_to_wav_bytes_stereo_interleaves_channels():
    # Equal-length channels -> 2-channel WAV with left == pcm, right == pcm2.
    left = np.array([100, -200, 300, -400], dtype="<i2")
    right = np.array([5, -6, 7, -8], dtype="<i2")
    nch, nframes, frames = _parse_wav(pcm_to_wav_bytes(left.tobytes(), right.tobytes()))
    assert nch == 2
    assert nframes == 4
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, 2)
    assert (samples[:, 0] == left).all()
    assert (samples[:, 1] == right).all()


def test_pcm_to_wav_bytes_stereo_pads_shorter_channel():
    # Unequal lengths -> the shorter channel is zero-padded; the frame count
    # matches the LONGER channel's sample count.
    left = np.array([1, 2, 3, 4], dtype="<i2")
    right = np.array([9, 8], dtype="<i2")
    nch, nframes, frames = _parse_wav(pcm_to_wav_bytes(left.tobytes(), right.tobytes()))
    assert nch == 2
    assert nframes == 4
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, 2)
    assert (samples[:, 0] == left).all()
    assert (samples[:, 1] == [9, 8, 0, 0]).all()


async def test_stored_run_audio_is_stereo_when_second_channel_streamed(tmp_path, monkeypatch):
    # An end-to-end run with both mic streams stores a STEREO diagnostic WAV:
    # left = the selected (STT) channel, right = the other raw channel.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)
    pipeline.core.vad.mic_channel = 0

    other = b"\x03\x04" * (len(PHRASE) // 2)
    await say(pipeline, other)

    assert len(store.audio_calls) == 1
    nch, nframes, frames = _parse_wav(store.audio_calls[0][1])
    assert nch == 2
    assert nframes == len(PHRASE) // 2
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, 2)
    assert samples[:, 0].tobytes() == PHRASE  # left: what STT received
    assert samples[:, 1].tobytes() == other   # right: the other raw channel


async def test_stored_run_audio_swaps_the_channels_for_mic_channel1(tmp_path, monkeypatch):
    # With mic_channel=1 the second stream is what STT gets (left) and the first one
    # the diagnostic channel (right).
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)
    pipeline.core.vad.mic_channel = 1

    other = b"\x03\x04" * (len(PHRASE) // 2)
    await say(pipeline, other)

    nch, _, frames = _parse_wav(store.audio_calls[0][1])
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, 2)
    assert (nch, samples[:, 0].tobytes(), samples[:, 1].tobytes()) == (2, other, PHRASE)


async def test_stored_run_audio_stays_mono_without_second_channel(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await say(pipeline)

    assert len(store.audio_calls) == 1
    nch, nframes, frames = _parse_wav(store.audio_calls[0][1])
    assert nch == 1
    assert nframes == len(PHRASE) // 2
    assert frames == PHRASE


# --- pipeline-isolation coverage gaps (R1/R3/R4/R5 audit) -------------------------


async def test_llm_non_stage_exception_is_recorded_and_never_spoken(tmp_path, monkeypatch):
    # A NON-StageError escaping the LLM call (a bug, not a classified stage failure)
    # bypasses the LLM fallback branch and lands in the pipeline's top-level handler.
    # Mirrors test_run_top_level_exception_is_recorded_and_run_ends (the STT variant):
    # ERROR(code="server_error") before RUN_END, no TTS at all, and the raw error
    # text is never synthesized for the user's ears.
    patch_llm(monkeypatch, error=ValueError("oops"))
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store,
    )

    await say(pipeline)

    types = types_of(events)
    # No TTS stage was ever entered: the raw error text must not be spoken.
    assert StageEvent.TTS_START not in types
    # The error event is emitted before RUN_END, and RUN_END still fires.
    assert StageEvent.ERROR in types
    assert StageEvent.RUN_END in types
    assert types.index(StageEvent.ERROR) < types.index(StageEvent.RUN_END)
    data = dict(events)
    assert data[StageEvent.ERROR]["code"] == "server_error"
    assert "oops" in data[StageEvent.ERROR]["message"]
    assert_all_str(events)

    # Recorded once as a pipeline-stage error; nothing served to the audio cache.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "pipeline"
    assert "oops" in rec["error_text"]
    assert pipeline.play.calls == []


async def test_run_completes_headless_when_send_event_is_none(tmp_path, monkeypatch):
    # R4 contract: the pipeline is transport-neutral. With NO injected send_event
    # (headless: no device/event transport bound) a full run still completes and is
    # recorded — emission is guarded, never required.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store,
    )
    pipeline.send_event = None  # drop the recording transport installed by the harness

    await say(pipeline)  # must not raise anywhere along the run

    # The run completed fully and was recorded exactly once.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "ok"
    assert rec["stt_text"] == "включи свет"
    assert rec["llm_text"] == "готово"
    # The reply audio was still synthesized and served despite no event transport.
    assert pipeline.play.calls == [(b"MP3", "audio/mpeg")]


async def test_intent_end_payload_carries_conversation_id(tmp_path, monkeypatch):
    # INTENT_END carries the run's conversation id as a flat str->str payload (the
    # transport-neutral StageEvent contract).
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")

    await say(pipeline)

    assert dict(events)[StageEvent.INTENT_END] == {"conversation_id": "cid"}
    assert_all_str(events)


async def test_filler_think_block_stripped_before_announce(tmp_path, monkeypatch):
    # The filler content goes through clean_llm_output before being spoken/recorded:
    # a <think> reasoning block is stripped, only the spoken line survives.
    patch_llm_with_filler(
        monkeypatch, tool_names=["search_events"],
        content="<think>план</think>Щас гляну", reply="готово",
    )
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="что у меня в календаре", runs_store=store,
        tts_backend=EchoTts(),
    )

    await say(pipeline)
    await _drain_filler_tasks(pipeline)

    # The played and the recorded filler text are exactly the cleaned line.
    assert "Щас гляну" in played_texts(pipeline)
    assert store.records[0]["filler_text"] == "Щас гляну"


async def test_llm_reply_reaches_tts_verbatim(tmp_path, monkeypatch):
    # R3 canonical text contract: the LLM reply (with '+' stress marks and units)
    # is handed to the TTS backend VERBATIM — any engine-specific adaptation is the
    # backend's own job, never the pipeline's.
    class _RecordingTts(FakeTtsBackend):
        def __init__(self):
            super().__init__(mime="audio/mpeg", audio=b"MP3")
            self.calls = []  # (text, lang) per synthesize

        async def synthesize(self, text, lang="ru"):
            self.calls.append((text, lang))
            return await super().synthesize(text, lang)

    reply = "прив+ет, сейчас 5 °C"
    patch_llm(monkeypatch, reply=reply)
    tts = _RecordingTts()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="какая погода", tts_backend=tts,
    )

    await say(pipeline)

    # The exact canonical string reached synthesize(); the event echoes it too.
    assert tts.calls == [(reply, "ru")]
    assert dict(events)[StageEvent.TTS_START] == {"text": reply}
    assert_all_str(events)


# --- pipeline <-> LlmStage wiring (R7 contract) -----------------------------------


async def test_llm_request_carries_assembled_prompt_and_device(tmp_path, monkeypatch):
    # The orchestrator assembles the system prompt (build_system_prompt over the
    # active PromptStore profile) and names the originating device on the LlmRequest.
    # Catches: an empty/missing system prompt silently stripping the assistant of
    # all of its instructions.
    seen = []  # the LlmRequest of each respond() call

    class CapturingLlmStage:
        def __init__(self, backend, hub, cfg):
            pass

        async def respond(self, req, *, on_filler=None, has_next=False):
            seen.append(req)
            return make_llm_result("готово")

    monkeypatch.setattr("src.llm.LlmStage", CapturingLlmStage)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")

    await say(pipeline)

    assert len(seen) == 1
    req = seen[0]
    # The prompt file body written by make_pipeline reached the stage...
    assert "PROMPT BODY" in req.system_prompt
    # ...with the date placeholder expanded, not passed through verbatim.
    assert "<<<<<TDW>>>>>" not in req.system_prompt
    assert req.device == pipeline.name
    assert req.user_text == "включи свет"


async def test_llm_backend_hot_swap_applies_on_next_run(tmp_path, monkeypatch):
    # The pipeline constructs a fresh LlmStage per run from rt.llm_backends, so
    # swapping the backend on the runtime (what the reconfigurator does on an
    # llm.* change) takes effect on the NEXT run without rebuilding the pipeline
    # — mirrors test_vad_backend_hot_swap_applies_on_next_run.
    backends = []  # the backend each constructed stage received

    class RecordingLlmStage:
        def __init__(self, backend, hub, cfg):
            backends.append(backend)

        async def respond(self, req, *, on_filler=None, has_next=False):
            return make_llm_result("готово")

    monkeypatch.setattr("src.llm.LlmStage", RecordingLlmStage)
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    old = pipeline.rt.llm_backends["fake"]

    await say(pipeline)
    assert backends == [old]

    # Hot-swap: the next run's stage is constructed with the NEW backend object.
    new = object()
    pipeline.rt.llm_backends["fake"] = new
    await say(pipeline)
    assert backends == [old, new]


async def test_llm_stage_error_without_partial_attribute(tmp_path, monkeypatch):
    # A StageError WITHOUT .partial set (the attribute is optional on the
    # contract): the error path must read it defensively (getattr, never
    # e.partial directly), leave the observability fields at their empty
    # defaults, and still speak the configured fallback phrase.
    err = StageError("llm", "boom")
    assert not hasattr(err, "partial")
    patch_llm(monkeypatch, error=err)
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store,
    )

    await say(pipeline)

    # The run completed cleanly: RUN_END fired, no raw exception escaped (an
    # AttributeError on e.partial would land in the top-level ERROR branch).
    types = types_of(events)
    assert StageEvent.RUN_END in types
    assert StageEvent.ERROR not in types
    # The configured fallback phrase (never the raw error text) went to TTS.
    assert dict(events)[StageEvent.TTS_START] == {"text": LlmConfig().reply_error}
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "LLM"
    assert rec["model"] is None
    assert rec["tokens"] is None
    assert rec["rounds"] == []
    assert rec["request"] is None


# --- pipeline.run_text() (text requests from the agent MCP server) ------------


async def test_run_text_happy_path(tmp_path, monkeypatch):
    # A text run does the full LLM turn, appends context, plays the reply on the
    # device, records the run (reason="text", request in stt_text) and broadcasts
    # the finalized summary — all WITHOUT any stage events.
    from src import context

    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store, run_events=hub,
    )
    pipeline.play.stats = {"outcome": "played", "dry": 1, "gap_ms": 30}

    result = await pipeline.run_text("включи свет")

    assert result == {
        "reply": "готово", "result": "ok",
        "error_stage": None, "error_text": None,
    }
    # Recorded once, with the text-run shape.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["reason"] == "text"
    assert rec["stt_text"] == "включи свет"
    assert rec["llm_text"] == "готово"
    assert rec["t_vad"] == 0 and rec["t_stt"] == 0
    assert rec["t_total"] >= 0
    assert rec["error_stage"] is None
    assert (rec["play_dry"], rec["play_gap_ms"]) == (1, 30)
    # Context appended like a normal conversation turn.
    assert context.load_context(pipeline._context_path) == [
        {"role": "user", "content": "включи свет"},
        {"role": "assistant", "content": "готово"},
    ]
    # The reply was spoken: synthesized and played.
    assert pipeline.play.calls == [(b"MP3", "audio/mpeg")]
    # No stage events for a text run (no voice session).
    assert events == []
    # Exactly one finalized broadcast, after the insert (id echoes the store).
    assert len(hub.broadcasts) == 1
    assert hub.broadcasts[0]["type"] == "run"
    assert hub.broadcasts[0]["run"]["id"] == 1
    assert hub.broadcasts[0]["run"]["result"] == "ok"


async def test_run_text_llm_error_speaks_fallback_and_skips_context(tmp_path, monkeypatch):
    # An LLM StageError on a text run mirrors _run: the configured fallback
    # phrase is the reply, the run records error/LLM, and the failed turn is
    # NOT appended to the conversation context.
    from src import context

    appended = []
    monkeypatch.setattr(context, "append_context", lambda *a, **k: appended.append(a))
    patch_llm(monkeypatch, error=StageError("llm", "boom"))
    store = FakeRunsStore()
    pipeline, events = make_pipeline(tmp_path, monkeypatch, runs_store=store)

    result = await pipeline.run_text("привет")

    assert result["result"] == "error"
    assert result["error_stage"] == "LLM"
    assert result["error_text"] == "boom"
    assert result["reply"] == LlmConfig().reply_error
    assert appended == []
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "LLM"
    assert rec["llm_text"] == LlmConfig().reply_error
    # The fallback phrase is still spoken (TTS ran).
    assert pipeline.play.calls == [(b"MP3", "audio/mpeg")]
    assert events == []


async def test_run_text_speak_false_skips_tts(tmp_path, monkeypatch):
    # speak=False: the LLM turn runs but nothing is synthesized or announced.
    patch_llm(monkeypatch, reply="готово")
    tts_calls = []

    class SpyTts:
        async def synthesize(self, text, lang="ru"):
            tts_calls.append(text)
            return ("audio/mpeg", b"MP3")

    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, tts_backend=SpyTts(), runs_store=store,
    )

    result = await pipeline.run_text("включи свет", speak=False)

    assert result["result"] == "ok"
    assert result["reply"] == "готово"
    assert tts_calls == []
    assert pipeline.play.calls == []
    assert store.records[0]["t_tts"] == 0
    assert events == []


async def test_run_text_tts_failure_still_returns_reply(tmp_path, monkeypatch):
    # A TTS failure marks the run error/TTS but the text reply still comes back.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, tts_backend=RaisingTtsBackend(), runs_store=store,
    )

    result = await pipeline.run_text("включи свет")

    assert result["reply"] == "готово"
    assert result["result"] == "error"
    assert result["error_stage"] == "TTS"
    assert result["error_text"] == "tts boom"
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "TTS"
    assert rec["llm_text"] == "готово"
    assert events == []


async def test_run_text_without_store_or_hub(tmp_path, monkeypatch):
    # runs_store=None / run_events=None: the call still works and returns the reply.
    patch_llm(monkeypatch, reply="ок")
    pipeline, events = make_pipeline(tmp_path, monkeypatch)

    result = await pipeline.run_text("привет", speak=False)

    assert result["reply"] == "ок"
    assert result["result"] == "ok"
    assert events == []


# --- streaming TTS --------------------------------------------------------------


async def test_streaming_tts_midstream_error_records_tts_stage(tmp_path, monkeypatch):
    # A generator failing AFTER playback started (mid-stream): the error surfaces
    # from the play hook, the run still completes cleanly (RUN_END, no top-level
    # ERROR) and the record blames TTS.
    class _ExplodingStreamTts(TtsBackend):
        async def synthesize(self, text, lang="ru"):
            return ("audio/mpeg", b"")

        async def synthesize_stream(self, text, lang="ru"):
            async def _gen():
                yield b"part1"
                raise RuntimeError("stream boom")

            return ("audio/mpeg", _gen())

    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет",
        tts_backend=_ExplodingStreamTts(), runs_store=store,
    )

    await say(pipeline)

    types = types_of(events)
    assert StageEvent.RUN_END in types
    assert StageEvent.ERROR not in types  # handled mid-stream, not a pipeline crash
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "TTS"
    assert "stream boom" in rec["error_text"]
    # The aborted clip was not recorded as played, nor stored.
    assert pipeline.play.calls == []
    assert store.tts_audio_calls == []


async def test_streaming_tts_released_when_play_fails_before_reading(tmp_path, monkeypatch):
    # The device cannot play (it went offline) and never reads the stream: the
    # backend's already-open response must still be released.
    released = []

    class _OpenStreamTts(TtsBackend):
        async def synthesize(self, text, lang="ru"):
            return ("audio/wav", b"")

        async def synthesize_stream(self, text, lang="ru"):
            async def _gen():
                yield b"\x01\x00"

            async def _release():
                released.append(True)

            return (pcm_mime(16000), OwnedChunks(_gen(), _release))

    async def offline_play(mime, audio):
        raise RuntimeError("dev cannot play: offline")

    patch_llm(monkeypatch, reply="готово")
    pipeline, _events = make_pipeline(tmp_path, monkeypatch, tts_backend=_OpenStreamTts())
    pipeline.play = offline_play

    await say(pipeline)

    assert released == [True]


# --- streaming STT disconnect cleanup (Pipeline.on_disconnect) ---------------------


class _FakeStreamSession:
    """Streaming STT session double: records whether aclose() was called so the
    on_disconnect tests can assert the lingering gRPC session was released."""

    def __init__(self):
        self.closed = False

    def feed(self, pcm):
        pass

    async def finish(self):
        return ""

    async def aclose(self):
        self.closed = True


class _StreamingSttBackend:
    """STT double whose open_stream() returns a fake live session, so the pipeline
    opens a streaming STT session in on_wake (the streaming-capable path)."""

    def __init__(self):
        self.session = _FakeStreamSession()

    async def transcribe(self, pcm):
        return ""

    def open_stream(self):
        return self.session


async def test_on_disconnect_closes_unfinalized_streaming_session(tmp_path, monkeypatch):
    # Speaker vanished mid-utterance: on_wake opened a streaming session but the run
    # was never finalized. on_disconnect() must close that session and drop the refs so
    # the gRPC stream/channel don't linger until the next run.
    stt = _StreamingSttBackend()
    pipeline, _events = make_pipeline(tmp_path, monkeypatch, stt_backend=stt)
    install_fake_vad(pipeline, [False])  # no endpoint -> run never finalizes
    pipeline.on_wake("cid", None)

    await pipeline.on_disconnect()
    assert stt.session.closed is True


async def test_on_disconnect_skips_finalized_run(tmp_path, monkeypatch):
    # A FINALIZED run owns the streaming session and closes it itself in _run's
    # finally; on_disconnect() must NOT touch it (avoid racing finish() vs aclose()).
    stt = _StreamingSttBackend()
    pipeline, _events = make_pipeline(tmp_path, monkeypatch, stt_backend=stt)
    install_phrase_vad(pipeline)
    pipeline.on_wake("cid", None)
    # The phrase ends: the run is claimed, its run task not started yet.
    await pipeline.on_audio(PHRASE)

    await pipeline.on_disconnect()
    assert stt.session.closed is False
    await settle(pipeline)
    assert stt.session.closed is True  # by the run itself


# --- streaming STT open: non-streaming backend vs genuine open failure -----------


class _RaisingStreamSttBackend:
    """STT double that DOES expose open_stream() but whose open_stream() raises —
    a streaming-capable backend that fails during live channel setup. Records the
    call so the test can prove the run took the warn-and-fall-back-to-batch path."""

    def __init__(self):
        self.open_called = False

    async def transcribe(self, pcm):
        return "включи свет"

    def open_stream(self):
        self.open_called = True
        raise RuntimeError("channel setup failed")


@pytest.mark.parametrize("stt", [FakeSttBackend("включи свет"), _RaisingStreamSttBackend()])
async def test_a_backend_without_a_streaming_session_is_transcribed_in_one_shot(
        tmp_path, monkeypatch, stt):
    # A duck-typed backend WITHOUT open_stream() (a non-streaming backend) has no
    # streaming session — NOT an open failure to warn about; one whose open_stream()
    # raises is a GENUINE failure (the warn path), swallowed. Either way the run goes on
    # via the batch transcribe() route.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_backend=stt)

    await say(pipeline)
    assert getattr(stt, "open_called", True) is True
    assert dict(events)[StageEvent.STT_END] == {"text": "включи свет"}


# --- TTS variants: fallback before the speaker has any audio ---------------------


class _FailingTts(TtsBackend):
    """Fails to start (the default stream adapter calls synthesize()) after `delay`
    seconds; counts its calls."""

    def __init__(self, message, delay=0.0):
        self.message = message
        self.delay = delay
        self.calls = 0

    async def synthesize(self, text, lang="ru"):
        import asyncio

        self.calls += 1
        await asyncio.sleep(self.delay)
        raise RuntimeError(self.message)


class _ChunkIter:
    """Async chunk iterator double: yields `chunks`, then raises `error` (if any);
    records aclose()."""

    def __init__(self, chunks=(), error=None):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class _IterTts(TtsBackend):
    """Streaming TTS double handing out a prepared _ChunkIter."""

    def __init__(self, it):
        self.it = it
        self.calls = 0

    async def synthesize(self, text, lang="ru"):
        raise AssertionError("streaming path must not call buffered synthesize")

    async def synthesize_stream(self, text, lang="ru"):
        self.calls += 1
        return ("audio/mpeg", self.it)


def _tts_attempts(rec):
    return [(a["provider"], a["outcome"], a["error"])
            for a in rec["attempts"] if a["kind"] == "tts"]


async def test_tts_first_variant_fails_to_start_second_speaks(tmp_path, monkeypatch):
    # synthesize() raises, so the stream fails before any iterator exists: the next
    # variant speaks, and the run is a success whose t_tts counts from the first attempt.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _FailingTts("a down", delay=0.05), "b": FakeTtsBackend(audio=b"B")},
    )

    await say(pipeline)

    assert pipeline.play.calls == [(b"B", "audio/mpeg")]
    rec = store.records[0]
    assert (rec["result"], rec["error_stage"]) == ("ok", None)
    assert _tts_attempts(rec) == [("a", "error", "a down"), ("b", "spoke", None)]
    assert rec["t_tts"] >= 50


async def test_tts_stream_failing_on_first_chunk_switches_and_closes_it(tmp_path, monkeypatch):
    # The first variant's stream opens but fails on its first chunk: nothing was
    # handed to the device yet, so the next variant speaks; the failed iterator is
    # closed and never reached the device.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    broken = _ChunkIter(error=RuntimeError("first chunk boom"))
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _IterTts(broken), "b": _IterTts(_ChunkIter([b"b1", b"b2"]))},
    )

    await say(pipeline)

    assert broken.closed is True
    assert pipeline.play.calls == [(b"b1b2", "audio/mpeg")]
    rec = store.records[0]
    assert rec["result"] == "ok"
    assert _tts_attempts(rec) == [("a", "error", "first chunk boom"), ("b", "spoke", None)]


async def test_tts_failure_after_first_chunk_played_does_not_switch(tmp_path, monkeypatch):
    # The first variant started (its first chunk went to the device) and then fails
    # mid-stream: the device already plays it, so no other variant is tried.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    second = _IterTts(_ChunkIter([b"never"]))
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _IterTts(_ChunkIter([b"a1"], RuntimeError("mid boom"))),
                      "b": second},
    )
    played = []

    async def _player(mime, audio):
        async for chunk in audio:
            played.append(chunk)

    pipeline.play = _player

    await say(pipeline)

    assert played == [b"a1"]
    assert second.calls == 0
    rec = store.records[0]
    assert (rec["result"], rec["error_stage"], rec["error_text"]) == ("error", "TTS", "mid boom")
    assert _tts_attempts(rec) == [("a", "spoke", None)]


@pytest.mark.parametrize("path", ["voice", "text"])
async def test_time_to_speech_when_the_speech_breaks_off(tmp_path, monkeypatch, path):
    # The speech started (after a first variant that failed for 50 ms) and broke off
    # mid-stream: a voice answer keeps its time to speech, a text request's reply,
    # which never finished playing, keeps none.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _FailingTts("a down", delay=0.05),
                      "b": _IterTts(_ChunkIter([b"b1"], RuntimeError("mid boom")))},
    )

    async def _player(mime, audio):
        async for _ in audio:
            pass

    pipeline.play = _player
    if path == "voice":
        await say(pipeline)
    else:
        await pipeline.run_text("включи свет")

    rec = store.records[0]
    assert (rec["error_stage"], rec["error_text"]) == ("TTS", "mid boom")
    assert _tts_attempts(rec) == [("a", "error", "a down"), ("b", "spoke", None)]
    assert rec["t_tts"] >= 50 if path == "voice" else rec["t_tts"] == 0


async def test_an_answer_cut_off_mid_speech_keeps_its_time_to_speech(tmp_path, monkeypatch):
    # A stop word or a newer wake cancels the run while its answer plays: the record
    # still says how long the speech took to start.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _FailingTts("a down", delay=0.05), "b": FakeTtsBackend(audio=b"B")},
    )
    playing = asyncio.Event()

    async def _endless(mime, audio):
        playing.set()
        await asyncio.Event().wait()

    pipeline.play = _endless
    install_phrase_vad(pipeline)
    pipeline.on_wake("cid", None)
    await pipeline.on_audio(PHRASE)
    task = pipeline._run_task
    await playing.wait()
    pipeline.cancel_run()
    with pytest.raises(asyncio.CancelledError):
        await task

    rec = store.records[0]
    assert _tts_attempts(rec) == [("a", "error", "a down"), ("b", "spoke", None)]
    assert rec["t_tts"] >= 50


async def test_tts_all_variants_fail_reports_last_error(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _FailingTts("a down"), "b": _FailingTts("b down")},
    )

    await say(pipeline)

    assert pipeline.play.calls == []
    assert types_of(events)[-1] == StageEvent.RUN_END
    rec = store.records[0]
    assert (rec["result"], rec["error_stage"], rec["error_text"]) == ("error", "TTS", "b down")
    assert _tts_attempts(rec) == [("a", "error", "a down"), ("b", "error", "b down")]
    assert rec["t_tts"] == 0


async def test_tts_walks_only_the_enabled_variants_in_config_order(tmp_path, monkeypatch):
    # The order and the enabled set come from the config per utterance: a built
    # backend that is no enabled variant is never used, and an enabled variant whose
    # backend is not loaded counts as a failed attempt.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    disabled = _FailingTts("must not run")
    pipeline, _events = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"off": disabled, "b": FakeTtsBackend(audio=b"B")},
    )
    pipeline.rt.svc.variants = ["missing", "b"]

    await say(pipeline)

    assert disabled.calls == 0
    assert pipeline.play.calls == [(b"B", "audio/mpeg")]
    assert _tts_attempts(store.records[0]) == [
        ("missing", "error", "tts backend 'missing' is not loaded"), ("b", "spoke", None)]


async def test_speak_falls_back_to_the_next_variant(tmp_path, monkeypatch):
    # The filler goes through speak(): the same variant list as a reply.
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch,
        tts_backends={"a": _FailingTts("a down"), "b": FakeTtsBackend(audio=b"B")},
    )

    await pipeline.speak("привет")

    assert pipeline.play.calls == [(b"B", "audio/mpeg")]


async def test_speak_closes_prefetched_stream_when_play_fails(tmp_path, monkeypatch):
    # The pre-fetched stream goes to play() un-iterated; a play hook that fails before
    # iterating closes it (as Device.play does when the device is offline), and
    # closing it closes the backend's iterator.
    it = _ChunkIter([b"x", b"y"])
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, tts_backend=_IterTts(it))

    async def _offline(mime, audio):
        await audio.aclose()
        raise RuntimeError("cannot play: offline")

    pipeline.play = _offline
    with pytest.raises(RuntimeError, match="offline"):
        await pipeline.speak("привет")
    assert it.closed is True


async def test_run_text_records_tts_attempts_and_time_to_start(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, runs_store=store,
        tts_backends={"a": _FailingTts("a down", delay=0.05), "b": FakeTtsBackend(audio=b"B")},
    )

    result = await pipeline.run_text("включи свет")

    assert result["result"] == "ok"
    assert pipeline.play.calls == [(b"B", "audio/mpeg")]
    rec = store.records[0]
    assert [a["kind"] for a in rec["attempts"]] == ["chain", "tts", "tts"]
    assert _tts_attempts(rec) == [("a", "error", "a down"), ("b", "spoke", None)]
    assert rec["t_tts"] == sum(a["t"] for a in rec["attempts"] if a["kind"] == "tts")
    assert rec["t_tts"] >= 50
