import pytest
from aioesphomeapi import VoiceAssistantEventType as VAET

from src.core_config import AudioConfig, ContextConfig, CoreConfig
from src.pipeline import (
    CaptureBusyError,
    CaptureEmptyError,
    Pipeline,
    contains_stt_hallucination,
)
from src.plugins.llm.base import LlmConfig
from src.runs_store import _LIST_COLS
from src.runtime import Runtime

PUBLIC_BASE_URL = "http://10.0.0.10:8200"


class FakeSvc:
    """Tiny ConfigService stand-in for the Runtime: just exposes a live CoreConfig
    via `.core` and the selected LLM config via `.get("llm")`. The pipeline reads
    both through the Runtime, so mutating `core` here changes them live."""

    def __init__(self, core, llm_cfg):
        self._core = core
        self._llm = llm_cfg

    @property
    def core(self):
        return self._core

    def get(self, _category):
        return self._llm


class FakeSttBackend:
    """STT double: returns scripted text regardless of the PCM passed in."""

    def __init__(self, text="распознанный текст"):
        self.text = text

    async def transcribe(self, pcm):
        return self.text


class FakeTtsBackend:
    def __init__(self, mime="audio/mpeg", audio=b"MP3"):
        self.mime = mime
        self.audio = audio

    async def synthesize(self, text, lang="ru"):
        return (self.mime, self.audio)


class FakeAudioServer:
    def __init__(self):
        self.calls = []  # records (data, content_type) for assertions

    def put(self, data, content_type="audio/mpeg"):
        self.calls.append((data, content_type))
        return "abc123"


class FakeRunsStore:
    """In-memory RunsStore double: records inserted run dicts for assertions.

    Synchronous insert(), so it works as the target of asyncio.to_thread().
    """

    def __init__(self):
        self.records = []

    def insert(self, rec):
        self.records.append(rec)
        return len(self.records)


class FakeRunEvents:
    """Captures broadcast payloads for assertions."""
    def __init__(self):
        self.broadcasts = []
    async def broadcast(self, payload):
        self.broadcasts.append(payload)


def make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="распознанный текст",
                  tts_backend=None, runs_store=None, run_events=None):
    # The data dir is hardcoded in config_store; the pipeline reads it as a module
    # attribute, so isolate per-test context files by monkeypatching DATA_DIR to
    # tmp_path BEFORE the pipeline (and its _context_path) is built.
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    audio_server = FakeAudioServer()
    core = CoreConfig(
        audio=AudioConfig(public_base_url=PUBLIC_BASE_URL),
        context=ContextConfig(),
    )
    rt = Runtime(
        FakeSvc(core, LlmConfig()),
        stt_backend=FakeSttBackend(stt_text),
        llm_backend=object(),
        tts_backend=tts_backend or FakeTtsBackend(),
        hub=object(),
        audio_server=audio_server,
        runs_store=runs_store,
        run_events=run_events,
    )
    pipeline = Pipeline(name, rt)
    events = []
    pipeline.send_event = lambda et, data: events.append((et, data))
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


def set_small_vad_thresholds(pipeline):
    """Shrink VAD thresholds so end-pointing fires after only a few 20 ms frames.

    The pipeline reads these live off rt.core.vad, so we mutate the CoreConfig the
    runtime hands out (CoreConfig is a pydantic model: attribute assignment works)."""
    vad = pipeline.rt.core.vad
    vad.min_speech_ms = 40         # 2 frames of speech to arm
    vad.silence_ms = 100           # 5 frames of trailing silence to end
    vad.max_utterance_ms = 400     # 20 frames hard cap
    vad.no_speech_timeout_ms = 200  # 10 frames with no speech -> finalize


def patch_llm(monkeypatch, reply="ответ"):
    """Stub the whole LLM call. STT is injected as a fake backend, not patched."""

    async def fake(llm_backend, hub, text, **kwargs):
        return reply

    monkeypatch.setattr("src.llm.call_llm_api", fake)


def types_of(events):
    return [et for et, _ in events]


def assert_all_str(events):
    for _, data in events:
        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


async def test_happy_path(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")

    assert await pipeline.on_start("cid", 0, None, None) == 0
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_INTENT_START,
        VAET.VOICE_ASSISTANT_INTENT_END,
        VAET.VOICE_ASSISTANT_TTS_START,
        VAET.VOICE_ASSISTANT_TTS_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]

    data = dict(events)
    assert data[VAET.VOICE_ASSISTANT_STT_END] == {"text": "включи свет"}
    assert data[VAET.VOICE_ASSISTANT_TTS_START] == {"text": "готово"}
    url = data[VAET.VOICE_ASSISTANT_TTS_END]["url"]
    assert url.endswith("/tts/abc123.mp3")
    assert url.startswith(PUBLIC_BASE_URL)
    # MP3 backend -> audio_server stored the audio/mpeg mime.
    assert pipeline.audio_server.calls == [(b"MP3", "audio/mpeg")]
    assert_all_str(events)


async def test_happy_path_wav_extension(tmp_path, monkeypatch):
    # A WAV-producing backend (like Piper) -> url ends .wav and the stored mime is wav.
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch,
        stt_text="включи свет",
        tts_backend=FakeTtsBackend(mime="audio/wav", audio=b"RIFF...."),
    )

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    data = dict(events)
    url = data[VAET.VOICE_ASSISTANT_TTS_END]["url"]
    assert url.endswith("/tts/abc123.wav")
    assert pipeline.audio_server.calls == [(b"RIFF....", "audio/wav")]
    assert_all_str(events)


async def test_empty_audio(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch)

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert_all_str(events)


async def test_empty_stt(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="")

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    data = dict(events)
    assert data[VAET.VOICE_ASSISTANT_STT_END] == {"text": ""}
    assert_all_str(events)


def test_contains_stt_hallucination():
    assert contains_stt_hallucination("Субтитры создавал DimaTorzok")
    assert contains_stt_hallucination("dimatorzok")
    assert not contains_stt_hallucination("включи свет")


async def test_stt_hallucination_discarded(tmp_path, monkeypatch):
    # A Whisper hallucination ("DimaTorzok" subtitle-credit artifact) is blanked,
    # so the run ends like an empty transcription: no INTENT/TTS, STT_END empty.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, stt_text="Субтитры создавал DimaTorzok"
    )

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    data = dict(events)
    assert data[VAET.VOICE_ASSISTANT_STT_END] == {"text": ""}
    assert_all_str(events)


async def test_run_recorded_on_stt_hallucination(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch,
        stt_text="Субтитры создавал DimaTorzok", runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    # The hallucination is dropped: recorded like an empty transcription.
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "empty"
    assert rec["stt_text"] == ""


async def test_pipelines_are_independent(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    a, _ = make_pipeline(tmp_path, monkeypatch, name="a")
    b, _ = make_pipeline(tmp_path, monkeypatch, name="b")

    await a.on_start("cid", 0, None, None)
    await a.on_audio(b"\x01\x02" * 100)

    # Pushing audio only into "a" must not touch "b".
    assert len(b._buffer) == 0
    assert a._context_path != b._context_path
    assert a._context_path.endswith("context_a.txt")
    assert b._context_path.endswith("context_b.txt")


# A 640-byte (one 20 ms frame) chunk of silence-shaped PCM. The injected FakeVad
# decides speech/silence regardless of the bytes; one chunk == one VAD frame.
FRAME = b"\x00" * 640

FULL_SEQUENCE = [
    VAET.VOICE_ASSISTANT_RUN_START,
    VAET.VOICE_ASSISTANT_STT_START,
    VAET.VOICE_ASSISTANT_STT_END,
    VAET.VOICE_ASSISTANT_INTENT_START,
    VAET.VOICE_ASSISTANT_INTENT_END,
    VAET.VOICE_ASSISTANT_TTS_START,
    VAET.VOICE_ASSISTANT_TTS_END,
    VAET.VOICE_ASSISTANT_RUN_END,
]


async def test_vad_endpoint_finalize(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    set_small_vad_thresholds(pipeline)
    # 3 speech frames (>= 40 ms min) then 6 silence frames (>= 100 ms) -> endpoint.
    pipeline._vad = FakeVad([True, True, True, False, False, False, False, False, False])

    await pipeline.on_start("cid", 0, None, None)
    # 9 frames in one chunk so all are processed in a single on_audio call.
    await pipeline.on_audio(FRAME * 9)

    assert types_of(events) == FULL_SEQUENCE
    assert pipeline._finalized is True

    # A later device stop must NOT emit anything again (finalize-once).
    before = len(events)
    await pipeline.on_stop(False)
    assert len(events) == before
    assert_all_str(events)


async def test_vad_maxlen_finalize(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="ок")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="команда")
    set_small_vad_thresholds(pipeline)
    # Always speech: no trailing silence, so only the max-length cap can finalize.
    pipeline._vad = FakeVad([True])

    await pipeline.on_start("cid", 0, None, None)
    # max_utterance_ms=400 -> 20 frames; feed 21 to cross the cap.
    await pipeline.on_audio(FRAME * 21)

    assert pipeline._finalized is True
    assert VAET.VOICE_ASSISTANT_RUN_END in types_of(events)
    # Full happy path because STT returned text.
    assert types_of(events) == FULL_SEQUENCE

    before = len(events)
    await pipeline.on_stop(False)
    assert len(events) == before


async def test_vad_no_speech_finalize(tmp_path, monkeypatch):
    # No speech detected by VAD -> STT is SKIPPED entirely (so Whisper can't
    # hallucinate on non-speech audio); STT_END is emitted empty and the run ends
    # without intent/TTS.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="   ")
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    pipeline._vad = FakeVad([False])

    # Spy on transcribe to assert it is never called on a no-speech run.
    stt_calls = []
    orig_transcribe = pipeline.stt_backend.transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.stt_backend.transcribe = spy_transcribe

    await pipeline.on_start("cid", 0, None, None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)

    assert pipeline._finalized is True
    # STT must never run on a no-speech utterance.
    assert stt_calls == []
    # STT_START (from on_start) is balanced by an empty STT_END; no intent/TTS.
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    data = dict(events)
    assert data[VAET.VOICE_ASSISTANT_STT_END] == {"text": ""}
    assert_all_str(events)


async def test_no_speech_run_is_recorded_as_empty(tmp_path, monkeypatch):
    # A no-speech finalization skips STT but still records the run as empty
    # (result="empty", stt_text="", t_stt=0, reason="no_speech").
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="   ", runs_store=store)
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    pipeline._vad = FakeVad([False])

    stt_calls = []
    orig_transcribe = pipeline.stt_backend.transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.stt_backend.transcribe = spy_transcribe

    await pipeline.on_start("cid", 0, None, None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)

    # STT never ran, but the run is recorded once as an empty run.
    assert stt_calls == []
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "empty"
    assert rec["stt_text"] == ""
    assert rec["t_stt"] == 0
    assert rec["reason"] == "no_speech"


async def test_on_start_rebuilds_vad_when_aggressiveness_changed(tmp_path, monkeypatch):
    # webrtcvad.Vad bakes the aggressiveness in at construction, so on_start must rebuild
    # the real Vad object when rt.core.vad.aggressiveness changed since it was last built,
    # and leave it untouched (same object) when the value is unchanged.
    pipeline, _ = make_pipeline(tmp_path, monkeypatch)
    initial_vad = pipeline._vad
    initial_aggr = pipeline._vad_aggressiveness

    # Change to a different valid value (0..3) and start: the Vad object is rebuilt.
    new_aggr = (initial_aggr + 1) % 4
    pipeline.rt.core.vad.aggressiveness = new_aggr
    await pipeline.on_start("cid", 0, None, None)
    assert pipeline._vad is not initial_vad            # rebuilt (new object)
    assert pipeline._vad_aggressiveness == new_aggr     # tracked value updated

    # A second start with the SAME aggressiveness is a no-op: the object is not replaced.
    same_vad = pipeline._vad
    await pipeline.on_start("cid", 0, None, None)
    assert pipeline._vad is same_vad                    # unchanged -> not rebuilt
    assert pipeline._vad_aggressiveness == new_aggr


async def test_finalize_once_race(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="здравствуйте")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="привет")
    # Give it some audio so the first claim/run runs the full path. We claim+run
    # directly (no on_start), so it emits only the post-start subset.
    pipeline._buffer.extend(b"\x00" * 640)

    pcm = pipeline._claim()
    assert pcm is not None
    await pipeline._run("a", pcm, pipeline._conversation_id)
    after_first = list(types_of(events))
    assert after_first == [
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_INTENT_START,
        VAET.VOICE_ASSISTANT_INTENT_END,
        VAET.VOICE_ASSISTANT_TTS_START,
        VAET.VOICE_ASSISTANT_TTS_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]

    # A second claim is a no-op: already finalized -> returns None, so no second
    # run happens and no extra events are emitted.
    second = pipeline._claim()
    assert second is None
    assert types_of(events) == after_first


async def test_history_flows_across_runs(tmp_path, monkeypatch):
    import json

    from src import context

    # Capturing fake: records (text, history) per call and returns a scripted reply.
    seen = []  # list of (text, history)
    replies = {"первый вопрос": "первый ответ", "второй вопрос": "второй ответ"}

    async def fake(llm_backend, hub, text, **kwargs):
        seen.append((text, kwargs.get("history")))
        return replies[text]

    monkeypatch.setattr("src.llm.call_llm_api", fake)

    pipeline, _ = make_pipeline(tmp_path, monkeypatch, name="hist", stt_text="первый вопрос")

    # Run 1.
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

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

    # Run 2 on the same pipeline instance with a different utterance.
    # on_start resets all per-run state (including _finalized).
    pipeline.stt_backend.text = "второй вопрос"
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

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

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    # Exactly one run recorded (no double-insert across on_audio/on_stop).
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
    assert rec["audio_fmt"] == "mp3"
    assert rec["error_stage"] is None


async def test_run_broadcast_on_happy_path(tmp_path, monkeypatch):
    # With both a store and a run-events hub, a finalized run is broadcast once,
    # carrying the same summary shape /api/runs returns.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    hub = FakeRunEvents()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch, stt_text="включи свет", runs_store=store, run_events=hub,
    )

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    # Exactly one broadcast for the single recorded run.
    assert len(hub.broadcasts) == 1
    payload = hub.broadcasts[0]
    assert payload["type"] == "run"
    # The store returns id 1 for the first insert; the summary echoes it.
    assert payload["run"]["id"] == 1
    assert payload["run"]["result"] in ("ok", "tool")
    # The broadcast run dict is exactly the summary shape (_LIST_COLS).
    assert set(payload["run"].keys()) == set(_LIST_COLS)


async def test_no_broadcast_without_hub(tmp_path, monkeypatch):
    # run_events=None must not error and must record the run normally.
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет", runs_store=store)

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert len(store.records) == 1


async def test_run_recorded_on_empty_stt(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="", runs_store=store)

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    # Empty STT short-circuits but still records a run with result "empty".
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "empty"
    assert rec["stt_text"] == ""
    assert rec["llm_text"] == ""
    assert rec["t_total"] >= 0


async def test_truly_empty_audio_not_recorded(tmp_path, monkeypatch):
    # No PCM buffered at all -> early return before building a record; nothing logged.
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, runs_store=store)

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_stop(False)

    assert store.records == []


class RaisingTtsBackend:
    """TTS double that always fails, to exercise the TTS except branch."""

    async def synthesize(self, text, lang="ru"):
        raise RuntimeError("tts boom")


async def test_llm_error_then_tts_fail_keeps_error_stage_llm(tmp_path, monkeypatch):
    # When the LLM reply is an "Ошибка:" string the run is classified as an
    # LLM error but still continues into TTS. If TTS then fails, the TTS except
    # must NOT clobber the already-set LLM root cause.
    patch_llm(monkeypatch, reply="Ошибка: модель недоступна")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path, monkeypatch,
        stt_text="включи свет",
        tts_backend=RaisingTtsBackend(),
        runs_store=store,
    )

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    # LLM stage/text preserved despite the later TTS failure.
    assert rec["error_stage"] == "LLM"
    assert rec["error_text"] == "Ошибка: модель недоступна"


async def test_raw_capture_writes_wav_when_enabled(tmp_path, monkeypatch):
    # With capture enabled, the finalized utterance PCM is saved as a single
    # 16 kHz / mono / 16-bit WAV under capture.dir, matching the buffered bytes.
    import wave

    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.enabled = True
    pipeline.rt.core.capture.dir = str(cap_dir)

    pcm = b"\x01\x02" * 100  # 400 bytes -> 200 16-bit frames
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(pcm)
    await pipeline.on_stop(False)

    wavs = list(cap_dir.glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        # 2-byte samples: frame count == len(pcm) / 2.
        assert w.getnframes() == len(pcm) // 2
        assert w.readframes(w.getnframes()) == pcm


async def test_raw_capture_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    # Default capture.enabled is False, so no WAV is written.
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="включи свет")
    # Point dir at our path but leave enabled at its default (False).
    pipeline.rt.core.capture.dir = str(cap_dir)
    assert pipeline.rt.core.capture.enabled is False

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


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


async def test_capture_run_returns_wav_bytes_and_skips_pipeline(tmp_path, monkeypatch):
    # An armed capture run resolves its Future with the streamed PCM as in-memory
    # WAV bytes and runs NO STT/LLM/TTS: only RUN_START and RUN_END are emitted,
    # neither the STT backend nor the LLM is ever called, and NOTHING is written to
    # the capture dir (the manual capture is ephemeral).
    cap_dir = tmp_path / "captures"
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, monkeypatch, name="dev", stt_text="должно быть проигнорировано", runs_store=store,
    )
    pipeline.rt.core.capture.dir = str(cap_dir)

    # Track STT/LLM invocation: in capture mode neither must run.
    stt_calls = []
    orig_transcribe = pipeline.stt_backend.transcribe

    async def spy_transcribe(pcm):
        stt_calls.append(pcm)
        return await orig_transcribe(pcm)

    pipeline.stt_backend.transcribe = spy_transcribe
    llm_calls = []

    async def fake_llm(*a, **k):
        llm_calls.append(a)
        return "nope"

    monkeypatch.setattr("src.llm.call_llm_api", fake_llm)

    pcm = b"\x01\x02" * 100  # 400 bytes -> 200 16-bit frames
    future = pipeline.arm_capture(5)
    assert pipeline._capture_armed is True
    await pipeline.on_start("cid", 0, None, None)
    # arming is consumed into the per-run flag.
    assert pipeline._capture_armed is False
    assert pipeline._capture_run is True
    await pipeline.on_audio(pcm)
    await pipeline.on_stop(False)

    # Only the capture-only event pair, NO STT/INTENT/TTS events.
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert stt_calls == [] and llm_calls == []
    # No run recorded: capture-only never touches the runs store.
    assert store.records == []
    # The capture flag is cleared so the next start is a normal run.
    assert pipeline._capture_run is False

    # The Future resolved with a valid 16k/mono/16-bit WAV matching the PCM.
    assert future.done()
    nch, sw, fr, nframes, frames = _read_wav_bytes(future.result())
    assert (nch, sw, fr) == (1, 2, 16000)
    assert nframes == len(pcm) // 2  # 2-byte samples
    assert frames == pcm
    # Ephemeral: nothing written to the capture dir for the manual path.
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


async def test_capture_run_ends_on_deadline(tmp_path, monkeypatch):
    # When the device never signals stop, the server-side deadline ends the capture
    # on the next audio chunk and still resolves the Future with WAV bytes + emits
    # RUN_END — and writes nothing to disk.
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev")
    pipeline.rt.core.capture.dir = str(cap_dir)

    future = pipeline.arm_capture(5)
    await pipeline.on_start("cid", 0, None, None)
    # Force the deadline into the past so the next chunk ends the capture.
    pipeline._capture_deadline = 0.0
    await pipeline.on_audio(b"\x03\x04" * 50)  # 100 bytes

    assert pipeline._capture_run is False
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert future.done()
    _nch, _sw, _fr, _n, frames = _read_wav_bytes(future.result())
    assert frames == b"\x03\x04" * 50
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []

    # A later device stop must NOT emit anything again (finalize-once).
    before = len(events)
    await pipeline.on_stop(False)
    assert len(events) == before


async def test_capture_longer_than_60s_is_not_truncated_at_hard_cap(tmp_path, monkeypatch):
    # A capture for > 60 s must NOT be cut at the 60 s normal-run HARD_CAP_BYTES:
    # the capture branch sizes its own cap to (_capture_seconds + 2) s. Drive the
    # branch with > 60 s of PCM and _capture_seconds = 120, then stop the device and
    # assert the returned WAV contains ALL streamed bytes (well past HARD_CAP_BYTES).
    from src.pipeline import HARD_CAP_BYTES, SAMPLE_RATE

    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev")
    pipeline.rt.core.capture.dir = str(tmp_path / "captures")

    future = pipeline.arm_capture(120)  # requested 120 s capture
    await pipeline.on_start("cid", 0, None, None)
    assert pipeline._capture_run is True
    # The capture-specific cap is duration-based, not the 60 s HARD_CAP.
    assert pipeline._capture_cap_bytes() == (120 + 2) * SAMPLE_RATE * 2
    assert pipeline._capture_cap_bytes() > HARD_CAP_BYTES

    # Stream ~65 s of PCM in 1 s chunks: more than HARD_CAP_BYTES (60 s) but under the
    # 122 s capture cap, so none of it may be dropped.
    one_second = b"\x05\x06" * SAMPLE_RATE  # SAMPLE_RATE * 2 bytes = 1 s of 16-bit PCM
    total = bytearray()
    for _ in range(65):
        await pipeline.on_audio(bytes(one_second))
        total.extend(one_second)
    # Still capturing: neither the byte cap nor the deadline has fired.
    assert pipeline._capture_run is True
    assert len(total) > HARD_CAP_BYTES  # we really did exceed the 60 s normal cap

    await pipeline.on_stop(False)
    assert pipeline._capture_run is False
    assert future.done()
    _nch, _sw, _fr, _n, frames = _read_wav_bytes(future.result())
    # ALL streamed bytes survive — not truncated at 60 s HARD_CAP_BYTES.
    assert frames == bytes(total)
    assert len(frames) > HARD_CAP_BYTES


async def test_normal_run_hard_cap_truncates_at_60s(tmp_path, monkeypatch):
    # Regression guard: a NORMAL (non-capture) run is still truncated at the 60 s
    # HARD_CAP_BYTES — the duration-based capture cap must NOT leak into normal runs.
    # Feed > 60 s of PCM in one chunk; the run finalizes ("maxlen") with the buffer
    # capped at exactly HARD_CAP_BYTES.
    from src.pipeline import HARD_CAP_BYTES, SAMPLE_RATE

    captured = {}
    patch_llm(monkeypatch, reply="ок")
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="команда")
    pipeline._vad = FakeVad([True])  # always speech: only the cap can finalize

    orig_transcribe = pipeline.stt_backend.transcribe

    async def spy(pcm):
        captured["len"] = len(pcm)
        return await orig_transcribe(pcm)

    pipeline.stt_backend.transcribe = spy

    await pipeline.on_start("cid", 0, None, None)
    # 65 s worth of PCM in a single chunk -> HARD_CAP fires, buffer trimmed to 60 s.
    await pipeline.on_audio(b"\x07\x08" * (SAMPLE_RATE * 65))

    assert pipeline._finalized is True
    assert VAET.VOICE_ASSISTANT_RUN_END in types_of(events)
    # The transcribed PCM was capped at exactly the 60 s HARD_CAP_BYTES.
    assert captured["len"] == HARD_CAP_BYTES


async def test_capture_arming_does_not_affect_normal_run(tmp_path, monkeypatch):
    # A normal wake-word run on a pipeline that was NEVER armed runs the full
    # STT->LLM->TTS path unchanged (regression guard for the capture branch).
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    assert pipeline._capture_run is False
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == FULL_SEQUENCE
    # No manual-capture WAV written for a normal run.
    assert not cap_dir.exists() or list(cap_dir.glob("*_manual_*.wav")) == []


async def test_capture_run_after_normal_run_is_isolated(tmp_path, monkeypatch):
    # Arm only the SECOND run on a reused pipeline: run 1 is a normal full pipeline,
    # run 2 is capture-only. Confirms arming is per-run and does not leak backwards.
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    # Run 1: normal.
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)
    assert types_of(events) == FULL_SEQUENCE
    events.clear()

    # Run 2: capture-only.
    pcm = b"\x05\x06" * 80
    future = pipeline.arm_capture(3)
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(pcm)
    await pipeline.on_stop(False)
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    # Capture returns WAV bytes via the Future; nothing is written to disk.
    assert future.done()
    _nch, _sw, _fr, _n, frames = _read_wav_bytes(future.result())
    assert frames == pcm
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


async def test_expired_arm_does_not_capture_later_run(tmp_path, monkeypatch):
    # FIX 2: if the button press is lost / the device never starts, the armed flag
    # must expire instead of silently turning a later real wake-word run into a
    # capture-only run. Arm, blow past the arm-arrival deadline, then start a normal
    # run: it must run the full STT->LLM->TTS path, not capture.
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    future = pipeline.arm_capture(5)
    assert pipeline._capture_armed is True
    # Force the arm-arrival deadline into the past: the press effectively never landed.
    pipeline._capture_arm_deadline = 0.0

    # A real wake-word run arrives later (with a phrase). It must NOT be captured.
    await pipeline.on_start("cid", 0, None, "захар")
    assert pipeline._capture_run is False
    assert pipeline._capture_armed is False  # stale flag cleared, not consumed
    # The pending capture Future is failed so a waiting caller doesn't hang.
    assert future.done()
    with pytest.raises(RuntimeError):
        future.result()
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == FULL_SEQUENCE
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


async def test_wake_word_run_while_armed_keeps_flag_for_manual_start(tmp_path, monkeypatch):
    # FIX 3: between arm_capture() and the button-initiated start, a real wake word
    # could fire. Its on_start carries a wake_word_phrase, so it must NOT consume the
    # armed flag: it runs as a normal assistant run, and the flag survives so the
    # later phraseless manual start still gets captured.
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    future = pipeline.arm_capture(5)
    assert pipeline._capture_armed is True

    # A genuine wake-word run sneaks in WITH a phrase while armed.
    await pipeline.on_start("cid", 0, None, "захар")
    assert pipeline._capture_run is False           # not captured
    assert pipeline._capture_armed is True          # flag preserved for the manual start
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)
    assert types_of(events) == FULL_SEQUENCE        # ran the full assistant pipeline
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []
    events.clear()

    # Now the genuine manual start arrives (no phrase) and consumes the flag.
    await pipeline.on_start("cid", 0, None, None)
    assert pipeline._capture_run is True
    assert pipeline._capture_armed is False
    pcm = b"\x07\x08" * 60
    await pipeline.on_audio(pcm)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    # The capture resolved its Future with WAV bytes; nothing on disk.
    assert future.done()
    _nch, _sw, _fr, _n, frames = _read_wav_bytes(future.result())
    assert frames == pcm
    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


async def test_second_concurrent_capture_is_rejected_first_still_completes(
    tmp_path, monkeypatch
):
    # FIX A: while one capture is armed/in-flight, a second arm_capture on the SAME
    # pipeline must be refused with CaptureBusyError (no future overwrite, no cross-
    # request cancellation) — and the FIRST capture must still complete normally.
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev")
    pipeline.rt.core.capture.dir = str(cap_dir)

    # First capture armed and still pending (no start/audio yet).
    first = pipeline.arm_capture(5)
    assert pipeline._capture_armed is True
    assert not first.done()

    # A second concurrent capture is rejected; the first future is untouched.
    with pytest.raises(CaptureBusyError):
        pipeline.arm_capture(5)
    assert pipeline._capture_future is first  # not overwritten by the rejected arm
    assert not first.done()

    # The first capture now runs to completion and resolves its own future.
    pcm = b"\x01\x02" * 100
    await pipeline.on_start("cid", 0, None, None)
    assert pipeline._capture_run is True
    await pipeline.on_audio(pcm)
    await pipeline.on_stop(False)

    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert first.done()
    _nch, _sw, _fr, _n, frames = _read_wav_bytes(first.result())
    assert frames == pcm

    # With the first capture finished (future done), a fresh capture is allowed again.
    second = pipeline.arm_capture(3)
    assert second is not first


async def test_capture_empty_audio_fails_future_with_capture_empty_error(
    tmp_path, monkeypatch
):
    # FIX B: a capture run that produces NO audio fails the future with the distinct
    # CaptureEmptyError (mapped to HTTP 500 server-side), not a generic RuntimeError.
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, monkeypatch, name="dev")
    pipeline.rt.core.capture.dir = str(cap_dir)

    future = pipeline.arm_capture(5)
    await pipeline.on_start("cid", 0, None, None)
    # Device stops without ever streaming any audio -> empty PCM buffer.
    await pipeline.on_stop(False)

    assert future.done()
    with pytest.raises(CaptureEmptyError):
        future.result()
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert pipeline._capture_run is False


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

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert len(store.records) == 1
    rec = store.records[0]
    assert rec["result"] == "error"
    assert rec["error_stage"] == "TTS"
    assert rec["error_text"] == "tts boom"
