from aioesphomeapi import VoiceAssistantEventType as VAET

from src.core_config import AudioConfig, ContextConfig, CoreConfig
from src.pipeline import Pipeline
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


def make_pipeline(tmp_path, name="dev", stt_text="распознанный текст",
                  tts_backend=None, runs_store=None, run_events=None):
    audio_server = FakeAudioServer()
    core = CoreConfig(
        audio=AudioConfig(public_base_url=PUBLIC_BASE_URL),
        context=ContextConfig(dir=str(tmp_path)),
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
    pipeline, events = make_pipeline(tmp_path, stt_text="включи свет")

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
        tmp_path,
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
    pipeline, events = make_pipeline(tmp_path)

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
    pipeline, events = make_pipeline(tmp_path, stt_text="")

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


async def test_pipelines_are_independent(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    a, _ = make_pipeline(tmp_path, name="a")
    b, _ = make_pipeline(tmp_path, name="b")

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
    pipeline, events = make_pipeline(tmp_path, stt_text="включи свет")
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
    pipeline, events = make_pipeline(tmp_path, stt_text="команда")
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
    # STT returns whitespace -> after STT_END the run ends without intent/TTS.
    patch_llm(monkeypatch)
    pipeline, events = make_pipeline(tmp_path, stt_text="   ")
    set_small_vad_thresholds(pipeline)
    # Never any speech: only the no-speech timeout can finalize.
    pipeline._vad = FakeVad([False])

    await pipeline.on_start("cid", 0, None, None)
    # no_speech_timeout_ms=200 -> 10 frames; feed 11 to cross it.
    await pipeline.on_audio(FRAME * 11)

    assert pipeline._finalized is True
    # pcm is non-empty (we buffered it), STT runs and returns whitespace.
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_STT_START,
        VAET.VOICE_ASSISTANT_STT_END,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert_all_str(events)


async def test_finalize_once_race(tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="здравствуйте")
    pipeline, events = make_pipeline(tmp_path, stt_text="привет")
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

    pipeline, _ = make_pipeline(tmp_path, name="hist", stt_text="первый вопрос")

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
    pipeline, _ = make_pipeline(tmp_path, stt_text="включи свет", runs_store=store)

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
        tmp_path, stt_text="включи свет", runs_store=store, run_events=hub,
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
    pipeline, _ = make_pipeline(tmp_path, stt_text="включи свет", runs_store=store)

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert len(store.records) == 1


async def test_run_recorded_on_empty_stt(tmp_path, monkeypatch):
    patch_llm(monkeypatch)
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(tmp_path, stt_text="", runs_store=store)

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
    pipeline, _ = make_pipeline(tmp_path, runs_store=store)

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
        tmp_path,
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
    pipeline, _ = make_pipeline(tmp_path, name="dev", stt_text="включи свет")
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
    pipeline, _ = make_pipeline(tmp_path, name="dev", stt_text="включи свет")
    # Point dir at our path but leave enabled at its default (False).
    pipeline.rt.core.capture.dir = str(cap_dir)
    assert pipeline.rt.core.capture.enabled is False

    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert not cap_dir.exists() or list(cap_dir.glob("*.wav")) == []


# --- manual "record X seconds" capture-only mode -----------------------------

async def test_capture_run_writes_wav_and_skips_pipeline(tmp_path, monkeypatch):
    # An armed capture run records the streamed PCM to a WAV and runs NO STT/LLM/TTS:
    # only RUN_START and RUN_END are emitted, the WAV matches the buffered bytes, and
    # neither the STT backend nor the LLM is ever called.
    import wave

    cap_dir = tmp_path / "captures"
    store = FakeRunsStore()
    pipeline, events = make_pipeline(
        tmp_path, name="dev", stt_text="должно быть проигнорировано", runs_store=store,
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

    pcm = b"\x01\x02" * 100  # 400 bytes
    pipeline.arm_capture(5)
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

    wavs = list(cap_dir.glob("*_manual_*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.readframes(w.getnframes()) == pcm


async def test_capture_run_ends_on_deadline(tmp_path, monkeypatch):
    # When the device never signals stop, the server-side deadline ends the capture
    # on the next audio chunk and still writes the WAV + emits RUN_END.
    import wave

    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, name="dev")
    pipeline.rt.core.capture.dir = str(cap_dir)

    pipeline.arm_capture(5)
    await pipeline.on_start("cid", 0, None, None)
    # Force the deadline into the past so the next chunk ends the capture.
    pipeline._capture_deadline = 0.0
    await pipeline.on_audio(b"\x03\x04" * 50)  # 100 bytes

    assert pipeline._capture_run is False
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    wavs = list(cap_dir.glob("*_manual_*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.readframes(w.getnframes()) == b"\x03\x04" * 50

    # A later device stop must NOT emit anything again (finalize-once).
    before = len(events)
    await pipeline.on_stop(False)
    assert len(events) == before


async def test_capture_arming_does_not_affect_normal_run(tmp_path, monkeypatch):
    # A normal wake-word run on a pipeline that was NEVER armed runs the full
    # STT->LLM->TTS path unchanged (regression guard for the capture branch).
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, stt_text="включи свет")
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
    pipeline, events = make_pipeline(tmp_path, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    # Run 1: normal.
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)
    assert types_of(events) == FULL_SEQUENCE
    events.clear()

    # Run 2: capture-only.
    pipeline.arm_capture(3)
    await pipeline.on_start("cid", 0, None, None)
    await pipeline.on_audio(b"\x05\x06" * 80)
    await pipeline.on_stop(False)
    assert types_of(events) == [
        VAET.VOICE_ASSISTANT_RUN_START,
        VAET.VOICE_ASSISTANT_RUN_END,
    ]
    assert len(list(cap_dir.glob("*_manual_*.wav"))) == 1


async def test_expired_arm_does_not_capture_later_run(tmp_path, monkeypatch):
    # FIX 2: if the button press is lost / the device never starts, the armed flag
    # must expire instead of silently turning a later real wake-word run into a
    # capture-only run. Arm, blow past the arm-arrival deadline, then start a normal
    # run: it must run the full STT->LLM->TTS path, not capture.
    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    pipeline.arm_capture(5)
    assert pipeline._capture_armed is True
    # Force the arm-arrival deadline into the past: the press effectively never landed.
    pipeline._capture_arm_deadline = 0.0

    # A real wake-word run arrives later (with a phrase). It must NOT be captured.
    await pipeline.on_start("cid", 0, None, "захар")
    assert pipeline._capture_run is False
    assert pipeline._capture_armed is False  # stale flag cleared, not consumed
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)

    assert types_of(events) == FULL_SEQUENCE
    assert not cap_dir.exists() or list(cap_dir.glob("*_manual_*.wav")) == []


async def test_wake_word_run_while_armed_keeps_flag_for_manual_start(tmp_path, monkeypatch):
    # FIX 3: between arm_capture() and the button-initiated start, a real wake word
    # could fire. Its on_start carries a wake_word_phrase, so it must NOT consume the
    # armed flag: it runs as a normal assistant run, and the flag survives so the
    # later phraseless manual start still gets captured.
    import wave

    patch_llm(monkeypatch, reply="готово")
    cap_dir = tmp_path / "captures"
    pipeline, events = make_pipeline(tmp_path, name="dev", stt_text="включи свет")
    pipeline.rt.core.capture.dir = str(cap_dir)

    pipeline.arm_capture(5)
    assert pipeline._capture_armed is True

    # A genuine wake-word run sneaks in WITH a phrase while armed.
    await pipeline.on_start("cid", 0, None, "захар")
    assert pipeline._capture_run is False           # not captured
    assert pipeline._capture_armed is True          # flag preserved for the manual start
    await pipeline.on_audio(b"\x01\x02" * 100)
    await pipeline.on_stop(False)
    assert types_of(events) == FULL_SEQUENCE        # ran the full assistant pipeline
    assert not cap_dir.exists() or list(cap_dir.glob("*_manual_*.wav")) == []
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
    wavs = list(cap_dir.glob("*_manual_*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as w:
        assert w.readframes(w.getnframes()) == pcm


async def test_tts_fail_without_prior_error_sets_tts_stage(tmp_path, monkeypatch):
    # Sanity check the guard's other branch: a TTS failure with no earlier error
    # still records error_stage="TTS".
    patch_llm(monkeypatch, reply="готово")
    store = FakeRunsStore()
    pipeline, _ = make_pipeline(
        tmp_path,
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
