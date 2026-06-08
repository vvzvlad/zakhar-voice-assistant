from aioesphomeapi import VoiceAssistantEventType as VAET

from src.core_config import AudioConfig, ContextConfig, CoreConfig
from src.pipeline import Pipeline

PUBLIC_BASE_URL = "http://10.0.0.10:8200"


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


def make_pipeline(tmp_path, name="dev", stt_text="распознанный текст", tts_backend=None):
    audio_server = FakeAudioServer()
    core = CoreConfig(
        audio=AudioConfig(public_base_url=PUBLIC_BASE_URL),
        context=ContextConfig(dir=str(tmp_path)),
    )
    pipeline = Pipeline(
        name,
        hub=object(),
        stt_backend=FakeSttBackend(stt_text),
        llm_backend=object(),
        tts_backend=tts_backend or FakeTtsBackend(),
        audio_server=audio_server,
        weather_client=object(),
        core=core,
        max_tool_rounds=5,
    )
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
    """Shrink VAD thresholds so end-pointing fires after only a few 20 ms frames."""
    pipeline.vad_min_speech_ms = 40       # 2 frames of speech to arm
    pipeline.vad_silence_ms = 100         # 5 frames of trailing silence to end
    pipeline.vad_max_utterance_ms = 400   # 20 frames hard cap
    pipeline.vad_no_speech_timeout_ms = 200  # 10 frames with no speech -> finalize


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
    await pipeline._run("a", pcm)
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
