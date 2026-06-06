from aioesphomeapi import VoiceAssistantEventType as VAET

from src.pipeline import Pipeline

PUBLIC_BASE_URL = "http://10.0.0.10:8200"


class FakeTtsBackend:
    async def synthesize(self, text, lang="ru"):
        return ("audio/mpeg", b"MP3")


class FakeAudioServer:
    def put(self, data):
        return "abc123"


def make_pipeline(tmp_path, name="dev"):
    pipeline = Pipeline(
        name,
        client_ext=object(),
        client_local=object(),
        tts_backend=FakeTtsBackend(),
        audio_server=FakeAudioServer(),
        public_base_url=PUBLIC_BASE_URL,
        context_dir=str(tmp_path),
    )
    events = []
    pipeline.send_event = lambda et, data: events.append((et, data))
    return pipeline, events


def patch_brain(monkeypatch, stt_text="распознанный текст", reply="ответ"):
    async def fake_transcribe(client_ext, pcm):
        return stt_text

    async def fake_call_groq_api(client_ext, client_local, text):
        return reply

    monkeypatch.setattr("src.stt.transcribe", fake_transcribe)
    monkeypatch.setattr("src.llm.call_groq_api", fake_call_groq_api)


def types_of(events):
    return [et for et, _ in events]


def assert_all_str(events):
    for _, data in events:
        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, str)


async def test_happy_path(tmp_path, monkeypatch):
    patch_brain(monkeypatch, stt_text="включи свет", reply="готово")
    pipeline, events = make_pipeline(tmp_path)

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
    assert_all_str(events)


async def test_empty_audio(tmp_path, monkeypatch):
    patch_brain(monkeypatch)
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
    patch_brain(monkeypatch, stt_text="")
    pipeline, events = make_pipeline(tmp_path)

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
    patch_brain(monkeypatch)
    a, _ = make_pipeline(tmp_path, name="a")
    b, _ = make_pipeline(tmp_path, name="b")

    await a.on_start("cid", 0, None, None)
    await a.on_audio(b"\x01\x02" * 100)

    # Pushing audio only into "a" must not touch "b".
    assert len(b._buffer) == 0
    assert a._context_path != b._context_path
    assert a._context_path.endswith("context_a.txt")
    assert b._context_path.endswith("context_b.txt")
