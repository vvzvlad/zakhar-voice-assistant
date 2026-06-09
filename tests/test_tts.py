import io
import wave

import httpx
import pytest
import respx

from src.tts import (
    PiperTtsBackend,
    TeraTtsHttpBackend,
    YandexTtsBackend,
    _decode_v3_audio,
    split_sentences,
    wav_to_mp3,
    yandex_stress_markup,
)

YANDEX_URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"


def _make_wav(sample_rate: int = 16000, channels: int = 1, seconds: float = 0.1) -> bytes:
    """Build a tiny in-memory 16-bit PCM WAV of silence for transcoding tests."""
    frames = int(sample_rate * seconds)
    pcm = b"\x00\x00" * frames * channels  # 16-bit zeros (silence)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def test_wav_to_mp3_produces_mp3_frame():
    wav = _make_wav()
    out = wav_to_mp3(wav)
    assert out  # non-empty
    assert out[0] == 0xFF  # MP3 frame sync byte


def test_split_sentences_basic():
    assert split_sentences("Привет. Как дела?") == ["Привет.", "Как дела?"]


def test_split_sentences_ellipsis_normalized():
    assert split_sentences("Эхе-хе… ладно, барин.") == ["Эхе-хе.", "ладно, барин."]


def test_split_sentences_runs_of_dots_collapsed():
    assert split_sentences("Ну...  что ж.") == ["Ну.", "что ж."]


def test_split_sentences_no_terminal_punctuation():
    assert split_sentences("просто текст") == ["просто текст"]


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_split_sentences_drops_punctuation_only_fragment():
    assert split_sentences("Что? . Ну ладно.") == ["Что?", "Ну ладно."]


def test_split_sentences_runs_of_dots_only():
    assert split_sentences("...") == []


def test_split_sentences_ellipsis_only():
    assert split_sentences("…") == []


def test_split_sentences_drops_bang_only_fragment():
    assert split_sentences("Раз. ! Два.") == ["Раз.", "Два."]


def test_yandex_stress_markup_single_word():
    assert yandex_stress_markup("приве́т") == "прив+ет"


def test_yandex_stress_markup_two_words():
    assert yandex_stress_markup("больша́я ко́мната") == "больш+ая к+омната"


def test_yandex_stress_markup_passthrough():
    assert yandex_stress_markup("просто текст") == "просто текст"


def test_yandex_stress_markup_orphan_acute_dropped():
    # consonant + combining acute (U+0301) -> accent removed, no "+".
    # Built explicitly so the input is "к" + U+0301, not the precomposed U+045C.
    assert yandex_stress_markup("ќ") == "к"


def test_yandex_backend_requires_api_key():
    with pytest.raises(ValueError):
        YandexTtsBackend(None, api_key="", voice="zahar", role="neutral",
                         speed=1.0, folder_id="", url="http://x", timeout=5)


@respx.mock
async def test_yandex_synthesize_posts_mp3_and_returns_audio():
    import base64
    import json

    audio_bytes = b"\xff\xf3audio"
    chunk = json.dumps({"result": {"audioChunk": {"data": base64.b64encode(audio_bytes).decode()}}})
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text=chunk,
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, folder_id="",
                                   url=YANDEX_URL, timeout=10)
        mime, audio = await backend.synthesize("приве́т", "ru")
    assert mime == "audio/mpeg"
    assert audio == audio_bytes
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Api-Key k"
    sent = json.loads(route.calls.last.request.content)
    assert {"voice": "zahar"} in sent["hints"]
    assert {"role": "neutral"} in sent["hints"]
    assert sent["outputAudioSpec"]["containerAudio"]["containerAudioType"] == "MP3"
    assert "прив+ет" in sent["text"]   # stress markup from "приве́т"
    assert "x-folder-id" not in req.headers  # omitted when folder_id is empty


@respx.mock
async def test_yandex_synthesize_includes_folder_id_when_set():
    import base64
    import json

    chunk = json.dumps({"result": {"audioChunk": {"data": base64.b64encode(b"x").decode()}}})
    route = respx.post(YANDEX_URL).mock(return_value=httpx.Response(200, text=chunk,
                                        headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, folder_id="fld123",
                                   url=YANDEX_URL, timeout=10)
        await backend.synthesize("тест", "ru")
    assert route.calls.last.request.headers["x-folder-id"] == "fld123"


@respx.mock
async def test_teratts_synthesize_builds_url_and_returns_audio():
    audio = b"\xff\xf3mp3-bytes"
    # The text is URL-encoded into the path (quote(text, safe="")).
    route = respx.get("http://tera.local/synthesize/%D0%BF%D1%80%D0%B8%D0%B2%D0%B5%D1%82").mock(
        return_value=httpx.Response(200, content=audio, headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = TeraTtsHttpBackend("http://tera.local/", client, timeout=10)
        mime, data = await backend.synthesize("привет", "ru")
    assert route.called
    assert mime == "audio/mpeg"
    assert data == audio


@respx.mock
async def test_teratts_synthesize_defaults_mime_when_header_absent():
    # No Content-Type header -> falls back to audio/mpeg.
    route = respx.get("http://tera.local/synthesize/hi").mock(
        return_value=httpx.Response(200, content=b"x"))
    async with httpx.AsyncClient() as client:
        backend = TeraTtsHttpBackend("http://tera.local", client, timeout=10)
        mime, _ = await backend.synthesize("hi", "ru")
    assert route.called
    assert mime == "audio/mpeg"


@respx.mock
async def test_teratts_synthesize_raises_on_non_2xx():
    respx.get("http://tera.local/synthesize/hi").mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as client:
        backend = TeraTtsHttpBackend("http://tera.local", client, timeout=10)
        with pytest.raises(httpx.HTTPStatusError):
            await backend.synthesize("hi", "ru")


# --- Piper _synth tests (inject a stub voice; no real model load) -----------

_STUB_RATE = 22050
_STUB_WIDTH = 2  # 16-bit
_STUB_CHANNELS = 1


class _StubVoice:
    """Stub PiperVoice. synthesize_wav(sentence, wav_file) writes a tiny WAV
    whose frame count is taken from a per-sentence map; sentences in `raises`
    raise to drive the `except Exception: continue` path. Unknown sentences
    write a default number of frames."""

    def __init__(self, frames_for=None, raises=(), default_frames=10):
        self.frames_for = frames_for or {}
        self.raises = set(raises)
        self.default_frames = default_frames

    def synthesize_wav(self, sentence, wav_file):
        if sentence in self.raises:
            raise RuntimeError("unpronounceable")
        n = self.frames_for.get(sentence, self.default_frames)
        wav_file.setnchannels(_STUB_CHANNELS)
        wav_file.setsampwidth(_STUB_WIDTH)
        wav_file.setframerate(_STUB_RATE)
        # Non-silent marker bytes so real audio is distinguishable from padding.
        wav_file.writeframes(b"\x11\x22" * n)


def _synth_wav(backend, monkeypatch, text):
    """Run _synth but capture the raw WAV instead of the MP3 transcode, so the
    silence padding / frame alignment is observable at the byte level."""
    monkeypatch.setattr("src.tts.wav_to_mp3", lambda wav_bytes, **kw: wav_bytes)
    return backend._synth(text)


def test_synth_silence_padding_is_whole_frames_and_off_by_value(monkeypatch):
    # Two short sentences; compare 0.4s vs 0.0s sentence_silence. The only
    # difference must be exactly one inter-sentence silence gap of
    # int(framerate*0.4) whole frames.
    text = "Раз. Два."
    voice0 = _StubVoice(default_frames=5)
    voice4 = _StubVoice(default_frames=5)
    b0 = PiperTtsBackend.from_voice(voice0, sentence_silence=0.0)
    b4 = PiperTtsBackend.from_voice(voice4, sentence_silence=0.4)

    wav0 = _synth_wav(b0, monkeypatch, text)
    wav4 = _synth_wav(b4, monkeypatch, text)

    with wave.open(io.BytesIO(wav0), "rb") as r0:
        rate = r0.getframerate()
        width, ch = r0.getsampwidth(), r0.getnchannels()
        data0 = r0.readframes(r0.getnframes())
    with wave.open(io.BytesIO(wav4), "rb") as r4:
        data4 = r4.readframes(r4.getnframes())

    frame = width * ch
    expected_silence_bytes = int(rate * 0.4) * frame
    assert len(data4) - len(data0) == expected_silence_bytes
    # The extra bytes are a whole number of frames (no misalignment).
    assert (len(data4) - len(data0)) % frame == 0
    # And they are actual silence (all zero), inserted between the two sentences.
    assert data4[: len(b"\x11\x22" * 5)] == data0[: len(b"\x11\x22" * 5)]
    silence = data4[len(b"\x11\x22" * 5) : len(b"\x11\x22" * 5) + expected_silence_bytes]
    assert silence == b"\x00" * expected_silence_bytes


def test_synth_all_unpronounceable_returns_valid_silent_clip(monkeypatch):
    # Every fragment raises -> except/continue path for all, framerate stays None
    # -> 22050/1/16 fallback fires. Must yield a parseable empty-but-valid WAV.
    text = "Раз. Два."
    voice = _StubVoice(raises={"Раз.", "Два."})
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.4)

    wav = _synth_wav(backend, monkeypatch, text)
    with wave.open(io.BytesIO(wav), "rb") as r:
        assert r.getframerate() == 22050
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getnframes() == 0  # nothing pronounceable -> silent (empty) clip


def test_synth_empty_frames_fragment_contributes_no_gap(monkeypatch):
    # One real sentence + one zero-frame sentence. The empty one must be skipped
    # before any silence gap is added (the `if not frames: continue` guard), so
    # the output equals exactly the single real sentence with no padding.
    text = "Раз. Два."
    real_frames = 7
    voice = _StubVoice(frames_for={"Раз.": real_frames, "Два.": 0})
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.4)

    wav = _synth_wav(backend, monkeypatch, text)
    with wave.open(io.BytesIO(wav), "rb") as r:
        data = r.readframes(r.getnframes())
    # Exactly the real sentence's audio, no silence gap appended for the empty one.
    assert data == b"\x11\x22" * real_frames


def test_decode_v3_audio_ndjson_multi_chunk_concatenation():
    import base64
    import json

    a, b = b"first-chunk", b"second-chunk"
    line1 = json.dumps({"result": {"audioChunk": {"data": base64.b64encode(a).decode()}}})
    line2 = json.dumps({"result": {"audioChunk": {"data": base64.b64encode(b).decode()}}})
    body = line1 + "\n" + line2
    assert _decode_v3_audio(body) == a + b


def test_decode_v3_audio_error_object_raises_runtimeerror():
    import json

    body = json.dumps({"error": {"message": "quota"}})
    with pytest.raises(RuntimeError) as exc:
        _decode_v3_audio(body)
    # The payload must surface, not be swallowed into empty audio.
    assert "quota" in str(exc.value)


def test_decode_v3_audio_json_array_of_objects_accepted():
    import base64
    import json

    a, b = b"chunkA", b"chunkB"
    body = json.dumps([
        {"result": {"audioChunk": {"data": base64.b64encode(a).decode()}}},
        {"result": {"audioChunk": {"data": base64.b64encode(b).decode()}}},
    ])
    assert _decode_v3_audio(body) == a + b
