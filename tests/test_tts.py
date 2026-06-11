import io
import wave

import httpx
import pytest
import respx

from src.plugins.tts._ru_text import expand_units, sanitize_plus_stress
from src.tts import (
    PiperTtsBackend,
    TeraTtsHttpBackend,
    YandexTtsBackend,
    _chunk_for_v3,
    _decode_v3_audio,
    make_ack_chime_mp3,
    split_sentences,
    wav_to_mp3,
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


def test_make_ack_chime_mp3_is_deterministic_mp3():
    # The end-of-phrase ack chime is synthesized once and cached, so it must be
    # both a valid MP3 (frame sync byte) and byte-for-byte deterministic across
    # calls (same inputs -> identical bytes -> stable cache key).
    a = make_ack_chime_mp3()
    b = make_ack_chime_mp3()
    assert a  # non-empty
    assert a[0] == 0xFF  # MP3 frame sync byte
    assert a == b        # deterministic: identical bytes on every build


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


# --- _chunk_for_v3 (v3 250-char request limit) ------------------------------

# The real 325-char reply that triggered the original 400 Bad Request from
# Yandex v3 utteranceSynthesis.
_LONG_REPLY = (
    "Ишь ты, так вот што записано-то. До конца недели у вас: завтра, в среду, "
    "с полуночи воду отключат на часок, в четверг Влада в студии в пять часов "
    "вечера, потом в субботу день рождения Хлои да ещё в десять вечера Оземпик "
    "какой-то, а в понедельник счётчики снимать надо на казарменном в восемь "
    "вечера. Вот вся программа, барин."
)


def test_chunk_for_v3_short_text_single_chunk():
    result = _chunk_for_v3("Привет. Как дела?")
    assert len(result) == 1
    assert result[0] == "Привет. Как дела?"
    assert len(result[0]) <= 250


def test_chunk_for_v3_long_text_splits_under_limit():
    chunks = _chunk_for_v3(_LONG_REPLY)
    assert len(chunks) >= 2
    assert all(len(c) <= 250 for c in chunks)
    joined = " ".join(chunks)
    for word in ("Хлои", "Оземпик", "счётчики"):
        assert word in joined


def test_chunk_for_v3_oversized_single_sentence_word_split():
    text = " ".join(["слово"] * 60)  # no terminal punctuation, well over 250
    chunks = _chunk_for_v3(text)
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)


def test_chunk_for_v3_oversized_single_word_hard_split():
    text = "я" * 600
    chunks = _chunk_for_v3(text)
    assert all(len(c) <= 250 for c in chunks)
    assert len(chunks) == 3


def test_chunk_for_v3_empty_and_symbol_only():
    assert _chunk_for_v3("") == []
    assert _chunk_for_v3("...") == []


def test_yandex_backend_requires_api_key():
    with pytest.raises(ValueError):
        YandexTtsBackend(None, api_key="", voice="zahar", role="neutral",
                         speed=1.0, url="http://x", timeout=5)


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
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        # Canonical LLM->TTS text: the model's own "+vowel" stress notation,
        # which is also Yandex SpeechKit's native markup.
        mime, audio = await backend.synthesize("прив+ет", "ru")
    assert mime == "audio/mpeg"
    assert audio == audio_bytes
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Api-Key k"
    sent = json.loads(route.calls.last.request.content)
    assert {"voice": "zahar"} in sent["hints"]
    assert {"role": "neutral"} in sent["hints"]
    assert sent["outputAudioSpec"]["containerAudio"]["containerAudioType"] == "MP3"
    assert "прив+ет" in sent["text"]   # "+vowel" markup passes through untouched


@respx.mock
async def test_yandex_synthesize_chunks_long_text_and_concatenates_audio():
    import base64
    import json

    # Expected number of chunks for the 325-char reply, measured on the adapted text
    # exactly as synthesize() does, so the assertion tracks the real chunker.
    num_calls = len(_chunk_for_v3(sanitize_plus_stress(expand_units(_LONG_REPLY))))
    assert num_calls >= 2  # the long reply must split into multiple requests

    # Distinct audio per call so a wrong concatenation order would fail the test.
    audios = [b"\xff\xf3chunk-%d" % i for i in range(num_calls)]
    responses = [
        httpx.Response(
            200,
            text=json.dumps({"result": {"audioChunk": {"data": base64.b64encode(a).decode()}}}),
            headers={"Content-Type": "application/json"},
        )
        for a in audios
    ]
    route = respx.post(YANDEX_URL).mock(side_effect=responses)
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        mime, audio = await backend.synthesize(_LONG_REPLY, "ru")

    assert mime == "audio/mpeg"
    # (a) one POST per chunk.
    assert route.call_count == num_calls
    # (b) every request stays within the v3 250-char limit.
    for call in route.calls:
        sent_text = json.loads(call.request.content)["text"]
        assert len(sent_text) <= 250
    # (c) audio is the per-call audio concatenated IN ORDER.
    assert audio == b"".join(audios)


@respx.mock
async def test_yandex_synthesize_surfaces_error_body():
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(400, text='{"error":"text is too long"}'))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        with pytest.raises(Exception) as exc:
            await backend.synthesize("привет", "ru")
    assert route.called
    assert "text is too long" in str(exc.value)
    assert "400" in str(exc.value)


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


# --- Backend-side adaptation chains (canonical "+stress" contract, R3) -------


class _RecordingStubVoice(_StubVoice):
    """_StubVoice that also records every sentence passed to synthesize_wav,
    so tests can assert the exact adapted text the engine receives."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentences = []

    def synthesize_wav(self, sentence, wav_file):
        self.sentences.append(sentence)
        super().synthesize_wav(sentence, wav_file)


def test_piper_synth_applies_adaptation_chain_in_order(monkeypatch):
    # _synth must adapt the canonical "+vowel" text for espeak-ng/Piper:
    # stress_to_acute -> expand_units -> phonetic_ru. Order matters: only if
    # stress conversion runs BEFORE phonetic_ru does "чт+о" become "што́"
    # (the combining acute trails the vowel, so "что" still matches).
    voice = _RecordingStubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    _synth_wav(backend, monkeypatch, "чт+о нового? 50% и м/с")

    joined = " ".join(voice.sentences)
    assert "што́" in joined          # "чт+о" -> stressed phonetic "што" + combining acute
    assert "процентов" in joined          # "%" expanded
    assert "метров в секунду" in joined   # "м/с" expanded
    assert "+" not in joined              # no "+" leaks to the engine


@respx.mock
async def test_teratts_synthesize_applies_adaptation_chain():
    from urllib.parse import unquote

    # TeraTTS gets the same Piper-style adaptation; the adapted text is
    # URL-encoded into the path, so match the route with a wildcard path.
    route = respx.get(url__regex=r"http://tera\.local/synthesize/.+").mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3",
                                    headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = TeraTtsHttpBackend("http://tera.local", client, timeout=10)
        mime, _ = await backend.synthesize("чт+о, 50%?", "ru")

    assert route.called
    assert mime == "audio/mpeg"
    raw_url = str(route.calls.last.request.url)
    decoded = unquote(raw_url)
    assert "што́" in decoded      # stress applied before phonetic_ru
    assert "процентов" in decoded      # "%" expanded
    assert "+" not in decoded          # no "+" leaks into the request text
    # The combining acute (U+0301) must travel percent-encoded, never raw.
    assert "́" not in raw_url


@respx.mock
async def test_yandex_synthesize_adapts_text_keeps_native_stress():
    import json

    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text="{}",
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        await backend.synthesize("прив+ет: 50% и что", "ru")

    assert route.called
    sent_text = json.loads(route.calls.last.request.content)["text"]
    assert "прив+ет" in sent_text     # "+vowel" is Yandex-native, passes through
    assert "процентов" in sent_text   # "%" expanded
    assert "%" not in sent_text
    assert "што" not in sent_text     # no espeak phonetic hacks for cloud TTS
    assert "что" in sent_text         # the word stays unchanged


def test_piper_applies_russian_adaptation_chain(monkeypatch):
    # Exact-text proof that the backend runs phonetic_ru(expand_units(
    # stress_to_acute(...))): the engine must receive precisely the adapted
    # string — "што" + combining acute U+0301 + expanded "%". Substring checks
    # alone could pass with a partially-deleted chain; equality cannot.
    voice = _RecordingStubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    _synth_wav(backend, monkeypatch, "чт+о 50%")

    assert voice.sentences == ["што́ 50процентов"]


@respx.mock
async def test_teratts_applies_russian_adaptation_chain():
    from urllib.parse import quote

    # Exact-URL proof that TeraTTS runs the Piper-style chain: the GET path must
    # contain the URL-encoded adapted text. Encoded via quote(safe="") exactly
    # like the backend, so the expectation tracks the encoding, not hardcoded
    # percent-escapes (note the combining acute U+0301 after "о").
    route = respx.get(url__regex=r"http://tera\.local/synthesize/.+").mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3",
                                    headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = TeraTtsHttpBackend("http://tera.local", client, timeout=10)
        await backend.synthesize("чт+о 50%", "ru")

    assert route.called
    expected = quote("што́ 50процентов", safe="")
    assert expected in str(route.calls.last.request.url)


@respx.mock
async def test_yandex_applies_russian_adaptation_chain():
    import json

    # Exact-payload proof that Yandex runs sanitize_plus_stress(expand_units(...)):
    # the native "+vowel" stress survives, units expand, and no espeak phonetic
    # rewrite happens for cloud TTS.
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text="{}",
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        await backend.synthesize("чт+о 50%", "ru")

    assert route.called
    sent_text = json.loads(route.calls.last.request.content)["text"]
    assert sent_text == "чт+о 50процентов"


@respx.mock
async def test_yandex_synthesize_unvoiceable_input_makes_no_request():
    # Punctuation-only / empty input chunks to nothing pronounceable: the
    # backend must serve empty audio WITHOUT POSTing to Yandex (which would 400).
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text="{}",
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0,
                                   url=YANDEX_URL, timeout=10)
        for text in ("…", ""):
            mime, audio = await backend.synthesize(text, "ru")
            assert (mime, audio) == ("audio/mpeg", b"")
    assert not route.called


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
