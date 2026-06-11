import io
import wave

import httpx
import pytest
import respx

from src.audio_codec import to_playable, wav_to_mp3
from src.chime import build_ack_clip, make_ack_chime_mp3
from src.plugins.tts._ru_text import expand_units, sanitize_plus_stress
from src.plugins.tts.fishaudio import FISH_TTS_URL, FishAudioTtsBackend
from src.plugins.tts.piper import PiperTtsBackend
from src.plugins.tts.teratts import TeraTtsHttpBackend
from src.plugins.tts.yandex import (
    YANDEX_V3_URL,
    YandexTtsBackend,
    _chunk_for_v3,
    _decode_v3_audio,
    _split_oversized,
)
from src.tts import split_sentences

YANDEX_URL = YANDEX_V3_URL


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


def test_wav_to_mp3_stereo_16bit():
    out = wav_to_mp3(_make_wav(channels=2))
    assert out  # non-empty
    assert out[0] == 0xFF  # MP3 frame sync byte


def test_wav_to_mp3_mono_8khz_16bit():
    out = wav_to_mp3(_make_wav(sample_rate=8000))
    assert out  # non-empty
    assert out[0] == 0xFF  # MP3 frame sync byte


def test_wav_to_mp3_rejects_24bit_wav():
    # R-3: lameenc assumes 16-bit PCM; a 24-bit WAV used to be silently
    # mis-encoded into distorted audio. It must now raise loudly.
    frames = 100
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)  # 24-bit
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00\x00" * frames)
    with pytest.raises(ValueError) as exc:
        wav_to_mp3(buf.getvalue())
    assert "24-bit" in str(exc.value)


# --- to_playable (delivery-boundary format adaptation, R8) -------------------


def test_to_playable_mp3_passthrough():
    # Already speaker-decodable -> untouched bytes, same mime, no transcode.
    assert to_playable("audio/mpeg", b"\xff\xf3mp3") == ("audio/mpeg", b"\xff\xf3mp3")


def test_to_playable_flac_passthrough():
    assert to_playable("audio/flac", b"fLaC....") == ("audio/flac", b"fLaC....")


def test_to_playable_wav_transcodes_to_mp3():
    # WAV is not speaker-decodable -> transcoded to MP3 at the delivery boundary.
    wav = _make_wav()
    mime, out = to_playable("audio/wav", wav)
    assert mime == "audio/mpeg"
    assert out  # non-empty
    assert out[0] == 0xFF  # MP3 frame sync byte
    assert out != wav


def test_to_playable_unknown_mime_passthrough():
    # Unknown formats are served as-is (same lenient behavior as tts_url).
    assert to_playable("audio/ogg", b"OggS") == ("audio/ogg", b"OggS")


def test_make_ack_chime_mp3_is_deterministic_mp3():
    # The end-of-phrase ack chime is synthesized once and cached, so it must be
    # both a valid MP3 (frame sync byte) and byte-for-byte deterministic across
    # calls (same inputs -> identical bytes -> stable cache key).
    a = make_ack_chime_mp3()
    b = make_ack_chime_mp3()
    assert a  # non-empty
    assert a[0] == 0xFF  # MP3 frame sync byte
    assert a == b        # deterministic: identical bytes on every build


def test_make_ack_chime_mp3_edge_short_tone_no_gap():
    # tone_ms=1 makes n smaller than the default ~8 ms ramps and gap_ms=0 yields a
    # zero-length gap — pins the edge clamp (edge <= n // 2) and the n <= 0 guard:
    # must still produce a valid non-empty MP3, not raise.
    out = make_ack_chime_mp3(tone_ms=1, gap_ms=0)
    assert out  # non-empty
    assert out[0] == 0xFF  # MP3 frame sync byte


# --- build_ack_clip (file-based chime loading + synthesized fallback) ---------


@pytest.mark.parametrize("sound_path", ["", "   ", "/no/such/file.wav"])
def test_build_ack_clip_empty_or_missing_path_yields_synthesized_chime(sound_path):
    # make_ack_chime_mp3 is deterministic, so the fallback is byte-comparable.
    assert build_ack_clip(sound_path) == ("audio/mpeg", make_ack_chime_mp3())


def test_build_ack_clip_corrupt_wav_falls_back_to_synthesized_chime(tmp_path):
    # A .wav file that is not actually a WAV must NOT raise — playback must never
    # go silent — and falls back to the synthesized chime.
    bad = tmp_path / "chime.wav"
    bad.write_bytes(b"garbage")
    assert build_ack_clip(str(bad)) == ("audio/mpeg", make_ack_chime_mp3())


def test_build_ack_clip_wav_is_transcoded_to_mp3(tmp_path):
    wav = _make_wav()
    p = tmp_path / "chime.wav"
    p.write_bytes(wav)
    mime, audio = build_ack_clip(str(p))
    assert mime == "audio/mpeg"
    assert audio  # non-empty
    assert audio != wav        # transcoded, not served verbatim
    assert audio[0] == 0xFF    # MP3 frame sync byte


def test_build_ack_clip_uppercase_wav_extension_also_transcoded(tmp_path):
    # Extension matching is case-insensitive: "chime.WAV" is still a WAV.
    wav = _make_wav()
    p = tmp_path / "chime.WAV"
    p.write_bytes(wav)
    mime, audio = build_ack_clip(str(p))
    assert mime == "audio/mpeg"
    assert audio and audio != wav
    assert audio[0] == 0xFF


def test_build_ack_clip_flac_served_verbatim(tmp_path):
    raw = b"fLaC-not-really-flac"
    p = tmp_path / "chime.flac"
    p.write_bytes(raw)
    assert build_ack_clip(str(p)) == ("audio/flac", raw)


@pytest.mark.parametrize("filename", ["chime.mp3", "chime.xyz"])
def test_build_ack_clip_mp3_and_unknown_ext_served_verbatim(tmp_path, filename):
    # mp3 (and any unknown extension) is served verbatim as audio/mpeg.
    raw = b"\xff\xf3raw-bytes"
    p = tmp_path / filename
    p.write_bytes(raw)
    assert build_ack_clip(str(p)) == ("audio/mpeg", raw)


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


def test_chunk_for_v3_sentence_of_exactly_limit_is_one_chunk():
    # A single sentence of EXACTLY 250 chars must fit in one chunk (the boundary
    # is inclusive: len(s) > limit splits, len(s) == limit does not).
    s = "а" * 249 + "."
    assert len(s) == 250
    chunks = _chunk_for_v3(s)
    assert chunks == [s]
    assert all(len(c) <= 250 for c in chunks)


def test_chunk_for_v3_two_sentences_packing_to_exactly_limit_join_into_one_chunk():
    # Greedy packing joins sentences with a single space; when the joined length
    # is EXACTLY the limit they still pack into one chunk.
    s1 = "а" * 149 + "."   # 150 chars
    s2 = "б" * 98 + "."    # 99 chars; 150 + 1 (space) + 99 == 250
    assert len(s1) + 1 + len(s2) == 250
    chunks = _chunk_for_v3(f"{s1} {s2}")
    assert chunks == [f"{s1} {s2}"]
    assert all(len(c) <= 250 for c in chunks)


def test_chunk_for_v3_packing_one_char_over_limit_splits_into_two_chunks():
    # Same as above but the joined length would be limit+1 -> the second sentence
    # starts a new chunk.
    s1 = "а" * 149 + "."   # 150 chars
    s2 = "б" * 99 + "."    # 100 chars; 150 + 1 + 100 == 251
    assert len(s1) + 1 + len(s2) == 251
    chunks = _chunk_for_v3(f"{s1} {s2}")
    assert chunks == [s1, s2]
    assert all(len(c) <= 250 for c in chunks)


# --- _split_oversized (word-boundary splitting of over-limit fragments) ------


def test_split_oversized_word_of_exactly_limit_kept_whole():
    limit = 10
    big = "b" * limit  # exactly the limit -> NOT hard-sliced
    out = _split_oversized(f"aaa {big} ccc", limit)
    assert big in out  # the limit-length word survives as one piece
    assert all(len(p) <= limit for p in out)
    # No word lost or duplicated: the word sequence reconstructs exactly.
    assert " ".join(out).split() == ["aaa", big, "ccc"]


def test_split_oversized_flushes_around_hard_sliced_word():
    limit = 10
    big = "b" * (limit + 3)  # over the limit -> hard-sliced into limit-sized pieces
    out = _split_oversized(f"aa {big} cc", limit)
    assert all(len(p) <= limit for p in out)
    # Flush correctness: character-level reconstruction proves no word (or word
    # piece) was lost or duplicated across the flush boundaries.
    assert "".join(p.replace(" ", "") for p in out) == f"aa{big}cc"
    # The preceding word was flushed BEFORE the slices and the following word
    # starts a fresh accumulator after them.
    assert out == ["aa", "b" * limit, "bbb", "cc"]


def test_yandex_backend_requires_api_key():
    with pytest.raises(ValueError):
        YandexTtsBackend(None, api_key="", voice="zahar", role="neutral",
                         speed=1.0, timeout=5)


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
                                   timeout=10)
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
                                   timeout=10)
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
                                   timeout=10)
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


def _synth_wav(backend, text):
    """Run _synth and return its raw WAV output (the backend's native format
    since R8 — no MP3 transcode in synthesis), so the silence padding / frame
    alignment is observable at the byte level."""
    return backend._synth(text)


def test_synth_silence_padding_is_whole_frames_and_off_by_value():
    # Two short sentences; compare 0.4s vs 0.0s sentence_silence. The only
    # difference must be exactly one inter-sentence silence gap of
    # int(framerate*0.4) whole frames.
    text = "Раз. Два."
    voice0 = _StubVoice(default_frames=5)
    voice4 = _StubVoice(default_frames=5)
    b0 = PiperTtsBackend.from_voice(voice0, sentence_silence=0.0)
    b4 = PiperTtsBackend.from_voice(voice4, sentence_silence=0.4)

    wav0 = _synth_wav(b0, text)
    wav4 = _synth_wav(b4, text)

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


def test_synth_all_unpronounceable_returns_valid_silent_clip():
    # Every fragment raises -> except/continue path for all, framerate stays None
    # -> 22050/1/16 fallback fires. Must yield a parseable empty-but-valid WAV.
    text = "Раз. Два."
    voice = _StubVoice(raises={"Раз.", "Два."})
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.4)

    wav = _synth_wav(backend, text)
    with wave.open(io.BytesIO(wav), "rb") as r:
        assert r.getframerate() == 22050
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getnframes() == 0  # nothing pronounceable -> silent (empty) clip


def test_synth_empty_frames_fragment_contributes_no_gap():
    # One real sentence + one zero-frame sentence. The empty one must be skipped
    # before any silence gap is added (the `if not frames: continue` guard), so
    # the output equals exactly the single real sentence with no padding.
    text = "Раз. Два."
    real_frames = 7
    voice = _StubVoice(frames_for={"Раз.": real_frames, "Два.": 0})
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.4)

    wav = _synth_wav(backend, text)
    with wave.open(io.BytesIO(wav), "rb") as r:
        data = r.readframes(r.getnframes())
    # Exactly the real sentence's audio, no silence gap appended for the empty one.
    assert data == b"\x11\x22" * real_frames


async def test_piper_synthesize_returns_native_wav():
    # R8 contract: the backend returns its engine's NATIVE format (audio/wav),
    # NOT a device-ready MP3 — adapting to the speaker is the delivery
    # boundary's job (audio_codec.to_playable in pipeline.serve_audio).
    voice = _StubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    mime, audio = await backend.synthesize("Раз.", "ru")

    assert mime == "audio/wav"
    with wave.open(io.BytesIO(audio), "rb") as r:  # parseable, real WAV bytes
        assert r.getframerate() == _STUB_RATE
        assert r.getnchannels() == _STUB_CHANNELS
        assert r.getsampwidth() == _STUB_WIDTH
        assert r.readframes(r.getnframes()) == b"\x11\x22" * 5


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


def test_piper_synth_applies_adaptation_chain_in_order():
    # _synth must adapt the canonical "+vowel" text for espeak-ng/Piper:
    # stress_to_acute -> expand_units -> phonetic_ru. Order matters: only if
    # stress conversion runs BEFORE phonetic_ru does "чт+о" become "што́"
    # (the combining acute trails the vowel, so "что" still matches).
    voice = _RecordingStubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    _synth_wav(backend, "чт+о нового? 50% и м/с")

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
                                   timeout=10)
        await backend.synthesize("прив+ет: 50% и что", "ru")

    assert route.called
    sent_text = json.loads(route.calls.last.request.content)["text"]
    assert "прив+ет" in sent_text     # "+vowel" is Yandex-native, passes through
    assert "процентов" in sent_text   # "%" expanded
    assert "%" not in sent_text
    assert "што" not in sent_text     # no espeak phonetic hacks for cloud TTS
    assert "что" in sent_text         # the word stays unchanged


def test_piper_applies_russian_adaptation_chain():
    # Exact-text proof that the backend runs phonetic_ru(expand_units(
    # stress_to_acute(...))): the engine must receive precisely the adapted
    # string — "што" + combining acute U+0301 + expanded "%". Substring checks
    # alone could pass with a partially-deleted chain; equality cannot.
    voice = _RecordingStubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    _synth_wav(backend, "чт+о 50%")

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
                                   timeout=10)
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
                                   timeout=10)
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


# --- Fish Audio backend -------------------------------------------------------


def test_fishaudio_backend_requires_api_key():
    with pytest.raises(ValueError):
        FishAudioTtsBackend(None, api_key="", reference_id="", model="s2-pro",
                            speed=1.0, timeout=5)


@respx.mock
async def test_fishaudio_synthesize_posts_json_and_returns_audio():
    import json

    audio_bytes = b"\xff\xf3fish-mp3"
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=audio_bytes,
                                    headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="ref-1",
                                      model="s2-pro", speed=1.2, timeout=10)
        mime, audio = await backend.synthesize("привет", "ru")
    assert mime == "audio/mpeg"
    assert audio == audio_bytes
    req = route.calls.last.request
    # Auth is a Bearer key; the TTS model generation travels as a `model` header.
    assert req.headers["Authorization"] == "Bearer k"
    assert req.headers["model"] == "s2-pro"
    sent = json.loads(req.content)
    assert sent["text"] == "привет"
    assert sent["format"] == "mp3"
    assert sent["prosody"]["speed"] == 1.2
    assert sent["reference_id"] == "ref-1"


@respx.mock
async def test_fishaudio_synthesize_omits_reference_id_when_empty():
    import json

    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3"))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        await backend.synthesize("привет", "ru")
    assert route.called
    sent = json.loads(route.calls.last.request.content)
    # Empty reference_id -> the field is omitted entirely (fish.audio then uses
    # its default voice), not sent as "".
    assert "reference_id" not in sent


@respx.mock
async def test_fishaudio_synthesize_adapts_text():
    import json

    # Fish Audio takes plain text: units expand and the "+vowel" stress markup
    # is stripped entirely (a literal '+' could be voiced).
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3"))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        await backend.synthesize("прив+ет, 25°С", "ru")
    assert route.called
    sent_text = json.loads(route.calls.last.request.content)["text"]
    assert sent_text == "привет, 25градусов"
    assert "+" not in sent_text
    assert "привет" in sent_text          # stress markup stripped, plain vowel kept
    assert "градусов" in sent_text        # "°С" expanded


@respx.mock
async def test_fishaudio_synthesize_unvoiceable_input_makes_no_request():
    # Punctuation-only / empty input has nothing pronounceable: the backend must
    # serve empty audio WITHOUT POSTing to Fish Audio.
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3"))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        for text in ("...", "…", ""):
            mime, audio = await backend.synthesize(text, "ru")
            assert (mime, audio) == ("audio/mpeg", b"")
    assert not route.called


@respx.mock
async def test_fishaudio_synthesize_surfaces_error_body():
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(402, text='{"message":"insufficient credit"}'))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        with pytest.raises(RuntimeError) as exc:
            await backend.synthesize("привет", "ru")
    assert route.called
    assert "402" in str(exc.value)
    assert "insufficient credit" in str(exc.value)
