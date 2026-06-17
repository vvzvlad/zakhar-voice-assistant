import io
import wave

import httpx
import numpy as np
import pytest
import respx

from src.audio_codec import to_playable, wav_to_mp3
from src.chime import build_ack_clip, make_ack_chime_mp3
from src.plugins.tts._ru_text import expand_units, sanitize_plus_stress
from src.plugins.tts.fishaudio import FISH_TTS_URL, FishAudioTtsBackend
from src.plugins.tts.piper import (
    PiperConfig,
    PiperProvider,
    PiperTtsBackend,
    _list_piper_voices,
)
from src.plugins.tts.silero import (
    V4_RU_SPEAKERS,
    SileroTtsBackend,
    SileroTtsConfig,
    SileroTtsProvider,
    _list_silero_models,
)
from src.plugins.tts.yandex import (
    YANDEX_V3_URL,
    YandexTtsBackend,
    _aiter_v3_audio,
    _chunk_for_v3,
    _decode_v3_audio,
    _split_oversized,
)
from src.tts import TtsBackend, split_sentences

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


# --- Piper _synth tests (inject a stub voice; no real model load) -----------

_STUB_RATE = 22050
_STUB_WIDTH = 2  # 16-bit
_STUB_CHANNELS = 1


class _StubAudioChunk:
    """Lightweight stand-in for Piper's AudioChunk, carrying the fields the
    streaming path reads."""

    def __init__(self, pcm):
        self.audio_int16_bytes = pcm
        self.sample_rate = _STUB_RATE
        self.sample_width = _STUB_WIDTH
        self.sample_channels = _STUB_CHANNELS


class _StubVoice:
    """Stub PiperVoice. synthesize_wav(sentence, wav_file) writes a tiny WAV
    whose frame count is taken from a per-sentence map; sentences in `raises`
    raise to drive the `except Exception: continue` path. Unknown sentences
    write a default number of frames. synthesize(sentence) is the streaming
    generator counterpart, yielding _StubAudioChunk blocks driven by the same maps."""

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

    def synthesize(self, sentence):
        # Streaming generator: a sentence in `raises` raises (skip path); otherwise
        # yields one AudioChunk-like block of marker PCM driven by the frame map.
        if sentence in self.raises:
            raise RuntimeError("unpronounceable")
        n = self.frames_for.get(sentence, self.default_frames)
        yield _StubAudioChunk(b"\x11\x22" * n)


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


async def test_piper_synthesize_stream_yields_mp3():
    # Native streaming: multi-sentence input -> audio/mpeg with real (lameenc)
    # MP3 bytes flowing incrementally. The concatenated stream is a valid MP3.
    voice = _StubVoice(default_frames=200)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    mime, chunks = await backend.synthesize_stream("Раз. Два.", "ru")
    got = [c async for c in chunks]

    assert mime == "audio/mpeg"
    assert got  # at least one MP3 block was emitted
    joined = b"".join(got)
    assert joined  # non-empty MP3 byte stream
    assert joined[0] == 0xFF  # MP3 frame sync byte (like test_wav_to_mp3_produces_mp3_frame)


async def test_piper_synthesize_stream_unvoiceable_empty_stream_no_encoder():
    # "…" is dropped by split_sentences -> nothing to synthesize: the stream yields
    # NOTHING and no encoder is ever built (no trailing flush frames either).
    voice = _StubVoice(default_frames=5)
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.4)

    mime, chunks = await backend.synthesize_stream("…", "ru")
    got = [c async for c in chunks]

    assert mime == "audio/mpeg"
    assert got == []


async def test_piper_synthesize_stream_skips_raising_sentence():
    # One sentence raises in synthesize() (skip path), the other still produces
    # audio: the stream is non-empty and valid despite the skipped fragment.
    voice = _StubVoice(frames_for={"Раз.": 200}, raises={"Два."})
    backend = PiperTtsBackend.from_voice(voice, sentence_silence=0.0)

    mime, chunks = await backend.synthesize_stream("Раз. Два.", "ru")
    got = [c async for c in chunks]

    assert mime == "audio/mpeg"
    joined = b"".join(got)
    assert joined  # the surviving sentence still produced MP3
    assert joined[0] == 0xFF  # MP3 frame sync byte


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


# --- synthesize_stream (streaming synthesis contract) -------------------------


def _astream(*chunks):
    """Async byte generator for respx streaming response bodies (httpx keeps
    the chunk boundaries, so 'delivered as yielded' is observable)."""

    async def _gen():
        for c in chunks:
            yield c

    return _gen()


@respx.mock
async def test_fishaudio_synthesize_stream_yields_chunks_with_same_request():
    import json

    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=_astream(b"\xff\xf3aa", b"bbb")))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="ref-1",
                                      model="s2-pro", speed=1.2, timeout=10)
        mime, chunks = await backend.synthesize_stream("привет", "ru")
        got = [c async for c in chunks]
    assert mime == "audio/mpeg"
    # Chunks are delivered exactly as the server yields them, in order.
    assert got == [b"\xff\xf3aa", b"bbb"]
    # The streamed request carries the SAME headers/payload as the buffered one.
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer k"
    assert req.headers["model"] == "s2-pro"
    sent = json.loads(req.content)
    assert sent["text"] == "привет"
    assert sent["format"] == "mp3"
    assert sent["prosody"]["speed"] == 1.2
    assert sent["reference_id"] == "ref-1"


@respx.mock
async def test_fishaudio_synthesize_stream_error_raises_before_iteration():
    # An HTTP error must raise from synthesize_stream itself (before the
    # iterator is returned), surfacing the diagnostic body like the buffered path.
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(402, text='{"message":"insufficient credit"}'))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        with pytest.raises(RuntimeError) as exc:
            await backend.synthesize_stream("привет", "ru")
    assert route.called
    assert "402" in str(exc.value)
    assert "insufficient credit" in str(exc.value)


@respx.mock
async def test_fishaudio_synthesize_stream_unvoiceable_empty_stream_no_request():
    route = respx.post(FISH_TTS_URL).mock(
        return_value=httpx.Response(200, content=b"\xff\xf3mp3"))
    async with httpx.AsyncClient() as client:
        backend = FishAudioTtsBackend(client, api_key="k", reference_id="",
                                      model="s2-pro", speed=1.0, timeout=10)
        mime, chunks = await backend.synthesize_stream("…", "ru")
        got = [c async for c in chunks]
    assert (mime, got) == ("audio/mpeg", [])
    assert not route.called


@respx.mock
async def test_yandex_synthesize_stream_yields_one_audio_block_per_post_in_order():
    import base64
    import json

    # Same chunking as the buffered path, measured on the adapted text.
    num_calls = len(_chunk_for_v3(sanitize_plus_stress(expand_units(_LONG_REPLY))))
    assert num_calls >= 2

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
                                   role="neutral", speed=1.0, timeout=10)
        mime, chunks = await backend.synthesize_stream(_LONG_REPLY, "ru")
        # Only the FIRST request ran eagerly (the connect/auth error gate);
        # the rest are synthesized lazily as the stream is consumed.
        assert route.call_count == 1
        got = [c async for c in chunks]

    assert mime == "audio/mpeg"
    assert route.call_count == num_calls
    # One decoded MP3 block per POST, yielded in request order.
    assert got == audios


@respx.mock
async def test_yandex_synthesize_stream_first_request_error_raises_before_iterator():
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(400, text='{"error":"text is too long"}'))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, timeout=10)
        with pytest.raises(Exception) as exc:
            await backend.synthesize_stream("привет", "ru")
    assert route.called
    assert "400" in str(exc.value)
    assert "text is too long" in str(exc.value)


@respx.mock
async def test_yandex_synthesize_stream_unvoiceable_empty_stream_no_request():
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text="{}",
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, timeout=10)
        mime, chunks = await backend.synthesize_stream("…", "ru")
        got = [c async for c in chunks]
    assert (mime, got) == ("audio/mpeg", [])
    assert not route.called


@respx.mock
async def test_yandex_synthesize_stream_decodes_multiple_audiochunks_in_one_response():
    import base64
    import json

    # A single short reply (one <=250-char request) whose v3 response is a stream
    # of MULTIPLE audioChunk objects (NDJSON). True intra-request streaming must
    # yield each decoded chunk as its own block, all from ONE POST.
    blocks = [b"\xff\xf3frame-%d" % i for i in range(3)]
    body = "\n".join(
        json.dumps({"result": {"audioChunk": {"data": base64.b64encode(b).decode()}}})
        for b in blocks
    )
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text=body,
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, timeout=10)
        mime, chunks = await backend.synthesize_stream("Привет. Как дела?", "ru")
        got = [c async for c in chunks]
    assert mime == "audio/mpeg"
    # One short reply -> exactly one POST...
    assert route.call_count == 1
    # ...but three separate decoded MP3 blocks, in arrival order.
    assert got == blocks


@respx.mock
async def test_yandex_synthesize_stream_concatenation_matches_buffered():
    import base64
    import json

    # Same multi-chunk reply, same per-request bodies for both paths: the streamed
    # bytes (joined) must be byte-identical to the buffered synthesize() bytes.
    num_calls = len(_chunk_for_v3(sanitize_plus_stress(expand_units(_LONG_REPLY))))
    assert num_calls >= 2

    # Each request returns several audioChunks so the per-request decode itself is
    # multi-block; the buffered path concatenates them, the stream yields them.
    def _body(call_idx: int) -> str:
        parts = [b"\xff\xf3c%d-%d" % (call_idx, j) for j in range(2)]
        return "\n".join(
            json.dumps({"result": {"audioChunk": {"data": base64.b64encode(p).decode()}}})
            for p in parts
        )

    bodies = [_body(i) for i in range(num_calls)]
    responses = [
        httpx.Response(200, text=b, headers={"Content-Type": "application/json"})
        for b in bodies
    ]

    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, timeout=10)
        respx.post(YANDEX_URL).mock(side_effect=[
            httpx.Response(200, text=b, headers={"Content-Type": "application/json"})
            for b in bodies
        ])
        _, buffered = await backend.synthesize(_LONG_REPLY, "ru")

        respx.post(YANDEX_URL).mock(side_effect=responses)
        _, chunks = await backend.synthesize_stream(_LONG_REPLY, "ru")
        streamed = b"".join([c async for c in chunks])

    assert streamed == buffered


@respx.mock
async def test_yandex_synthesize_stream_error_object_mid_stream_raises():
    import base64
    import json

    # A response that streams an audioChunk THEN an error object: the first block
    # must be delivered, then the iterator must raise naming the error.
    good = b"\xff\xf3frame-0"
    body = (
        json.dumps({"result": {"audioChunk": {"data": base64.b64encode(good).decode()}}})
        + "\n"
        + json.dumps({"error": "synthesis failed midway"})
    )
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, text=body,
                                    headers={"Content-Type": "application/json"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   role="neutral", speed=1.0, timeout=10)
        mime, chunks = await backend.synthesize_stream("Привет. Как дела?", "ru")
        it = chunks.__aiter__()
        first = await it.__anext__()
        assert first == good
        with pytest.raises(RuntimeError) as exc:
            await it.__anext__()
    assert route.called
    assert "synthesis failed midway" in str(exc.value)


async def test_aiter_v3_audio_buffers_object_split_across_reads():
    import base64
    import json

    # The decoder must buffer a JSON object split across two network reads: a
    # tiny fake response whose aiter_text() hands back half an object, then the
    # rest, must still yield exactly one decoded block.
    payload = b"\xff\xf3split-frame"
    obj = json.dumps({"result": {"audioChunk": {"data": base64.b64encode(payload).decode()}}})
    half = len(obj) // 2

    class _FakeResp:
        async def aiter_text(self):
            yield obj[:half]
            yield obj[half:]

    got = [block async for block in _aiter_v3_audio(_FakeResp())]
    assert got == [payload]


async def test_default_synthesize_stream_adapter_yields_single_chunk():
    # The TtsBackend default adapter wraps buffered synthesize() in a
    # single-chunk stream, so every backend is streamable.
    class _Buffered(TtsBackend):
        async def synthesize(self, text, lang="ru"):
            return ("audio/mpeg", b"WHOLE-CLIP")

    mime, chunks = await _Buffered().synthesize_stream("привет")
    assert mime == "audio/mpeg"
    assert [c async for c in chunks] == [b"WHOLE-CLIP"]


async def test_default_synthesize_stream_adapter_skips_empty_audio():
    # Empty buffered audio (unvoiceable text) yields an EMPTY stream, not one
    # empty chunk.
    class _Silent(TtsBackend):
        async def synthesize(self, text, lang="ru"):
            return ("audio/mpeg", b"")

    mime, chunks = await _Silent().synthesize_stream("…")
    assert mime == "audio/mpeg"
    assert [c async for c in chunks] == []


# --- _list_piper_voices / PiperProvider.options (local voice enumeration) -----


def _make_piper_voice(dir_path, name: str, *, with_json: bool = True):
    """Create a fake Piper voice in dir_path: <name>.onnx and (optionally) its
    sibling <name>.onnx.json config. Returns the .onnx file path."""
    onnx = dir_path / f"{name}.onnx"
    onnx.write_bytes(b"onnx")
    if with_json:
        (dir_path / f"{name}.onnx.json").write_text("{}")
    return onnx


def test_list_piper_voices_returns_paired_onnx_sorted_by_label(tmp_path):
    _make_piper_voice(tmp_path, "b")
    _make_piper_voice(tmp_path, "a")
    out = _list_piper_voices(str(tmp_path))
    assert out == [
        {"value": str(tmp_path / "a.onnx"), "label": "a"},
        {"value": str(tmp_path / "b.onnx"), "label": "b"},
    ]
    # The value points at the .onnx file create() loads (PiperVoice.load(value, ...)).
    assert all(o["value"].endswith(".onnx") for o in out)


def test_list_piper_voices_excludes_onnx_without_sibling_json(tmp_path):
    _make_piper_voice(tmp_path, "good")
    _make_piper_voice(tmp_path, "orphan", with_json=False)
    out = _list_piper_voices(str(tmp_path))
    assert [o["label"] for o in out] == ["good"]


def test_list_piper_voices_excludes_hidden_entries(tmp_path):
    _make_piper_voice(tmp_path, ".x")  # ".x.onnx" + ".x.onnx.json"
    _make_piper_voice(tmp_path, "visible")
    out = _list_piper_voices(str(tmp_path))
    assert [o["label"] for o in out] == ["visible"]


def test_list_piper_voices_missing_dir_returns_empty(tmp_path):
    assert _list_piper_voices(str(tmp_path / "nope")) == []


def test_piper_provider_options_voice_path_scans_configured_dir(tmp_path):
    onnx = _make_piper_voice(tmp_path, "a")
    _make_piper_voice(tmp_path, "b")
    # options() ignores deps (no network); pass None.
    out = PiperProvider().options("voice_path", PiperConfig(voice_path=str(onnx)), None)
    assert out == [
        {"value": str(tmp_path / "a.onnx"), "label": "a"},
        {"value": str(tmp_path / "b.onnx"), "label": "b"},
    ]


def test_piper_provider_options_other_field_returns_none(tmp_path):
    onnx = _make_piper_voice(tmp_path, "a")
    assert PiperProvider().options("sentence_silence", PiperConfig(voice_path=str(onnx)), None) is None


# --- Silero _synth tests (inject a stub model; no torch / real model load) ----


class _FakeTensor:
    """Stands in for the torch tensor apply_tts returns: exposes .numpy()."""
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


class _StubSileroModel:
    """Stub Silero model. apply_tts records its kwargs and returns a _FakeTensor
    of `samples_for[text]` (default `default_samples`) constant samples; texts in
    `raises` raise ValueError to drive the except/continue path."""
    def __init__(self, samples_for=None, raises=(), default_samples=10):
        self.samples_for = samples_for or {}
        self.raises = set(raises)
        self.default_samples = default_samples
        self.calls = []

    def apply_tts(self, text, speaker, sample_rate, put_accent=True, put_yo=True):
        self.calls.append({"text": text, "speaker": speaker, "sample_rate": sample_rate,
                           "put_accent": put_accent, "put_yo": put_yo})
        if text in self.raises:
            raise ValueError("unsupported symbols")
        n = self.samples_for.get(text, self.default_samples)
        return _FakeTensor(np.full(n, 0.5, dtype=np.float32))


def test_silero_synth_silence_padding_is_whole_frames_and_off_by_value():
    # Two short sentences; compare 0.4s vs 0.0s sentence_silence. The only
    # difference must be exactly one inter-sentence silence gap of
    # int(sample_rate*0.4) whole frames (mono 16-bit -> 2 bytes/frame).
    text = "Раз. Два."
    rate = 24000
    model0 = _StubSileroModel(default_samples=5)
    model4 = _StubSileroModel(default_samples=5)
    b0 = SileroTtsBackend.from_model(model0, sample_rate=rate, sentence_silence=0.0)
    b4 = SileroTtsBackend.from_model(model4, sample_rate=rate, sentence_silence=0.4)

    wav0 = b0._synth(text)
    wav4 = b4._synth(text)

    with wave.open(io.BytesIO(wav0), "rb") as r0:
        assert r0.getframerate() == rate
        width, ch = r0.getsampwidth(), r0.getnchannels()
        data0 = r0.readframes(r0.getnframes())
    with wave.open(io.BytesIO(wav4), "rb") as r4:
        data4 = r4.readframes(r4.getnframes())

    frame = width * ch  # mono 16-bit -> 2 bytes
    expected_silence_bytes = int(rate * 0.4) * frame
    assert len(data4) - len(data0) == expected_silence_bytes
    # The extra bytes are a whole number of frames (no misalignment).
    assert (len(data4) - len(data0)) % frame == 0
    # The extra bytes are actual silence (all zero), inserted between sentences.
    first_sentence_bytes = 5 * 2  # 5 samples * 2 bytes
    silence = data4[first_sentence_bytes: first_sentence_bytes + expected_silence_bytes]
    assert silence == b"\x00" * expected_silence_bytes


def test_silero_synth_all_unpronounceable_returns_valid_silent_clip():
    # Every fragment raises -> except/continue for all -> pcm stays empty. Must
    # yield a parseable empty-but-valid WAV at the configured sample rate.
    text = "Раз. Два."
    rate = 24000
    model = _StubSileroModel(raises={"Раз.", "Два."})
    backend = SileroTtsBackend.from_model(model, sample_rate=rate, sentence_silence=0.4)

    wav = backend._synth(text)
    with wave.open(io.BytesIO(wav), "rb") as r:
        assert r.getframerate() == rate
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getnframes() == 0  # nothing pronounceable -> silent (empty) clip


async def test_silero_synthesize_returns_native_wav():
    # R8 contract: the backend returns its engine's NATIVE format (audio/wav) at
    # the configured sample rate, NOT a device-ready MP3.
    rate = 24000
    model = _StubSileroModel(default_samples=5)
    backend = SileroTtsBackend.from_model(model, sample_rate=rate, sentence_silence=0.0)

    mime, audio = await backend.synthesize("Раз.", "ru")

    assert mime == "audio/wav"
    with wave.open(io.BytesIO(audio), "rb") as r:  # parseable, real WAV bytes
        assert r.getframerate() == rate
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2


def test_silero_synth_keeps_stress_expands_units_no_phonetic_mangling():
    # _synth runs sanitize_plus_stress(expand_units(...)): the native "+vowel"
    # stress survives, units expand, and NO espeak phonetic rewrite happens
    # (Silero pronounces "что" correctly).
    model = _StubSileroModel(default_samples=5)
    backend = SileroTtsBackend.from_model(model, sample_rate=24000, sentence_silence=0.0)

    backend._synth("Что там, прив+ет, 5%.")

    sent = model.calls[0]["text"]
    assert "прив+ет" in sent          # "+vowel" markup kept (Silero-native)
    assert "процентов" in sent        # "%" expanded
    assert "%" not in sent
    assert "Что" in sent              # no phonetic mangling for Silero
    assert "Што" not in sent


def test_silero_synth_propagates_speaker_rate_accent_yo():
    # The configured speaker / sample_rate / put_accent / put_yo must reach apply_tts.
    model = _StubSileroModel(default_samples=5)
    backend = SileroTtsBackend.from_model(
        model, speaker="baya", sample_rate=8000, put_accent=False, put_yo=False
    )

    backend._synth("Привет.")

    call = model.calls[0]
    assert call["speaker"] == "baya"
    assert call["sample_rate"] == 8000
    assert call["put_accent"] is False
    assert call["put_yo"] is False


def test_silero_provider_options(tmp_path):
    deps = None  # options() for these fields ignores deps (no network)
    # speaker -> the static v4_ru roster includes "xenia".
    speakers = SileroTtsProvider().options("speaker", SileroTtsConfig(), deps)
    assert "xenia" in speakers
    # sample_rate -> a list of {"value", "label"} whose values include 48000.
    rates = SileroTtsProvider().options("sample_rate", SileroTtsConfig(), deps)
    assert 48000 in [o["value"] for o in rates]
    # model_path -> local-disk scan keeps only .pt files.
    (tmp_path / "voice.pt").write_bytes(b"pt")
    (tmp_path / "notes.txt").write_text("not a model")
    out = _list_silero_models(str(tmp_path))
    assert out == [{"value": str(tmp_path / "voice.pt"), "label": "voice"}]
    # And the provider routes model_path through that scan.
    cfg = SileroTtsConfig(model_path=str(tmp_path / "x.pt"))
    via_provider = SileroTtsProvider().options("model_path", cfg, deps)
    assert via_provider == [{"value": str(tmp_path / "voice.pt"), "label": "voice"}]


def test_silero_provider_describe_includes_speaker():
    assert SileroTtsProvider().describe(SileroTtsConfig()) == "silero/silero_tts_v4_ru.pt/xenia"


def test_silero_config_defaults():
    cfg = SileroTtsConfig()
    assert cfg.speaker == "xenia"
    assert cfg.sample_rate == 48000
    assert cfg.put_accent is True
    assert cfg.put_yo is True
    assert cfg.sentence_silence == 0.4
    assert cfg.model_path == "models/silero_tts_v4_ru.pt"


def test_silero_v4_ru_speakers_roster():
    assert V4_RU_SPEAKERS == ["aidar", "baya", "eugene", "kseniya", "xenia", "random"]
