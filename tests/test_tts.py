import io
import wave

import httpx
import pytest
import respx

from src.tts import (
    YandexTtsBackend,
    split_sentences,
    wav_to_mp3,
    yandex_stress_markup,
)

YANDEX_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"


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
        YandexTtsBackend(None, api_key="", voice="zahar", emotion="neutral",
                         speed=1.0, folder_id="", url="http://x", timeout=5)


@respx.mock
async def test_yandex_synthesize_posts_mp3_and_returns_audio():
    route = respx.post(YANDEX_URL).mock(
        return_value=httpx.Response(200, content=b"\xff\xf3audio",
                                    headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   emotion="neutral", speed=1.0, folder_id="",
                                   url=YANDEX_URL, timeout=10)
        mime, audio = await backend.synthesize("приве́т", "ru")
    assert mime == "audio/mpeg"
    assert audio == b"\xff\xf3audio"
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Api-Key k"
    body = req.content.decode()
    assert "voice=zahar" in body
    assert "lang=ru-RU" in body
    assert "format=mp3" in body
    assert "%2B" in body          # the stress "+" (url-encoded), i.e. "прив+ет"
    assert "folderId" not in body  # omitted when folder_id is empty


@respx.mock
async def test_yandex_synthesize_includes_folder_id_when_set():
    route = respx.post(YANDEX_URL).mock(return_value=httpx.Response(200, content=b"x",
                                        headers={"Content-Type": "audio/mpeg"}))
    async with httpx.AsyncClient() as client:
        backend = YandexTtsBackend(client, api_key="k", voice="zahar",
                                   emotion="neutral", speed=1.0, folder_id="fld123",
                                   url=YANDEX_URL, timeout=10)
        await backend.synthesize("тест", "ru")
    assert "folderId=fld123" in route.calls.last.request.content.decode()
