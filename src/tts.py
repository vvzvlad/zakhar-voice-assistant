"""Text-to-speech backends (pluggable; TeraTTS HTTP is the default)."""

import asyncio
import io
import re
import wave
from abc import ABC, abstractmethod
from urllib.parse import quote

import httpx
from loguru import logger

from src.settings import settings


# Yandex SpeechKit marks word stress with "+" BEFORE the stressed vowel (e.g.
# "прив+ет"). The pipeline's text post-processing (src/text.py) has already turned
# the model's "+vowel" notation into "vowel" + combining acute accent (U+0301)
# placed AFTER the vowel, for espeak/Piper. Translate it back to Yandex's "+vowel"
# form here, and drop any orphan accents. Pure and unit-testable.
_ACUTE = "́"
_VOWEL_ACUTE_RE = re.compile(r"([аеёиоуыэюяАЕЁИОУЫЭЮЯ])́")


def yandex_stress_markup(text: str) -> str:
    text = _VOWEL_ACUTE_RE.sub(r"+\1", text)   # "приве́т" -> "прив+ет"
    return text.replace(_ACUTE, "")            # remove any leftover orphan accents


# Sentence-ending punctuation; ellipsis "…" is normalized to "." first because
# espeak-ng does not treat the "…" character as a pause.
def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping terminal punctuation. Ellipsis "…" and
    runs of dots are normalized to a single ".". Returns non-empty, stripped parts."""
    text = text.replace("…", ".")
    text = re.sub(r"\.{2,}", ".", text)              # "..." -> "."
    parts = re.split(r"(?<=[.!?])\s+", text.strip())  # split after . ! ?
    # Keep only fragments with a word character, so punctuation-only pieces
    # (e.g. "." / "?" / "…"->".") that piper can't voice are dropped.
    return [p.strip() for p in parts if p.strip() and re.search(r"\w", p, re.UNICODE)]


class TtsBackend(ABC):
    """Abstract TTS backend: text -> (mime, audio_bytes)."""

    @abstractmethod
    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        ...


class TeraTtsHttpBackend(TtsBackend):
    """TeraTTS HTTP service backend. Returns MP3 (audio/mpeg)."""

    def __init__(self, base_url: str, client: httpx.AsyncClient, timeout: int):
        self.base_url = base_url
        self.client = client
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        url = f"{self.base_url.rstrip('/')}/synthesize/{quote(text, safe='')}"
        resp = await self.client.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)


def wav_to_mp3(wav_bytes: bytes, bit_rate: int = 64, quality: int = 2) -> bytes:
    """Transcode a 16-bit PCM WAV (mono/stereo) to MP3 via lameenc.

    The speaker firmware can't decode WAV, so Piper output is served as MP3.
    """
    import lameenc  # local import: only needed when Piper is used

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sample_rate)
    enc.set_channels(channels)
    enc.set_bit_rate(bit_rate)
    enc.set_quality(quality)
    return bytes(enc.encode(pcm) + enc.flush())


class PiperTtsBackend(TtsBackend):
    """In-process Piper backend (neural VITS via onnxruntime). Returns MP3.

    The voice is loaded once and shared (espeak-ng-data is bundled in the Piper
    package, so no system espeak is needed). Synthesis is blocking, so it runs in
    a worker thread. Piper produces WAV 22050 Hz mono 16-bit, which is then
    transcoded to MP3 (audio/mpeg) because the speaker firmware can't decode WAV.
    """

    def __init__(self, voice_path: str):
        # Imported lazily so the heavy dependency/model are only required when the
        # Piper backend is actually selected at runtime (never in tests/CI).
        from piper import PiperVoice

        # The config json sits next to the onnx at <path>.json.
        self._voice = PiperVoice.load(voice_path, voice_path + ".json")
        self._sentence_silence = max(0.0, float(settings.tts_sentence_silence))
        logger.info(f"Piper TTS voice loaded: {voice_path}")

    def _synth(self, text: str) -> bytes:
        sentences = split_sentences(text)
        pcm = bytearray()
        framerate = channels = sampwidth = None
        for sentence in sentences:
            b = io.BytesIO()
            try:
                with wave.open(b, "wb") as wf:
                    self._voice.synthesize_wav(sentence, wf)
            except Exception:
                # piper produced no audio for this fragment (e.g. symbols only); skip it.
                continue
            with wave.open(io.BytesIO(b.getvalue()), "rb") as rf:
                if framerate is None:
                    framerate, channels, sampwidth = rf.getframerate(), rf.getnchannels(), rf.getsampwidth()
                frames = rf.readframes(rf.getnframes())
            if not frames:
                continue
            if pcm and self._sentence_silence > 0:
                # whole number of frames of silence, so samples stay aligned
                frame = sampwidth * channels
                pcm += b"\x00" * (int(framerate * self._sentence_silence) * frame)
            pcm += frames
        if framerate is None:
            # Nothing pronounceable -> return a short silent clip (don't crash).
            framerate, channels, sampwidth = 22050, 1, 2
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setframerate(framerate)
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.writeframes(bytes(pcm))
        # Transcode here so the blocking lameenc call runs in the worker thread.
        return wav_to_mp3(out.getvalue())

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        mp3_bytes = await asyncio.to_thread(self._synth, text)
        return ("audio/mpeg", mp3_bytes)


class YandexTtsBackend(TtsBackend):
    """Yandex SpeechKit v1 cloud TTS. Requests MP3 directly (audio/mpeg), so no
    transcoding is needed. Auth uses an API key bound to a service account
    (`Authorization: Api-Key <key>`); folderId is only needed for user-account (IAM)
    auth, so it is optional and sent only when configured. Russian stress marks are
    converted to Yandex's "+vowel" notation via yandex_stress_markup()."""

    def __init__(self, client, *, api_key, voice, emotion, speed, folder_id, url, timeout):
        if not api_key:
            raise ValueError("YANDEX_TTS_API_KEY is required when TTS_BACKEND=yandex")
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.emotion = emotion
        self.speed = speed
        self.folder_id = folder_id
        self.url = url
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        data = {
            "text": yandex_stress_markup(text),
            "lang": "ru-RU" if lang == "ru" else lang,
            "voice": self.voice,
            "emotion": self.emotion,
            "speed": str(self.speed),
            "format": "mp3",  # served straight to the speaker; no WAV->MP3 transcode
        }
        if self.folder_id:
            # Only for user-account (IAM) auth; a service-account API key infers the folder.
            data["folderId"] = self.folder_id
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        resp = await self.client.post(self.url, headers=headers, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)


def make_tts_backend(
    name: str, base_url: str, client: httpx.AsyncClient, timeout: int
) -> TtsBackend:
    """Construct a TTS backend by name."""
    if name == "teratts":
        return TeraTtsHttpBackend(base_url, client, timeout)
    if name == "piper":
        return PiperTtsBackend(settings.piper_voice_path)
    if name == "yandex":
        return YandexTtsBackend(
            client,
            api_key=settings.yandex_tts_api_key,
            voice=settings.yandex_tts_voice,
            emotion=settings.yandex_tts_emotion,
            speed=settings.yandex_tts_speed,
            folder_id=settings.yandex_tts_folder_id,
            url=settings.yandex_tts_url,
            timeout=timeout,
        )
    raise ValueError(f"Unknown TTS backend: {name}")
