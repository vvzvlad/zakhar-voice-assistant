"""Text-to-speech backends (pluggable; TeraTTS HTTP is the default)."""

import asyncio
import base64
import io
import json
import re
import wave
from abc import ABC, abstractmethod
from urllib.parse import quote

import httpx
from loguru import logger


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

    def __init__(self, voice_path: str | None = None, *, sentence_silence: float = 0.4, voice=None):
        # The loaded voice is injectable so _synth can be exercised in CI without
        # the heavy model load. Production still passes voice_path and gets the
        # real PiperVoice.load(...) path; voice is only used by tests/from_voice.
        if voice is None:
            # Imported lazily so the heavy dependency/model are only required when
            # the Piper backend is actually selected at runtime (never in tests/CI).
            from piper import PiperVoice

            # The config json sits next to the onnx at <path>.json.
            voice = PiperVoice.load(voice_path, voice_path + ".json")
            logger.info(f"Piper TTS voice loaded: {voice_path}")
        self._voice = voice
        self._sentence_silence = max(0.0, float(sentence_silence))

    @classmethod
    def from_voice(cls, voice, *, sentence_silence: float = 0.4) -> "PiperTtsBackend":
        """Build a backend around an already-loaded voice (skips PiperVoice.load)."""
        return cls(voice=voice, sentence_silence=sentence_silence)

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


def _decode_v3_audio(body: str) -> bytes:
    """Reassemble audio from a SpeechKit v3 utteranceSynthesis response.

    The REST response is a stream of JSON objects (newline-delimited or
    concatenated), each shaped like {"result": {"audioChunk": {"data": "<base64>"}}}.
    Decode and concatenate every audio chunk; an {"error": ...} object raises.
    Tolerant to a single object, NDJSON, or a JSON array of objects.
    """
    chunks = bytearray()
    decoder = json.JSONDecoder()
    idx, length = 0, len(body)
    while idx < length:
        while idx < length and body[idx] in " \r\n\t":
            idx += 1
        if idx >= length:
            break
        message, idx = decoder.raw_decode(body, idx)
        for obj in (message if isinstance(message, list) else [message]):
            if not isinstance(obj, dict):
                continue
            if "error" in obj:
                raise RuntimeError(f"Yandex TTS v3 error: {obj['error']}")
            data = (obj.get("result") or {}).get("audioChunk", {}).get("data")
            if data:
                chunks.extend(base64.b64decode(data))
    return bytes(chunks)


class YandexTtsBackend(TtsBackend):
    """Yandex SpeechKit v3 cloud TTS (utteranceSynthesis). The v3 REST endpoint is
    server-streaming: the response is a stream of JSON objects, each carrying a
    base64-encoded MP3 chunk; the chunks are decoded and concatenated into a valid
    MP3 (audio/mpeg), so no transcoding is needed. Auth uses an API key bound to a
    service account (`Authorization: Api-Key <key>`); the `x-folder-id` header is
    only needed for user-account (IAM) auth, so it is sent only when folder_id is
    configured. Russian stress marks are converted to Yandex's "+vowel" notation via
    yandex_stress_markup()."""

    def __init__(self, client, *, api_key, voice, role, speed, folder_id, url, timeout):
        if not api_key:
            raise ValueError("YANDEX_TTS_API_KEY is required when TTS_BACKEND=yandex")
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.role = role
        self.speed = speed
        self.folder_id = folder_id
        self.url = url
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        # v3 carries voice/role/speed as "hints"; the role hint is sent only when a
        # role is configured (voices without an amplua reject an empty role).
        hints = [{"voice": self.voice}, {"speed": self.speed}]
        if self.role:
            hints.insert(1, {"role": self.role})
        payload = {
            "text": yandex_stress_markup(text),
            "hints": hints,
            "outputAudioSpec": {"containerAudio": {"containerAudioType": "MP3"}},
            "loudnessNormalizationType": "LUFS",
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        if self.folder_id:
            # Only for user-account (IAM) auth; a service-account API key infers the folder.
            headers["x-folder-id"] = self.folder_id
        resp = await self.client.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return ("audio/mpeg", _decode_v3_audio(resp.text))
