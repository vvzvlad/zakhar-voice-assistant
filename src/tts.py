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

# The canonical LLM->TTS text is the model's own notation: plain text with "+"
# before the stressed vowel (e.g. "прив+ет"). Each backend below adapts that
# canon to its engine via the shared opt-in helpers.
from src.plugins.tts._ru_text import (
    expand_units,
    phonetic_ru,
    sanitize_plus_stress,
    stress_to_acute,
)


# Yandex SpeechKit v3 utteranceSynthesis rejects requests whose text exceeds 250
# characters (and ~24 s of audio, but 250 chars is the binding limit). Long replies
# must be split into <=250-char parts, synthesized separately, and concatenated.
# Source: yandex.cloud/docs/speechkit limits, API v3.
YANDEX_V3_TEXT_LIMIT = 250


def _split_oversized(fragment: str, limit: int) -> list[str]:
    """Split a single over-limit fragment into <=limit pieces on word boundaries.
    A single word longer than the limit is hard-sliced (rare; may break a
    Yandex "+vowel" stress pair, acceptable for such pathological input)."""
    out: list[str] = []
    cur = ""
    for word in fragment.split():
        if len(word) > limit:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(word), limit):
                out.append(word[i:i + limit])
            continue
        candidate = f"{cur} {word}" if cur else word
        if len(candidate) <= limit:
            cur = candidate
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def _chunk_for_v3(text: str, limit: int = YANDEX_V3_TEXT_LIMIT) -> list[str]:
    """Split already-stress-marked text into request chunks, each <=limit chars.
    Packs whole sentences greedily; an over-limit sentence is split on words.
    Returns [] for empty / punctuation-only input."""
    # split_sentences already drops fragments with no word char; inputs with zero
    # word characters (pure punctuation/emoji) are unvoiceable and rejected by
    # Yandex with 400, so returning [] for them (empty audio, no request) is correct.
    sentences = split_sentences(text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_oversized(s, limit))
            continue
        candidate = f"{cur} {s}" if cur else s
        if len(candidate) <= limit:
            cur = candidate
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


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
        # Same adaptation chain as Piper: TeraTTS historically received
        # Piper-style processed text, so this preserves its behavior.
        text = phonetic_ru(expand_units(stress_to_acute(text)))
        url = f"{self.base_url.rstrip('/')}/synthesize/{quote(text, safe='')}"
        resp = await self.client.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "audio/mpeg"), resp.content)


def make_ack_chime_mp3(
    *,
    sample_rate: int = 22050,
    tones_hz: tuple[float, float] = (880.0, 1320.0),
    tone_ms: int = 130,
    gap_ms: int = 20,
    amplitude: float = 0.5,
) -> bytes:
    """Synthesize a short two-tone confirmation chime ("блям") and return it as MP3.

    A pleasant ~300 ms rising two-tone beep with a per-tone attack/decay envelope (so
    there are no clicks), built with numpy and transcoded to MP3 via wav_to_mp3 so the
    speaker firmware can decode it. Pure/deterministic — the pipeline builds it ONCE and
    caches the bytes (see Pipeline._ack_clip / Pipeline._ack_clip_bytes). numpy is
    already a dependency.
    """
    import numpy as np

    def _tone(freq: float, ms: int) -> "np.ndarray":
        n = int(sample_rate * ms / 1000)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        t = np.arange(n, dtype=np.float32) / sample_rate
        wave_arr = np.sin(2 * np.pi * freq * t)
        # Short raised-cosine attack/release so the tone fades in and out (no clicks).
        # Clamp edge to n // 2 so the attack (env[:edge]) and release (env[-edge:])
        # ramps never overlap for a very short tone (when 2*edge > n); the defaults
        # (130 ms tone) are far above this bound, so they are unchanged.
        edge = max(1, min(int(sample_rate * 0.008), n // 2))  # ~8 ms ramps
        env = np.ones(n, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        env[:edge] = ramp
        env[-edge:] = ramp[::-1]
        return wave_arr * env

    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    signal = np.concatenate([_tone(tones_hz[0], tone_ms), gap, _tone(tones_hz[1], tone_ms)])
    pcm16 = np.clip(signal * amplitude, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")  # little-endian 16-bit PCM
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setframerate(sample_rate)
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.writeframes(pcm16.tobytes())
    return wav_to_mp3(buf.getvalue())


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
        # Adapt the canonical "+vowel" text to the engine: combining acute for
        # stress (espeak-ng notation), spelled-out units, phonetic hacks.
        # Stress conversion must run before phonetic_ru (see _ru_text).
        text = phonetic_ru(expand_units(stress_to_acute(text)))
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
    service account (`Authorization: Api-Key <key>`). The input text already
    arrives in Yandex's native "+vowel" stress notation (the canonical LLM->TTS
    contract), so no stress conversion is needed — only unit expansion and
    dropping stray '+' signs."""

    def __init__(self, client, *, api_key, voice, role, speed, url, timeout):
        if not api_key:
            raise ValueError("YANDEX_TTS_API_KEY is required when TTS_BACKEND=yandex")
        self.client = client
        self.api_key = api_key
        self.voice = voice
        self.role = role
        self.speed = speed
        self.url = url
        self.timeout = timeout

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        # v3 caps each request at YANDEX_V3_TEXT_LIMIT chars; adapt the canonical
        # "+vowel" text once (expand units, drop stray '+'), then split into
        # bounded chunks, synthesize each, and concatenate the MP3 audio.
        marked = sanitize_plus_stress(expand_units(text))
        chunks = _chunk_for_v3(marked, YANDEX_V3_TEXT_LIMIT)
        # Nothing pronounceable (empty / punctuation-only) -> serve no audio, don't POST.
        audio = bytearray()
        for chunk in chunks:
            audio.extend(await self._synthesize_chunk(chunk))
        return ("audio/mpeg", bytes(audio))

    async def _synthesize_chunk(self, text: str) -> bytes:
        # `text` is already adapted (units expanded, stray '+' dropped) and within
        # the length limit; it carries Yandex-native "+vowel" stress markup as-is.
        # v3 carries voice/role/speed as "hints"; the role hint is sent only when a
        # role is configured (voices without an amplua reject an empty role).
        hints = [{"voice": self.voice}, {"speed": self.speed}]
        if self.role:
            hints.insert(1, {"role": self.role})
        payload = {
            "text": text,
            "hints": hints,
            "outputAudioSpec": {"containerAudio": {"containerAudioType": "MP3"}},
            "loudnessNormalizationType": "LUFS",
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        resp = await self.client.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface Yandex's diagnostic body (it names the real cause, e.g. text too
            # long / bad voice / bad role); raise_for_status() alone hides it. Same
            # philosophy as src/llm.py logging status + body.
            raise RuntimeError(
                f"Yandex TTS v3 {resp.status_code}: {resp.text[:500]}"
            ) from e
        return _decode_v3_audio(resp.text)
