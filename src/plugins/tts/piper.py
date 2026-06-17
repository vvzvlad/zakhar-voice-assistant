"""Piper offline TTS brick: config schema and in-process backend."""

import asyncio
import io
import os
import wave

from loguru import logger
from pydantic import BaseModel, Field

from src.audio_codec import make_mp3_stream_encoder
from src.plugins.base import LOCAL_MODEL_FIELD_EXTRA, Deps, Provider, register
# The canonical LLM->TTS text is the model's own notation: plain text with "+"
# before the stressed vowel (e.g. "прив+ет"). The backend adapts that canon to
# its engine via the shared opt-in helpers.
from src.plugins.tts._ru_text import expand_units, phonetic_ru, stress_to_acute
from src.tts import TtsBackend, split_sentences


class PiperTtsBackend(TtsBackend):
    """In-process Piper backend (neural VITS via onnxruntime). Returns WAV.

    The voice is loaded once and shared (espeak-ng-data is bundled in the Piper
    package, so no system espeak is needed). Synthesis is blocking, so it runs in
    a worker thread. Returns the engine's NATIVE format — WAV 22050 Hz mono
    16-bit (audio/wav); adapting it to what the consumer can play is the
    delivery boundary's job (see audio_codec.to_playable), not synthesis'.
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
        return out.getvalue()

    async def synthesize(self, text: str, lang: str = "ru") -> tuple[str, bytes]:
        wav_bytes = await asyncio.to_thread(self._synth, text)
        return ("audio/wav", wav_bytes)

    def _synth_sentence_pcm(self, sentence: str) -> tuple[bytes, int, int, int]:
        """Synthesize ONE already-adapted sentence to raw PCM via Piper's streaming
        generator. Returns (pcm, sample_rate, channels, sample_width); an
        unpronounceable fragment (Piper raises, e.g. symbols only) yields (b"",0,0,0)
        so the caller skips it — same tolerance as the buffered _synth path. Blocking
        (onnx); call via asyncio.to_thread."""
        pcm = bytearray()
        sr = ch = sw = 0
        try:
            for chunk in self._voice.synthesize(sentence):
                if sr == 0:
                    sr, ch, sw = chunk.sample_rate, chunk.sample_channels, chunk.sample_width
                pcm += chunk.audio_int16_bytes
        except Exception:
            # Piper produced no audio for this fragment (e.g. symbols only); skip it.
            return (b"", 0, 0, 0)
        return (bytes(pcm), sr, ch, sw)

    async def synthesize_stream(self, text: str, lang: str = "ru"):
        """Native streaming synthesis: synthesize sentence-by-sentence and emit
        incremental MP3 (audio/mpeg) so the delivery boundary streams it live (WAV
        would be buffered+transcoded whole — zero latency win). The device already
        always gets MP3 for Piper (to_playable transcodes the buffered WAV), so this
        produces the SAME final format, just earlier and per-sentence. The buffered
        synthesize() stays native WAV. Inter-sentence silence matches _synth."""
        # Same canonical "+vowel" -> engine adaptation as the buffered _synth.
        text = phonetic_ru(expand_units(stress_to_acute(text)))
        sentences = split_sentences(text)

        async def _gen():
            enc = None
            silence = b""
            emitted = False
            for sentence in sentences:
                pcm, sr, ch, sw = await asyncio.to_thread(self._synth_sentence_pcm, sentence)
                if not pcm:
                    continue
                if enc is None:
                    # First real audio fixes the stream format and the silence block.
                    enc = make_mp3_stream_encoder(sr, ch, sw)
                    if self._sentence_silence > 0:
                        frame = sw * ch  # bytes per audio frame, so silence stays aligned
                        silence = b"\x00" * (int(sr * self._sentence_silence) * frame)
                # Prepend the inter-sentence gap only BETWEEN emitted sentences.
                prefix = silence if (emitted and self._sentence_silence > 0) else b""
                mp3 = await asyncio.to_thread(enc.encode, prefix + pcm)
                emitted = True
                if mp3:
                    yield mp3
            if enc is not None:
                tail = await asyncio.to_thread(enc.flush)
                if tail:
                    yield tail

        return ("audio/mpeg", _gen())


class PiperConfig(BaseModel):
    voice_path: str = Field("models/ru_RU-ruslan-medium.onnx", json_schema_extra=LOCAL_MODEL_FIELD_EXTRA)
    sentence_silence: float = 0.4


def _list_piper_voices(base_dir: str) -> list[dict]:
    """Scan base_dir for Piper voices: *.onnx files that have a sibling
    <name>.onnx.json config (the pair create() loads). Returns
    [{"value": <onnx path>, "label": <name without .onnx>}, ...] sorted by
    label (case-insensitive). Any filesystem error yields an empty list."""
    try:
        names = os.listdir(base_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith(".") or not name.endswith(".onnx"):
            continue
        onnx_path = os.path.join(base_dir, name)
        if not os.path.isfile(onnx_path) or not os.path.isfile(onnx_path + ".json"):
            continue
        out.append({"value": onnx_path, "label": name[: -len(".onnx")]})
    out.sort(key=lambda o: o["label"].lower())
    return out


@register
class PiperProvider(Provider):
    category = "tts"
    id = "piper"
    label = "Piper (offline)"
    ConfigModel = PiperConfig

    def create(self, cfg: PiperConfig, deps: Deps):
        return PiperTtsBackend(cfg.voice_path, sentence_silence=cfg.sentence_silence)

    def options(self, field: str, cfg: PiperConfig, deps: Deps, query: str = ""):
        # Local-disk scan of installed Piper voices next to the configured path
        # (defaults to models/). Synchronous: return a plain list. rstrip the
        # trailing slash for symmetry with the Vosk scan (no-op for a real voice
        # file path, which never ends in a slash).
        if field == "voice_path":
            return _list_piper_voices(os.path.dirname(cfg.voice_path.rstrip("/")) or "models")
        return None
