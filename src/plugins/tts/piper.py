"""Piper offline TTS brick: config schema and in-process backend."""

import asyncio
import io
import wave

from loguru import logger
from pydantic import BaseModel

from src.plugins.base import Deps, Provider, register
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


class PiperConfig(BaseModel):
    voice_path: str = "models/ru_RU-ruslan-medium.onnx"
    sentence_silence: float = 0.4


@register
class PiperProvider(Provider):
    category = "tts"
    id = "piper"
    label = "Piper (offline)"
    ConfigModel = PiperConfig

    def create(self, cfg: PiperConfig, deps: Deps):
        return PiperTtsBackend(cfg.voice_path, sentence_silence=cfg.sentence_silence)
