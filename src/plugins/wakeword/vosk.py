"""Vosk grammar-restricted wakeword verifier: config schema and in-process backend.

Second-stage wake-word gate. The pre-roll PCM captured around the on-device wake
trigger is decoded by a Vosk recognizer constrained to a tiny grammar (the wake
keywords plus "[unk]"). Grammar mode keeps decode at ~10ms for a 1.5s clip (vs
~112ms full-vocab), so it is required. The verdict gates the rest of the pipeline.
"""

import asyncio
import json
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.wakeword import PassthroughVerifier, WakewordVerdict, WakewordVerifier


class VoskWakewordVerifier(WakewordVerifier):
    """In-process Vosk verifier built on a grammar-restricted KaldiRecognizer.

    The model is loaded once and shared; a grammar-restricted recognizer is
    spawned per call. Decoding is blocking C code, so it runs in a worker thread.
    verify() NEVER raises: on any internal error it returns the fail-open/closed
    verdict so the pipeline's gate stays well-defined.
    """

    def __init__(self, *, model_path, keywords, window_ms, timeout_ms, on_error, model=None):
        # The model is injectable for testing. When None (production path), it is
        # loaded lazily so the heavy dependency/model are only required when the
        # Vosk wakeword backend is actually selected at runtime (never in tests/CI).
        if model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(model_path)  # fail fast if the dir is missing
        self._model = model
        # Normalize keywords (lowercase + strip) for BOTH the grammar JSON and the
        # accept-set. The Russian Vosk model emits lowercase text, so a capitalized
        # operator-configured keyword would never match the decoded tokens otherwise.
        self._keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
        self._window_ms = window_ms
        # Read by the pipeline to size the pre-roll head it hands to verify()
        # (the config field flows in via the provider's create()).
        self.window_ms = window_ms
        # Read by the pipeline to apply the timeout + fail-open/closed policy.
        self.timeout_ms = timeout_ms
        self.fail_open = (on_error == "open")

    def _make_recognizer(self):
        """Build a grammar-restricted KaldiRecognizer for the shared model.

        Constraining the recognizer to the keyword grammar (plus "[unk]") is what
        keeps decode fast. Overridable in tests."""
        from vosk import KaldiRecognizer

        grammar = json.dumps(self._keywords + ["[unk]"], ensure_ascii=False)
        return KaldiRecognizer(self._model, 16000, grammar)

    def _decode(self, pcm: bytes) -> WakewordVerdict:
        rec = self._make_recognizer()
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.FinalResult())
        tokens = result.get("text", "").split()
        keyset = set(self._keywords)
        accepted = any(tok in keyset for tok in tokens)
        # Prefer the per-word confidence of a matched keyword when present.
        confs = [w["conf"] for w in result.get("result", [])
                 if w.get("word") in keyset and "conf" in w]
        if confs:
            score = max(confs)
        else:
            score = 1.0 if accepted else 0.0
        return WakewordVerdict(accepted=accepted, score=score)

    async def verify(self, pcm: bytes) -> WakewordVerdict:
        if not pcm:
            # No audio -> apply the configured fail policy (open accepts, closed rejects).
            return WakewordVerdict(self.fail_open, None)
        # Pre-roll PCM is already 16 kHz mono 16-bit — no resampling needed.
        try:
            return await asyncio.to_thread(self._decode, pcm)
        except Exception as e:
            # NEVER raise out of verify: a decode failure must not crash the gate.
            # Fall back to the configured fail policy so the pipeline stays defined.
            logger.warning(f"Vosk wakeword decode failed: {e}")
            return WakewordVerdict(self.fail_open, None)


class VoskWakewordConfig(BaseModel):
    enabled: bool = Field(
        True,
        description="Verify the on-device wake trigger with a grammar-restricted "
        "Vosk decode before running the pipeline. A reject suppresses STT/LLM/TTS. "
        "Loads a model on first enable.",
    )
    model_path: str = Field(
        "models/vosk-model-small-ru-0.22",
        title="Model path",
        description="Path to the Vosk model directory used for the grammar decode.",
    )
    keywords: list[str] = Field(
        default_factory=lambda: ["захар"],
        title="Keywords",
        description="Wake words: the grammar vocabulary AND the accept set. The "
        "decode is constrained to these (plus an unknown token); a verdict accepts "
        "when any keyword is recognised.",
    )
    window_ms: int = Field(
        1500,
        title="Window (ms)",
        description="Length of the pre-roll audio window decoded for verification.",
    )
    timeout_ms: int = Field(
        300,
        title="Timeout (ms)",
        description="Per-verification time budget the pipeline applies before "
        "falling back to the on-error policy.",
    )
    on_error: Literal["open", "closed"] = Field(
        "open",
        title="On error",
        description="Verdict on timeout/decode error: 'open' accepts (fail-open, "
        "never blocks a real wake), 'closed' rejects (fail-closed, stricter).",
    )


@register
class VoskWakewordProvider(Provider):
    category = "wakeword"
    id = "vosk"
    label = "Vosk (keyword)"
    ConfigModel = VoskWakewordConfig

    def create(self, cfg: VoskWakewordConfig, deps: Deps):
        if not cfg.enabled:
            # Disabled stage: no model load, the gate always accepts.
            return PassthroughVerifier()
        return VoskWakewordVerifier(
            model_path=cfg.model_path,
            keywords=cfg.keywords,
            window_ms=cfg.window_ms,
            timeout_ms=cfg.timeout_ms,
            on_error=cfg.on_error,
        )

    def describe(self, cfg: VoskWakewordConfig) -> str:
        import os

        return f"{self.id}/{os.path.basename(cfg.model_path)}" if cfg.enabled else f"{self.id}/off"
