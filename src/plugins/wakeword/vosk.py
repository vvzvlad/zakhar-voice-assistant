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
from pydantic import BaseModel, Field, field_validator

from src.logging_setup import capture_native_stderr
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
            # The model load is where Kaldi can emit native WARN/ERR straight to
            # fd 2; capture that window (minimal — only the load) into loguru.
            with capture_native_stderr("vosk-wakeword"):
                model = Model(model_path)  # fail fast if the dir is missing
        self._model = model
        # Normalize keywords (lowercase + strip) for BOTH the grammar JSON and the
        # accept-set. The Russian Vosk model emits lowercase text, so a capitalized
        # operator-configured keyword would never match the decoded tokens otherwise.
        # Defensive: the config field_validator already normalizes; this also covers
        # direct instantiation (e.g. in tests) that bypasses the config layer.
        self._keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
        self._window_ms = window_ms
        # Read by the pipeline to size the pre-roll head it hands to verify()
        # (the config field flows in via the provider's create()).
        self.window_ms = window_ms
        # Read by the pipeline to apply the timeout + fail-open/closed policy.
        self.timeout_ms = timeout_ms
        self.fail_open = (on_error == "open")
        # One-time grammar probe (production path only): build a single throwaway
        # grammar recognizer now, inside the fd-2 capture window, so Kaldi's
        # "Ignoring word missing in vocabulary" WARN for a misconfigured keyword is
        # surfaced to loguru ONCE here at creation/config-reload instead of being
        # emitted (and swallowed) per decode. Guarded so it never runs for an empty
        # keyword set, and wrapped defensively: a probe failure (e.g. a missing
        # native dependency when a sentinel model is injected) must NEVER break
        # verifier construction, so it is logged at debug and ignored.
        if self._keywords:
            try:
                with capture_native_stderr("vosk-wakeword"):
                    self._make_recognizer()  # discard: built only to trip the WARN
            except Exception as e:
                logger.debug(f"Vosk wakeword grammar probe skipped: {e}")

    def _make_recognizer(self):
        """Build a grammar-restricted KaldiRecognizer for the shared model.

        Constraining the recognizer to the keyword grammar (plus "[unk]") is what
        keeps decode fast. Overridable in tests."""
        from vosk import KaldiRecognizer

        grammar = json.dumps(self._keywords + ["[unk]"], ensure_ascii=False)
        # NO fd-2 capture here: this runs per verify in a worker thread, and
        # redirecting process-global fd 2 around every decode would swallow loguru
        # lines emitted by other threads (an observability regression). The
        # "Ignoring word missing in vocabulary" WARN that fires at grammar parse is
        # surfaced to loguru ONCE by the creation-time probe in __init__ instead.
        # A per-decode recognizer built here may still emit that raw native warning
        # to fd 2 for a misconfigured keyword until the operator fixes it, but the
        # actionable warning is already routed to loguru at construction.
        return KaldiRecognizer(self._model, 16000, grammar)

    def _decode(self, pcm: bytes) -> WakewordVerdict:
        rec = self._make_recognizer()
        rec.SetWords(True)
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.FinalResult())
        tokens = result.get("text", "").split()
        # Match each keyword as a CONTIGUOUS run of words in the recognized token
        # list. A single-word keyword reduces to the old "word in tokens" check; a
        # multi-word phrase (e.g. "окей захар") matches only when all of its words
        # appear contiguously — the whole phrase string is never a token by itself.
        # Collect the per-word "conf" of every matched run so the score reflects the
        # confidence of the actual hit; `result["result"]` is aligned 1:1 with
        # `text.split()`, so it is only usable when the lengths agree.
        words = result.get("result", [])
        aligned = words if len(words) == len(tokens) else None
        accepted = False
        confs = []
        for kw in self._keywords:
            kw_words = kw.split()
            if not kw_words:
                continue
            n = len(kw_words)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == kw_words:
                    accepted = True
                    if aligned is not None:
                        confs += [aligned[j]["conf"] for j in range(i, i + n)
                                  if "conf" in aligned[j]]
        if confs:
            score = max(confs)
        else:
            score = 1.0 if accepted else 0.0
        return WakewordVerdict(accepted=accepted, score=score)

    async def verify(self, pcm: bytes) -> WakewordVerdict:
        if not self._keywords:
            # No keywords configured: an unconfigured gate must not silently reject
            # every wake (which would make the assistant deaf). Accept without even
            # building the empty-grammar recognizer.
            return WakewordVerdict(True, None)
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
        description="Wake words: the grammar vocabulary AND the accept set. One "
        "entry per wake word; the gate accepts when ANY entry is recognised (OR). "
        "A multi-word entry is treated as a phrase — all of its words must be "
        "recognised contiguously for it to match. Leave empty to accept every wake "
        "(the gate is disabled).",
    )

    @field_validator("keywords")
    @classmethod
    def _normalize_keywords(cls, v: list[str]) -> list[str]:
        # Force keywords to stripped lowercase at the CONFIG layer so the stored
        # value and the panel display are already normalized (not just the runtime
        # grammar). The Russian Vosk model emits lowercase, and matching is
        # case-insensitive, so a capitalized entry would never match otherwise.
        # Empties (left after stripping) are dropped.
        return [kw.strip().lower() for kw in v if kw.strip()]

    window_ms: int = Field(
        2500,
        title="Window (ms)",
        description="Length of the pre-roll audio window decoded for verification. "
        "MUST be >= the device firmware's preroll_duration (currently 2000 ms): the "
        "wake word sits at the TAIL of the pre-roll, so a window shorter than the "
        "pre-roll decodes only the START of the keyword (e.g. 'заха') and the gate "
        "rejects every wake. Kept with a margin above preroll_duration so the whole "
        "keyword is always inside the decoded window.",
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
