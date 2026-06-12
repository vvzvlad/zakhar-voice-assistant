"""RuAccent stage brick: config schema and in-process accent backend.

Places Russian stress marks ("+" before the stressed vowel) on the LLM reply
before TTS, using the RuAccent library. The output IS the canonical "+vowel"
LLM->TTS contract, so every TTS backend already adapts it unchanged.
"""

import asyncio
import os
import warnings

from loguru import logger
from typing import Literal
from pydantic import BaseModel, Field

from src import config_store
from src.plugins.base import Deps, Provider, register
from src.accent import Accentizer, PassthroughAccentizer
from src.plugins.tts._ru_text import drop_plus_stress, stress_to_acute, stress_to_uppercase

# RuAccent omograph model sizes, ordered roughly by RAM and grouped by family
# (tiny* smallest, then the turbo* family, "big_poetry" largest/last). The exact
# per-model RAM is in RUACCENT_MODEL_RAM below and shown in the dropdown options.
RUACCENT_MODELS = ["tiny", "tiny2", "tiny2.1", "turbo2", "turbo3", "turbo3.1", "turbo", "big_poetry"]

# Approximate per-model RAM footprint. The omograph model.onnx dominates the
# per-model memory delta; byte sizes are from the ruaccent/accentuator HF repo.
RUACCENT_MODEL_RAM = {
    "tiny": "~10 MB",
    "tiny2": "~10 MB",
    "tiny2.1": "~40 MB",
    "turbo2": "~360 MB",
    "turbo3": "~360 MB",
    "turbo3.1": "~360 MB",
    "turbo": "~330 MB",
    "big_poetry": "~700 MB",
}
# Dropdown display labels: "<model id> (<approx RAM>)", keyed by value so the
# STORED value stays the bare model id. Surfaced to the panel via json_schema_extra.
RUACCENT_MODEL_LABELS = {m: f"{m} ({RUACCENT_MODEL_RAM.get(m, '?')})" for m in RUACCENT_MODELS}

# How the stage emits stress marks. "plus" keeps RuAccent's native "+vowel"
# output (the canonical LLM->TTS contract every TTS backend already adapts);
# the others convert it for engines that strip "+" (e.g. Fish Audio).
STRESS_FORMATTERS = {
    "plus": lambda t: t,                # native "+vowel" (no change)
    "acute": stress_to_acute,           # vowel + combining acute (U+0301)
    "uppercase": stress_to_uppercase,   # capitalise the stressed vowel
}


class RuAccentBackend(Accentizer):
    """In-process RuAccent backend (ONNX inference via onnxruntime).

    The model is loaded once and shared. Inference is blocking, so it runs in a
    worker thread. The loaded accentizer is injectable so the backend can be
    exercised in tests/CI without the real model load.
    """

    def __init__(self, *, model_size, use_dictionary, tiny_mode, device, workdir, accentizer=None, stress_format: str = "plus"):
        # Selected output notation, applied to RuAccent's "+vowel" output for both
        # the real and injected-accentizer paths.
        self._format = STRESS_FORMATTERS.get(stress_format, STRESS_FORMATTERS["plus"])
        if accentizer is None:
            # Imported lazily so the heavy dependency/models are only required when
            # the RuAccent stage is actually enabled at runtime (never in tests/CI).
            # RuAccent runs its models on onnxruntime, not torch, so transformers'
            # import-time "PyTorch was not found" notice is cosmetic noise. Quiet it
            # (setdefault so an operator can still override via the env).
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from ruaccent import RUAccent

            accentizer = RUAccent()
            # ruaccent passes the deprecated local_dir_use_symlinks arg to
            # hf_hub_download; the resulting huggingface_hub UserWarning is upstream
            # noise we can't fix. Scope the suppression to just that message so other
            # warnings still surface.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
                accentizer.load(
                    omograph_model_size=model_size,
                    use_dictionary=use_dictionary,
                    tiny_mode=tiny_mode,
                    device=device,
                    workdir=workdir,
                )
            logger.info(
                f"RuAccent model loaded: {model_size} "
                f"(dict={use_dictionary}, tiny={tiny_mode}, device={device})"
            )
        self._accentizer = accentizer

    def _process(self, text: str) -> str:
        # Strip any existing "+vowel" marks first so RuAccent is the single source of
        # stress and we never produce double marks like "при+в+ет".
        text = drop_plus_stress(text)              # RuAccent is the single source of stress
        text = self._accentizer.process_all(text)  # -> "+vowel"
        return self._format(text)                  # apply the selected output notation

    async def accentize(self, text: str) -> str:
        if not text.strip():
            return text
        # Inference blocks -> run it in a worker thread.
        return await asyncio.to_thread(self._process, text)


class RuAccentConfig(BaseModel):
    enabled: bool = Field(
        True,
        description="Place Russian stress marks on the assistant's reply before TTS "
        "so words are pronounced correctly. Loads a model on first enable.",
    )
    model_size: str = Field(
        "turbo3.1",
        title="Model",
        # `enum` (not a bare `options` array) is what the panel's SchemaForm reads to
        # render a dropdown; the static `options` list it was using has no widget path.
        json_schema_extra={"widget": "select", "enum": RUACCENT_MODELS, "enumLabels": RUACCENT_MODEL_LABELS},
        description="RuAccent omograph model. Larger = better homograph handling, more RAM "
        "(approx per-model RAM shown in each option).",
    )
    stress_format: Literal["plus", "acute", "uppercase"] = Field(
        "plus",
        title="Stress output",
        description="How stress is emitted. 'plus' keeps the native '+vowel' "
        "notation (default; converted per TTS engine). 'acute' uses a combining "
        "acute accent, 'uppercase' capitalises the stressed vowel — use these for "
        "engines that strip '+' (e.g. Fish Audio).",
    )
    use_dictionary: bool = Field(
        True,
        description="Load the full stress dictionary (more RAM, better accuracy). "
        "Ignored in tiny mode.",
    )
    tiny_mode: bool = Field(
        False,
        description="Minimal mode (~512MB RAM): disables the rule pipeline and the "
        "dictionary. Lower quality on homographs.",
    )
    device: Literal["CPU", "CUDA"] = Field(
        "CPU",
        description="Inference device. CUDA requires onnxruntime-gpu + CUDA.",
    )


@register
class RuAccentProvider(Provider):
    category = "ruaccent"
    id = "ruaccent"
    label = "RuAccent"
    ConfigModel = RuAccentConfig

    def create(self, cfg: RuAccentConfig, deps: Deps):
        if not cfg.enabled:
            # Disabled stage: no model load, the reply passes through unchanged.
            return PassthroughAccentizer()
        # Models download into this workdir from HuggingFace on first load().
        workdir = os.path.join(config_store.DATA_DIR, "ruaccent")
        os.makedirs(workdir, exist_ok=True)
        return RuAccentBackend(
            model_size=cfg.model_size,
            use_dictionary=cfg.use_dictionary,
            tiny_mode=cfg.tiny_mode,
            device=cfg.device,
            workdir=workdir,
            stress_format=cfg.stress_format,
        )

    def describe(self, cfg: RuAccentConfig) -> str:
        return f"{self.id}/{cfg.model_size}" if cfg.enabled else f"{self.id}/off"
