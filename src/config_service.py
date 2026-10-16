"""Plugin-agnostic config core: catalog / get / create / apply / options.

ConfigService is the only thing the panel/API talks to. It knows about categories,
provider ids, their ConfigModel schema and the runtime object create() returns — but
never about any concrete provider field. Stage providers come from the REGISTRY; core
sections come from CoreConfig. Both expose themselves to the UI as JSON Schema.

Store path injection: the constructor takes an optional `path` (defaults to
config_store.DEFAULT_PATH); apply() persists there, and tests point it at a tmp file.
"""

import copy

from pydantic import BaseModel

from src import config_store
from src.core_config import CoreConfig
from src.plugins.base import Deps, get_provider, providers

STAGE_CATEGORIES = ("vad", "wakeword", "stt", "llm", "stress", "tts")
# Categories whose providers are chosen per recognition chain (core.chains), not by
# one `selected` id: the slot is a pool of provider instances.
POOL_CATEGORIES = ("stt", "llm")
# tts holds an ordered list of fallback variants instead of one `selected` id.
SLOT_CATEGORIES = tuple(c for c in STAGE_CATEGORIES if c not in POOL_CATEGORIES + ("tts",))


class StageSlot(BaseModel):
    selected: str
    instances: dict[str, dict] = {}


class StagePool(BaseModel):
    instances: dict[str, dict] = {}


class TtsVariant(BaseModel):
    provider: str     # tts provider id (its settings live in tts.instances.<id>)
    enabled: bool = True


class TtsVariants(BaseModel):
    """The TTS stage: enabled variants are tried in list order, the next one taking
    over when a provider fails before the speaker has received any audio."""
    variants: list[TtsVariant]
    instances: dict[str, dict] = {}


class ConfigDoc(BaseModel):
    version: int = 1
    # vad has a default slot (unlike stt/llm/tts) so configs written before the VAD
    # stage existed still parse; the webrtc provider's own defaults then apply.
    vad: StageSlot = StageSlot(selected="webrtc")
    # wakeword has a default slot (like vad/stress) so configs written before the
    # wakeword stage existed still parse; the vosk provider's own defaults apply.
    wakeword: StageSlot = StageSlot(selected="vosk")
    stt: StagePool
    llm: StagePool
    # stress has a default slot (like vad) so configs written before the stress
    # stage existed still parse; the ruaccent provider's own defaults then apply.
    stress: StageSlot = StageSlot(selected="ruaccent")
    tts: TtsVariants
    core: CoreConfig = CoreConfig()


def _validate(doc: ConfigDoc) -> None:
    """Checks a parsed document must pass beyond its pydantic structure: every
    selected slot instance validates; tts.variants is usable — at least one enabled
    variant, no provider listed twice, every variant's provider registered and its
    instance config valid; and core.chains is usable — at least one enabled chain,
    unique non-empty names, registered provider ids, and every referenced instance
    config validates. Raises ValueError/ValidationError."""
    for cat in SLOT_CATEGORIES:
        slot = getattr(doc, cat)
        get_provider(cat, slot.selected).ConfigModel(**slot.instances.get(slot.selected, {}))
    variants = doc.tts.variants
    if not any(v.enabled for v in variants):
        raise ValueError("tts.variants: at least one enabled variant is required")
    pids = [v.provider for v in variants]
    dupes = sorted({p for p in pids if pids.count(p) > 1})
    if dupes:
        raise ValueError(f"tts.variants: duplicate providers: {dupes}")
    for pid in pids:
        get_provider("tts", pid).ConfigModel(**doc.tts.instances.get(pid, {}))
    chains = doc.core.chains
    if not any(c.enabled for c in chains):
        raise ValueError("core.chains: at least one enabled chain is required")
    names = [c.name for c in chains]
    if any(not n.strip() for n in names):
        raise ValueError("core.chains: every chain needs a name")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"core.chains: duplicate chain names: {dupes}")
    for chain in chains:
        for cat, pid in (("stt", chain.stt), ("llm", chain.handler)):
            get_provider(cat, pid).ConfigModel(**getattr(doc, cat).instances.get(pid, {}))


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge `patch` into a copy of `base` (dicts merge, others replace)."""
    out = copy.deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _ref_name(node: dict) -> str | None:
    """Return the $defs name a core property references, or None for a scalar/list.

    Pydantic v2 emits sub-models either as a direct `$ref` or (when other keys like a
    default are present) wrapped under `allOf`/`anyOf`. Handle whichever is present."""
    ref = node.get("$ref")
    if ref is None:
        for combinator in ("allOf", "anyOf"):
            for sub in node.get(combinator, []):
                if isinstance(sub, dict) and "$ref" in sub:
                    ref = sub["$ref"]
                    break
            if ref is not None:
                break
    if not ref:
        return None
    return ref.rsplit("/", 1)[-1]   # "#/$defs/AudioConfig" -> "AudioConfig"


def _annotate_core_apply(schema: dict, action_for) -> None:
    """Inject the per-field `apply` class into the nested CoreConfig JSON Schema.

    For each top-level core property: if it references a sub-model, annotate each of that
    def's sub-properties with action_for(f"core.{section}.{subfield}"); if it is a scalar or
    list property, annotate the property itself with action_for(f"core.{section}"). Each
    core section has a UNIQUE def, so annotating the shared def in $defs is unambiguous."""
    defs = schema.get("$defs", {})
    for section, node in schema.get("properties", {}).items():
        ref = _ref_name(node)
        if ref is not None and ref in defs:
            sub = defs[ref]
            for subfield, subnode in sub.get("properties", {}).items():
                subnode["apply"] = action_for(f"core.{section}.{subfield}")
        else:
            # Scalar (tts_timeout/log_level) or list (mcp_servers/devices) property.
            node["apply"] = action_for(f"core.{section}")


class ConfigService:
    def __init__(self, doc: dict, deps: Deps, path: str | None = None):
        self._doc = ConfigDoc(**doc)      # raises on structural errors
        self._deps = deps
        self._path = path or config_store.DEFAULT_PATH
        self._cbs = []
        _validate(self._doc)

    @property
    def core(self) -> CoreConfig:
        return self._doc.core

    def document(self) -> dict:
        """Return the current raw config document (as plain JSON-able dict)."""
        return self._doc.model_dump(mode="json")

    def _slot(self, category) -> StageSlot:
        return getattr(self._doc, category)

    def get(self, category: str) -> BaseModel:
        """Validated config model of the SELECTED provider of a slot category."""
        return self.instance_config(category, self._slot(category).selected)

    def instance_config(self, category: str, provider_id: str) -> BaseModel:
        """Validated config model of one provider instance of a category."""
        prov = get_provider(category, provider_id)
        return prov.ConfigModel(**self._slot(category).instances.get(provider_id, {}))

    def create(self, category: str):
        """Build the backend of the SELECTED provider of a slot category."""
        return self.create_instance(category, self._slot(category).selected)

    def create_instance(self, category: str, provider_id: str):
        """Build the backend of one provider instance of a category."""
        prov = get_provider(category, provider_id)
        cfg = self.instance_config(category, provider_id)
        backend = prov.create(cfg, self._deps)
        self._stamp_desc(backend, prov, cfg)
        return backend

    @staticmethod
    def _stamp_desc(backend, prov, cfg) -> None:
        # Stamp the human-readable "provider/model" identity onto the backend so
        # log lines can name the exact backend that executed a stage. Best-effort:
        # a backend that rejects attributes simply stays unlabeled.
        try:
            backend.backend_desc = prov.describe(cfg)
        except Exception:
            pass

    def provider(self, category: str):
        """Return the selected provider of a slot category (for runtime reconfig)."""
        return get_provider(category, self._slot(category).selected)

    def tts_variants(self) -> list[str]:
        """Provider ids of the enabled TTS variants, in fallback order. Read per
        utterance, so a reorder applies without a rebuild."""
        return [v.provider for v in self._doc.tts.variants if v.enabled]

    def vocabulary(self, handler_id: str) -> list[str]:
        """The closed command vocabulary of an llm provider instance (e.g. Simple
        NLU's phrase/verb/number words) that a chain narrows its STT to; [] for a
        free-form LLM. Read per utterance, so handler edits apply without a rebuild."""
        return list(get_provider("llm", handler_id).vocabulary(self.instance_config("llm", handler_id)))

    def options(self, category, plugin, field, query: str = ""):
        prov = get_provider(category, plugin)
        cfg = prov.ConfigModel(**self._slot(category).instances.get(plugin, {}))
        return prov.options(field, cfg, self._deps, query)

    def create_adhoc(self, category, plugin, overrides: dict):
        """Build a transient provider backend from caller-supplied settings
        (e.g. the panel's unsaved draft), without touching the stored config
        or the running pipeline. ValidationError/ValueError propagate."""
        prov = get_provider(category, plugin)
        cfg = prov.ConfigModel(**(overrides or {}))
        backend = prov.create(cfg, self._deps)
        self._stamp_desc(backend, prov, cfg)
        return backend

    def catalog(self) -> dict:
        """Everything the panel needs: per-category providers + JSON Schema +
        plain values + core schema/values.

        The per-field `apply` class (the single source of truth for how a change is
        applied at runtime) is computed here from reconfig.action_for and injected into
        every field's schema, so the frontend never has to know the classification."""
        # Imported function-local to avoid an import cycle: reconfig does not import
        # config_service, but config_service.apply() imports reconfig.changed_paths, so
        # importing reconfig at module scope here would be circular at import time.
        from src.reconfig import action_for

        categories = []
        for cat in STAGE_CATEGORIES:
            slot = self._slot(cat)
            provs = []
            for pid, prov in providers(cat).items():
                values = prov.ConfigModel(**slot.instances.get(pid, {}))
                schema = prov.ConfigModel.model_json_schema()
                # Stage provider schemas are flat: annotate each field with its action.
                for field, node in schema.get("properties", {}).items():
                    node["apply"] = action_for(f"{cat}.instances.{pid}.{field}")
                entry = {
                    "id": pid,
                    "label": prov.label,
                    "schema": schema,
                    "values": values.model_dump(mode="json"),
                }
                # What a chain's `narrow` can do: an STT that takes a per-call
                # vocabulary, and a handler that has one to give.
                if cat == "stt":
                    entry["supports_vocabulary"] = prov.supports_vocabulary
                elif cat == "llm":
                    entry["has_vocabulary"] = bool(prov.vocabulary(values))
                provs.append(entry)
            if cat in POOL_CATEGORIES:
                categories.append({"id": cat, "providers": provs})
            elif cat == "tts":
                variants = [v.model_dump(mode="json") for v in slot.variants]
                categories.append({"id": cat, "variants": variants, "providers": provs})
            else:
                categories.append({"id": cat, "selected": slot.selected, "providers": provs})
        core_schema = CoreConfig.model_json_schema()
        _annotate_core_apply(core_schema, action_for)
        return {
            "categories": categories,
            "core": {
                "schema": core_schema,
                "values": self._doc.core.model_dump(mode="json"),
            },
        }

    def apply(self, patch: dict):
        """Deep-merge `patch` into the current document, re-validate, persist, then
        fire callbacks with the set of changed dotted paths. Raises (without
        persisting and without firing callbacks) on any validation error. Returns
        the changed paths.

        A partial patch only overrides the keys it carries; sibling keys survive the
        deep merge untouched."""
        base = self._doc.model_dump(mode="json")
        merged = _deep_merge(base, patch)
        new_doc = ConfigDoc(**merged)      # structural + core validation
        _validate(new_doc)

        # Compute the change set BEFORE swapping the document. Imported locally to
        # avoid any import-cycle risk with src.reconfig.
        from src.reconfig import changed_paths
        paths = changed_paths(base, merged)

        config_store.save(merged, self._path)
        self._doc = new_doc
        for cb in self._cbs:
            cb(paths)
        return paths

    def on_change(self, cb):
        self._cbs.append(cb)
