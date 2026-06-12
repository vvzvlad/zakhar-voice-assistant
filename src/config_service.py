"""Plugin-agnostic config core: catalog / get / create / apply / options.

ConfigService is the only thing the panel/API talks to. It knows about categories,
provider ids, their ConfigModel schema and the runtime object create() returns — but
never about any concrete provider field. Stage providers come from the REGISTRY; core
sections come from CoreConfig. Both expose themselves to the UI as JSON Schema.

Store path injection: the constructor takes an optional `path` (defaults to
config_store.DEFAULT_PATH). apply() persists there. This is cleaner than monkeypatching
the module global and lets tests point at a tmp file.
"""

import copy

from pydantic import BaseModel

from src import config_store
from src.core_config import CoreConfig
from src.plugins.base import Deps, get_provider, providers

STAGE_CATEGORIES = ("vad", "stt", "llm", "tts")


class StageSlot(BaseModel):
    selected: str
    instances: dict[str, dict] = {}


class ConfigDoc(BaseModel):
    version: int = 1
    # vad has a default slot (unlike stt/llm/tts) so configs written before the VAD
    # stage existed still parse; the webrtc provider's own defaults then apply.
    vad: StageSlot = StageSlot(selected="webrtc")
    stt: StageSlot
    llm: StageSlot
    tts: StageSlot
    core: CoreConfig = CoreConfig()


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
        # Eagerly validate the selected providers' instance configs.
        for cat in STAGE_CATEGORIES:
            self.get(cat)

    @property
    def core(self) -> CoreConfig:
        return self._doc.core

    def document(self) -> dict:
        """Return the current raw config document (as plain JSON-able dict)."""
        return self._doc.model_dump(mode="json")

    def _slot(self, category) -> StageSlot:
        return getattr(self._doc, category)

    def get(self, category: str) -> BaseModel:
        """Validated config model of the SELECTED provider for this category."""
        slot = self._slot(category)
        prov = get_provider(category, slot.selected)
        return prov.ConfigModel(**slot.instances.get(slot.selected, {}))

    def create(self, category: str):
        slot = self._slot(category)
        prov = get_provider(category, slot.selected)
        return prov.create(self.get(category), self._deps)

    def provider(self, category: str):
        """Return the selected provider instance for a category (for runtime reconfig)."""
        return get_provider(category, self._slot(category).selected)

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
        return prov.create(cfg, self._deps)

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
                provs.append({
                    "id": pid,
                    "label": prov.label,
                    "schema": schema,
                    "values": values.model_dump(mode="json"),
                })
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
        # Validate every touched provider's selected instance config.
        for cat in STAGE_CATEGORIES:
            slot = getattr(new_doc, cat)
            prov = get_provider(cat, slot.selected)
            prov.ConfigModel(**slot.instances.get(slot.selected, {}))

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
