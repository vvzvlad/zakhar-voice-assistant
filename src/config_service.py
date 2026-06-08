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

STAGE_CATEGORIES = ("stt", "llm", "tts")


class StageSlot(BaseModel):
    selected: str
    instances: dict[str, dict] = {}


class ConfigDoc(BaseModel):
    version: int = 1
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

    def options(self, category, plugin, field):
        prov = get_provider(category, plugin)
        cfg = prov.ConfigModel(**self._slot(category).instances.get(plugin, {}))
        return prov.options(field, cfg, self._deps)

    def catalog(self) -> dict:
        """Everything the panel needs: per-category providers + JSON Schema +
        plain values + core schema/values."""
        categories = []
        for cat in STAGE_CATEGORIES:
            slot = self._slot(cat)
            provs = []
            for pid, prov in providers(cat).items():
                values = prov.ConfigModel(**slot.instances.get(pid, {}))
                provs.append({
                    "id": pid,
                    "label": prov.label,
                    "schema": prov.ConfigModel.model_json_schema(),
                    "values": values.model_dump(mode="json"),
                })
            categories.append({"id": cat, "selected": slot.selected, "providers": provs})
        return {
            "categories": categories,
            "core": {
                "schema": CoreConfig.model_json_schema(),
                "values": self._doc.core.model_dump(mode="json"),
            },
        }

    def apply(self, patch: dict):
        """Deep-merge `patch` into the current document, re-validate, persist, then
        fire callbacks. Raises (without persisting) on any validation error.

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

        config_store.save(merged, self._path)
        self._doc = new_doc
        for cb in self._cbs:
            cb()

    def on_change(self, cb):
        self._cbs.append(cb)
