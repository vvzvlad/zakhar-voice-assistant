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

from pydantic import BaseModel, SecretStr

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


def _is_secret_field(field) -> bool:
    """A field is secret if its schema extra marks it, or its type is SecretStr."""
    extra = field.json_schema_extra
    if isinstance(extra, dict) and extra.get("secret") is True:
        return True
    return field.annotation is SecretStr


def mask_secrets(model: BaseModel) -> dict:
    """Dump a pydantic model to a JSON-able dict with every secret field replaced
    by {"is_set": <bool>}. Recurses into nested models and lists of models. Never
    emits secret plaintext."""

    def _value(field, val):
        if _is_secret_field(field):
            return {"is_set": _truthy_secret(val)}
        return _dump(val)

    def _truthy_secret(val) -> bool:
        if isinstance(val, SecretStr):
            return bool(val.get_secret_value())
        return bool(val)

    def _dump(val):
        if isinstance(val, BaseModel):
            out = {}
            for name, field in val.__class__.model_fields.items():
                out[name] = _value(field, getattr(val, name))
            return out
        if isinstance(val, list):
            return [_dump(v) for v in val]
        if isinstance(val, dict):
            return {k: _dump(v) for k, v in val.items()}
        if isinstance(val, SecretStr):
            # A bare SecretStr reached without field context — never leak it.
            return {"is_set": bool(val.get_secret_value())}
        return val

    return _dump(model)


def _reveal(val):
    """Convert a model_dump(mode='python') tree into a JSON-able dict with SecretStr
    revealed to plaintext. Used to build the merge/persist base so secrets survive an
    apply() round-trip (mode='json' would have masked them to '**********')."""
    if isinstance(val, SecretStr):
        return val.get_secret_value()
    if isinstance(val, dict):
        return {k: _reveal(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_reveal(v) for v in val]
    return val


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
        masked values + core schema/values."""
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
                    "values": mask_secrets(values),
                })
            categories.append({"id": cat, "selected": slot.selected, "providers": provs})
        return {
            "categories": categories,
            "core": {
                "schema": CoreConfig.model_json_schema(),
                "values": mask_secrets(self._doc.core),
            },
        }

    def apply(self, patch: dict):
        """Deep-merge `patch` into the current document, re-validate, persist, then
        fire callbacks. Raises (without persisting) on any validation error.

        Masked-secret contract: the panel MUST NOT send back masked secret
        placeholders (the {"is_set": ...} shape catalog() emits). To keep a secret's
        current value, omit that field from the patch — it survives the merge
        untouched. Sending a masked placeholder for a secret field fails validation
        by design (a dict is not a valid SecretStr), so the file is NOT written and
        the plaintext is never clobbered with the placeholder."""
        base = _reveal(self._doc.model_dump(mode="python"))
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
