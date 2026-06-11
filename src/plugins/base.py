"""Plugin foundation: the provider base class, the static registry and a Deps bag.

The config core is plugin-agnostic: it only knows about categories, provider ids,
their ConfigModel (a pydantic schema) and the runtime object create() returns. Each
provider declares its own settings schema, so adding one is a single new module that
self-registers via @register on `import src.plugins`.
"""

from collections import defaultdict
from dataclasses import dataclass

import httpx
from pydantic import BaseModel


@dataclass
class Deps:
    """Shared runtime deps handed to provider.create()."""

    http_cloud: httpx.AsyncClient   # proxied client for cloud APIs (STT/LLM/Yandex TTS)
    http_local: httpx.AsyncClient   # direct client for local services (TeraTTS)
    tts_timeout: int = 30


# Shared schema annotation for the `model` field: rendered as a dynamic select
# (option list fetched from the provider's model-list API) that still accepts an
# arbitrary, not-listed model id (`freeform` is consumed by the frontend).
# Subclasses overriding `model` must re-attach this dict explicitly — pydantic
# does NOT inherit Field metadata on overridden fields.
MODEL_FIELD_EXTRA: dict = {"widget": "select", "options": "dynamic", "freeform": True}


class Provider:
    """Base for all stage providers. Subclasses set the class attrs and implement create()."""

    category: str = ""        # "stt" | "llm" | "tts"
    id: str = ""              # unique within category, e.g. "yandex"
    label: str = ""           # human label for the UI
    ConfigModel: type[BaseModel] = BaseModel
    uses_http_cloud: bool = False   # True if create() builds a backend bound to deps.http_cloud (proxied)

    def create(self, cfg: BaseModel, deps: Deps):
        raise NotImplementedError

    def options(self, field: str, cfg: BaseModel, deps: Deps, query: str = ""):
        """Optional dynamic option lists (voices/models) for a config field.

        May return a list (or None for "no dynamic options for this field"), OR an
        awaitable resolving to such a list for network-backed catalogs (e.g. model
        lists fetched from the provider's API). Callers that may receive a
        network-backed list must check `inspect.isawaitable` and await it.

        `query` is an optional user-typed search string for provider catalogs that
        support server-side search (fields marked `"search": "remote"` in their
        json_schema_extra); providers without server-side search ignore it (their
        full list is filtered client-side)."""
        return None


REGISTRY: dict[str, dict[str, Provider]] = defaultdict(dict)


def register(cls):
    """Class decorator: instantiate and register the provider under its category/id."""
    inst = cls()
    # Refuse to silently shadow an already-registered provider: as the set grows a
    # duplicate category/id would otherwise overwrite the earlier one without a trace.
    if inst.id in REGISTRY[inst.category]:
        raise ValueError(f"Duplicate provider {inst.category}/{inst.id!r} already registered")
    REGISTRY[inst.category][inst.id] = inst
    return cls


def get_provider(category: str, id: str) -> Provider:
    try:
        return REGISTRY[category][id]
    except KeyError:
        raise ValueError(f"Unknown {category} provider: {id!r}")


def providers(category: str) -> dict[str, Provider]:
    return dict(REGISTRY.get(category, {}))
