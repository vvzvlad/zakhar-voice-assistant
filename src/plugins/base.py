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


class Provider:
    """Base for all stage providers. Subclasses set the class attrs and implement create()."""

    category: str = ""        # "stt" | "llm" | "tts"
    id: str = ""              # unique within category, e.g. "yandex"
    label: str = ""           # human label for the UI
    ConfigModel: type[BaseModel] = BaseModel

    def create(self, cfg: BaseModel, deps: Deps):
        raise NotImplementedError

    def options(self, field: str, cfg: BaseModel, deps: Deps):
        return None           # optional dynamic option lists (voices/models)


REGISTRY: dict[str, dict[str, Provider]] = defaultdict(dict)


def register(cls):
    """Class decorator: instantiate and register the provider under its category/id."""
    inst = cls()
    REGISTRY[inst.category][inst.id] = inst
    return cls


def get_provider(category: str, id: str) -> Provider:
    try:
        return REGISTRY[category][id]
    except KeyError:
        raise ValueError(f"Unknown {category} provider: {id!r}")


def providers(category: str) -> dict[str, Provider]:
    return dict(REGISTRY.get(category, {}))
