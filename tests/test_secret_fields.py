"""Safety net: every genuine credential config field must be explicitly tagged
`secret` so the admin panel masks it.

Masking in the panel is driven ONLY by the explicit `json_schema_extra={"secret":
True}` flag (frontend reads `node.secret === true`); the old name-heuristic that
guessed "this looks like a key" was removed because it both leaked (an untagged
credential would render as plaintext) and over-masked (any name containing "key",
e.g. the wakeword `keywords` field). This test is that runtime heuristic's
replacement: it walks every registered provider ConfigModel and every CoreConfig
sub-section and asserts that any field whose NAME is a known credential carries
the flag — so a future untagged credential fails CI instead of silently rendering
as plaintext.

It is a PRECISE-name guard: it keys off an exact-match SENSITIVE set, so it never
flags non-secret fields like `keywords` / `model_path` / `host`.
"""

import typing

import pytest
from pydantic import BaseModel

import src.plugins  # noqa: F401  triggers @register on all providers
from src.core_config import CoreConfig
from src.plugins.base import REGISTRY

# Exact field names that are genuine credentials and MUST be masked. Exact match
# (not substring): `keywords`, `keyboard_layout`, `model_path` etc. are NOT in here.
SENSITIVE = {"api_key", "token", "password", "psk", "secret", "client_secret", "access_token"}


def _nested_models(model: type[BaseModel], seen: set[type] | None = None) -> set[type[BaseModel]]:
    """All BaseModel subclasses reachable from `model`'s fields (incl. itself),
    looking through list/optional/union annotations so e.g. list[DeviceConfig] and
    Optional[Foo] sub-sections are covered."""
    if seen is None:
        seen = set()
    if model in seen:
        return seen
    seen.add(model)
    for field in model.model_fields.values():
        for arg in _iter_annotation_models(field.annotation):
            _nested_models(arg, seen)
    return seen


def _iter_annotation_models(annotation) -> list[type[BaseModel]]:
    """Yield every BaseModel subclass mentioned in a (possibly generic) annotation."""
    out: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        out.append(annotation)
        return out
    # list[X], Optional[X] (= Union[X, None]), dict[..., X], etc.
    for arg in typing.get_args(annotation):
        out.extend(_iter_annotation_models(arg))
    return out


def _all_config_models() -> list[type[BaseModel]]:
    """Every provider ConfigModel plus every CoreConfig sub-section (deduped)."""
    models: set[type[BaseModel]] = set()
    for category in REGISTRY.values():
        for provider in category.values():
            models.add(provider.ConfigModel)
    models |= _nested_models(CoreConfig)
    return sorted(models, key=lambda m: m.__name__)


def _sensitive_fields() -> list[tuple[str, str]]:
    """(model_name, field_name) pairs for every credential-named field anywhere in
    the config surface."""
    pairs: list[tuple[str, str]] = []
    for model in _all_config_models():
        for fname in model.model_fields:
            if fname in SENSITIVE:
                pairs.append((model.__name__, fname))
    return pairs


def test_every_credential_field_is_tagged_secret():
    """No silent leaks: a credential-named field that forgot the secret flag fails here."""
    pairs = _sensitive_fields()
    # Sanity: the config surface really does contain credential fields, so an empty
    # walk (e.g. a broken discovery) cannot make this test vacuously pass.
    assert pairs, "no credential fields discovered — discovery is broken"
    untagged: list[str] = []
    for model in _all_config_models():
        for fname, field in model.model_fields.items():
            if fname not in SENSITIVE:
                continue
            extra = field.json_schema_extra or {}
            if not (isinstance(extra, dict) and extra.get("secret") is True):
                untagged.append(f"{model.__name__}.{fname}")
    assert not untagged, (
        "these credential fields are not tagged json_schema_extra={'secret': True} "
        f"and would render as plaintext in the panel: {untagged}"
    )


def test_secret_flag_surfaces_in_json_schema():
    """The flag must reach the emitted JSON Schema (the panel reads model_json_schema,
    not the raw FieldInfo), as `secret: true` on the property node."""
    for model in _all_config_models():
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        for fname in model.model_fields:
            if fname in SENSITIVE:
                assert props.get(fname, {}).get("secret") is True, (
                    f"{model.__name__}.{fname}: secret flag missing from model_json_schema()"
                )


@pytest.mark.parametrize("non_secret", ["keywords", "model_path", "host", "voice", "keyboard_layout"])
def test_precise_name_guard_ignores_non_secret_fields(non_secret):
    """The guard keys off an EXACT-name set, so non-secret fields are never flagged —
    this is the bug the name heuristic had (masking `keywords` / `keyboard_layout`)."""
    assert non_secret not in SENSITIVE
