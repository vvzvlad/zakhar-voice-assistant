"""Unit tests for the RuAccent accent (Russian stress) stage.

The real `ruaccent` package and its models are NEVER imported here: the backend
takes an injected fake accentizer, the provider is exercised on the disabled
(passthrough) path, and registration is checked via the REGISTRY. So the suite
stays hermetic and the heavy dependency is not needed to run it.
"""

import httpx
import pytest

import src.plugins  # noqa: F401  register all providers
from src.stress import PassthroughAccentizer
from src.plugins.base import REGISTRY, Deps
from src.plugins.stress.ruaccent import (
    RuAccentBackend,
    RuAccentConfig,
    RuAccentProvider,
)


def _deps():
    return Deps(
        http_cloud=httpx.AsyncClient(),
        http_local=httpx.AsyncClient(),
        tts_timeout=30,
    )


class FakeAccentizer:
    """RUAccent double: records its input and returns a fixed sentinel."""

    def __init__(self, output="прив+ет м+ир"):
        self.output = output
        self.seen = []

    def process_all(self, text):
        self.seen.append(text)
        return self.output


async def test_backend_strips_existing_marks_then_delegates():
    # An existing "+vowel" mark must be stripped BEFORE the accentizer runs (so
    # RuAccent is the single source of stress, never producing "при+в+ет"), and
    # the backend returns the accentizer's output verbatim.
    fake = FakeAccentizer(output="прив+ет м+ир")
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake,
    )
    out = await backend.accentize("прив+ет мир")
    # The fake received the mark-free text (no "+").
    assert fake.seen == ["привет мир"]
    assert out == "прив+ет м+ир"


async def test_backend_stress_format_plus_keeps_native_notation():
    # "plus" (and the default) keep RuAccent's native "+vowel" output unchanged.
    fake = FakeAccentizer(output="прив+ет м+ир")
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake, stress_format="plus",
    )
    assert await backend.accentize("привет мир") == "прив+ет м+ир"


async def test_backend_stress_format_default_is_plus():
    # No stress_format passed -> behaves like "plus" (unchanged).
    fake = FakeAccentizer(output="прив+ет м+ир")
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake,
    )
    assert await backend.accentize("привет мир") == "прив+ет м+ир"


async def test_backend_stress_format_acute():
    # "acute" -> no "+", combining acute (U+0301) after each stressed vowel.
    fake = FakeAccentizer(output="прив+ет м+ир")
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake, stress_format="acute",
    )
    out = await backend.accentize("привет мир")
    assert "+" not in out
    assert "́" in out
    assert out == "приве́т ми́р"


async def test_backend_stress_format_uppercase():
    # "uppercase" -> no "+", the stressed vowel capitalised.
    fake = FakeAccentizer(output="прив+ет м+ир")
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake, stress_format="uppercase",
    )
    assert await backend.accentize("привет мир") == "привЕт мИр"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
async def test_backend_returns_blank_input_unchanged_without_calling(text):
    # Empty/whitespace input is returned unchanged and the accentizer is never invoked.
    fake = FakeAccentizer()
    backend = RuAccentBackend(
        model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
        device="CPU", workdir="/tmp", accentizer=fake,
    )
    assert await backend.accentize(text) == text
    assert fake.seen == []


async def test_passthrough_returns_text_unchanged():
    passthrough = PassthroughAccentizer()
    assert await passthrough.accentize("прив+ет мир") == "прив+ет мир"
    assert await passthrough.accentize("") == ""


def test_provider_disabled_returns_passthrough_without_model_load():
    # enabled=False must return a PassthroughAccentizer and never touch the model.
    backend = RuAccentProvider().create(RuAccentConfig(enabled=False), _deps())
    assert isinstance(backend, PassthroughAccentizer)


def test_provider_describe():
    prov = RuAccentProvider()
    assert prov.describe(RuAccentConfig(enabled=True)) == "ruaccent/turbo3.1"
    assert prov.describe(RuAccentConfig(enabled=False)) == "ruaccent/off"


def test_provider_is_registered():
    assert "ruaccent" in REGISTRY["stress"]
