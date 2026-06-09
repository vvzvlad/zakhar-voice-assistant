"""Unit tests for version._resolve_version precedence.

These exercise the resolution order (git -> build stamp -> hardcoded fallback)
without spawning real git: both source helpers are monkeypatched, so the test is
deterministic and independent of the checkout state / Docker stamp.
"""

import src.version as version


def test_resolve_version_prefers_git(monkeypatch):
    # When git describe yields a value, it wins regardless of the stamp.
    monkeypatch.setattr(version, "_from_git", lambda: "1.2.3")
    monkeypatch.setattr(version, "_from_stamp", lambda: "2.0.0")
    assert version._resolve_version() == "1.2.3"


def test_resolve_version_falls_back_to_stamp(monkeypatch):
    # No git checkout -> the build stamp is used.
    monkeypatch.setattr(version, "_from_git", lambda: None)
    monkeypatch.setattr(version, "_from_stamp", lambda: "2.0.0")
    assert version._resolve_version() == "2.0.0"


def test_resolve_version_falls_back_to_constant(monkeypatch):
    # Neither git nor stamp available -> the hardcoded fallback constant.
    monkeypatch.setattr(version, "_from_git", lambda: None)
    monkeypatch.setattr(version, "_from_stamp", lambda: None)
    assert version._resolve_version() == version._FALLBACK
    assert version._FALLBACK == "0.0.0+unknown"
