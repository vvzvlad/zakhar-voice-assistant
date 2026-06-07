import pytest
from pydantic import ValidationError

from src.settings import DeviceConfig, Settings


def _set_required(monkeypatch):
    """Set every required env var so Settings() can be constructed."""
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("WEATHER_API_KEY", "w-key")
    monkeypatch.setenv("MCP_SMARTHOME_URL", "http://mcp.test:8201/mcp")
    monkeypatch.setenv("TTS_BASE_URL", "http://tts.test:8124")
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://10.0.0.10:8200")
    monkeypatch.setenv("ESPHOME_DEVICES", "living|10.0.0.5|cHNr")


def test_loads_from_env(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    s = Settings(_env_file=None)
    assert s.groq_api_key == "g-key"
    assert s.weather_api_key == "w-key"
    assert s.mcp_smarthome_url == "http://mcp.test:8201/mcp"
    assert s.tts_base_url == "http://tts.test:8124"
    assert s.public_base_url == "http://10.0.0.10:8200"
    assert s.log_level == "DEBUG"
    # Defaults applied for non-secret config.
    assert s.esphome_port == 6053
    assert s.audio_port == 8200
    assert s.groq_proxy == ""


def test_missing_credential_fails(monkeypatch):
    # A missing credential must blow up at construction time, not silently default.
    _set_required(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_devices_single(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ESPHOME_DEVICES", "living|10.0.0.5|cHNr")
    s = Settings(_env_file=None)
    devices = s.devices
    assert devices == [DeviceConfig(name="living", host="10.0.0.5", psk="cHNr")]


def test_devices_multiple_in_order(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ESPHOME_DEVICES", "a|h1|p1;b|h2|p2")
    s = Settings(_env_file=None)
    devices = s.devices
    assert devices == [
        DeviceConfig(name="a", host="h1", psk="p1"),
        DeviceConfig(name="b", host="h2", psk="p2"),
    ]


def test_devices_trailing_semicolon_skipped(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ESPHOME_DEVICES", "a|h1|p1;")
    s = Settings(_env_file=None)
    assert s.devices == [DeviceConfig(name="a", host="h1", psk="p1")]


def test_devices_whitespace_stripped(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ESPHOME_DEVICES", " a | h1 | p1 ; b|h2|p2 ")
    s = Settings(_env_file=None)
    assert s.devices == [
        DeviceConfig(name="a", host="h1", psk="p1"),
        DeviceConfig(name="b", host="h2", psk="p2"),
    ]


def test_devices_malformed_raises(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ESPHOME_DEVICES", "bad|only2")
    s = Settings(_env_file=None)
    with pytest.raises(ValueError):
        _ = s.devices
