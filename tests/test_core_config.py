import pytest
from pydantic import ValidationError

from src.core_config import (
    AckConfig,
    CalendarConfig,
    CoreConfig,
    McpServerConfig,
    OpenWeatherMapConfig,
    VadConfig,
)


def test_mcp_servers_accepts_list_of_entries():
    core = CoreConfig(
        mcp_servers=[
            {"name": "home", "url": "http://ha/sse", "token": "t", "transport": "sse", "prompt": "lights"},
            {"name": "other", "url": "http://other"},
        ]
    )
    assert [s.name for s in core.mcp_servers] == ["home", "other"]
    first = core.mcp_servers[0]
    assert isinstance(first, McpServerConfig)
    assert first.transport == "sse"
    assert first.prompt == "lights"
    # Defaults for the second entry.
    assert core.mcp_servers[1].transport == "auto"
    assert core.mcp_servers[1].token == ""
    assert core.mcp_servers[1].prompt == ""


def test_mcp_servers_defaults_to_empty_list():
    assert CoreConfig().mcp_servers == []


def test_mcp_server_transport_literal_is_validated():
    with pytest.raises(ValidationError):
        McpServerConfig(name="bad", url="http://x", transport="websocket")


def test_mcp_server_requires_name():
    with pytest.raises(ValidationError):
        McpServerConfig(url="http://x")


def test_vad_trim_start_ms_defaults_to_200():
    assert CoreConfig().vad.trim_start_ms == 200


def test_vad_trim_start_ms_rejects_negative():
    with pytest.raises(ValidationError):
        VadConfig(trim_start_ms=-1)


def test_builtin_mcp_prompts_default_to_empty_and_accept_values():
    # Built-in weather/calendar sources expose their own optional prompt block,
    # empty by default and accepting a provided value (same as McpServerConfig.prompt).
    assert OpenWeatherMapConfig().prompt == ""
    assert CalendarConfig().prompt == ""
    assert OpenWeatherMapConfig(prompt="x").prompt == "x"
    assert CalendarConfig(prompt="y").prompt == "y"


def test_ack_config_defaults():
    # The end-of-phrase ack chime is ON by default with no sound-file override
    # (an empty path means the synthesized two-tone chime is used).
    ack = CoreConfig().ack
    assert ack.enabled is True
    assert ack.sound_path == ""


def test_ack_config_accepts_overrides():
    ack = AckConfig(enabled=False, sound_path="data/blyam.wav")
    assert ack.enabled is False
    assert ack.sound_path == "data/blyam.wav"


def test_builtin_prompt_fields_carry_textarea_widget():
    # The panel renders these prompt fields as a multi-line textarea via the
    # `widget` schema hint (SchemaForm), not a single-line input.
    owm = OpenWeatherMapConfig.model_json_schema()["properties"]["prompt"]
    cal = CalendarConfig.model_json_schema()["properties"]["prompt"]
    assert owm["widget"] == "textarea"
    assert cal["widget"] == "textarea"


def test_vad_mic_defaults_are_channel0_unity_gain():
    # Mic channel/gain live inside VadConfig (the Voice-capture section), not a
    # separate core.mic section. Defaults: processed channel 0, unity gain.
    v = VadConfig()
    assert v.mic_channel == 0
    assert v.mic_gain == 1.0


def test_vad_mic_channel_literal_rejects_out_of_range():
    VadConfig(mic_channel=0)
    VadConfig(mic_channel=1)
    with pytest.raises(ValidationError):
        VadConfig(mic_channel=2)


def test_vad_mic_gain_range_validated():
    # Gain spans 0.1–50.0: values below 1.0 attenuate, above amplify.
    VadConfig(mic_gain=0.1)   # min
    VadConfig(mic_gain=0.5)   # below 1.0 is valid (attenuation)
    VadConfig(mic_gain=50.0)  # max
    with pytest.raises(ValidationError):
        VadConfig(mic_gain=0.05)  # below 0.1
    with pytest.raises(ValidationError):
        VadConfig(mic_gain=50.1)  # above 50.0


def test_core_config_vad_has_mic_fields():
    assert CoreConfig().vad.mic_channel == 0
    assert CoreConfig().vad.mic_gain == 1.0
