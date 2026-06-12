import pytest
from pydantic import ValidationError

from src.core_config import (
    AckConfig,
    AgentMcpConfig,
    CalendarConfig,
    CoreConfig,
    DeviceConfig,
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


def test_device_config_enabled_defaults_true_and_round_trips_false():
    # Old config.json entries carry no `enabled` -> they must validate as enabled.
    d = DeviceConfig(name="a", host="10.0.0.1", psk="p")
    assert d.enabled is True
    # enabled=False survives a dump/validate round trip (persisted config flag).
    off = DeviceConfig(name="a", host="10.0.0.1", psk="p", enabled=False)
    assert DeviceConfig.model_validate(off.model_dump()).enabled is False


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


def test_agent_mcp_config_defaults():
    # The agent-facing MCP endpoint (served by the panel at /mcp) is ON by
    # default; old docs carry no core.agent_mcp key, so the default must cover them.
    amcp = AgentMcpConfig()
    assert amcp.enabled is True


def test_core_config_has_agent_mcp_section():
    core = CoreConfig()
    assert isinstance(core.agent_mcp, AgentMcpConfig)
    assert core.agent_mcp.enabled is True


def test_agent_mcp_config_accepts_overrides_and_ignores_stale_keys():
    # Old docs may still carry host/port (the endpoint used to be a standalone
    # server): pydantic must ignore them on parse — no migration needed.
    amcp = AgentMcpConfig(enabled=False, host="127.0.0.1", port=9000)
    assert amcp.enabled is False
    assert not hasattr(amcp, "host")
    assert not hasattr(amcp, "port")


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


def test_vad_mic_defaults_are_channel0_conditioning_off():
    # Mic channel + conditioning toggles live inside VadConfig (the Voice-capture
    # section), not a separate core.mic section. Defaults: processed channel 0, the
    # peak-normalize, high-pass and decision-only VAD auto-gain toggles all OFF.
    v = VadConfig()
    assert v.mic_channel == 0
    assert v.mic_normalize is False
    assert v.mic_highpass is False
    assert v.mic_auto_gain is False


def test_vad_mic_channel_literal_rejects_out_of_range():
    VadConfig(mic_channel=0)
    VadConfig(mic_channel=1)
    with pytest.raises(ValidationError):
        VadConfig(mic_channel=2)


def test_vad_mic_conditioning_toggles_accept_overrides():
    # The conditioning toggles are independent bools that can be turned on.
    v = VadConfig(mic_normalize=True, mic_highpass=True, mic_auto_gain=True)
    assert v.mic_normalize is True
    assert v.mic_highpass is True
    assert v.mic_auto_gain is True


def test_core_config_vad_has_mic_fields():
    vad = CoreConfig().vad
    assert vad.mic_channel == 0
    assert vad.mic_normalize is False
    assert vad.mic_highpass is False


def test_audio_stream_tts_defaults_true():
    # Streaming TTS delivery is on by default; old docs carry no
    # core.audio.stream_tts key, so the default must cover them.
    assert CoreConfig().audio.stream_tts is True
