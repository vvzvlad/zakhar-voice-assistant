import pytest
from pydantic import ValidationError

from src.core_config import CoreConfig, McpServerConfig, VadConfig


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
