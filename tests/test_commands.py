import httpx
import respx

from src.commands import (
    extract_command_blocks,
    parse_command_payload,
    process_commands_in_content,
)
from src.settings import settings


def test_extract_command_blocks_multiple_case_insensitive_multiline():
    text = (
        "before <command>room_light:on</command> middle\n"
        "<COMMAND>kitchen_light:off</COMMAND>\n"
        "<command>multi\nline</command> end"
    )
    blocks = extract_command_blocks(text)
    assert blocks == ["room_light:on", "kitchen_light:off", "multi\nline"]


def test_parse_command_payload_on_off():
    assert parse_command_payload("room_light:on") == {"device_id": "room_light", "value": "on"}
    assert parse_command_payload(" kitchen_light : off ") == {"device_id": "kitchen_light", "value": "off"}


def test_parse_command_payload_numeric():
    assert parse_command_payload("room_ac:22") == {"device_id": "room_ac", "value": "22"}


def test_parse_command_payload_garbage_returns_none():
    assert parse_command_payload("this is not a command") is None
    assert parse_command_payload("") is None


@respx.mock
async def test_process_commands_in_content_dispatches_post():
    route = respx.post(settings.smarthome_url).mock(return_value=httpx.Response(200))
    content = (
        "Включаю свет.\n"
        "<command>room_light:on</command>\n"
        "<command>room_ac:22</command>"
    )
    async with httpx.AsyncClient(verify=False) as client:
        parsed = await process_commands_in_content(client, content)

    assert parsed == [
        {"device_id": "room_light", "value": "on"},
        {"device_id": "room_ac", "value": "22"},
    ]
    assert route.call_count == 2
    # Verify the request body shape for the first dispatched command.
    first_request = route.calls[0].request
    import json
    assert json.loads(first_request.content) == {"command": {"device_id": "room_light", "value": "on"}}


@respx.mock
async def test_process_commands_in_content_no_blocks():
    route = respx.post(settings.smarthome_url).mock(return_value=httpx.Response(200))
    async with httpx.AsyncClient(verify=False) as client:
        parsed = await process_commands_in_content(client, "no commands here")
    assert parsed == []
    assert route.call_count == 0
