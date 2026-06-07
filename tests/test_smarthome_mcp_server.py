import json

import httpx
import respx

from src.smarthome_mcp_server import (
    cfg,
    climate_command,
    dimmer_command,
    dispatch,
    light_command,
    lock_command,
    switch_command,
)


@respx.mock
async def test_dispatch_posts_node_red_payload():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    result = await dispatch("room_light", "on")
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "command": {"device_id": "room_light", "value": "on"}
    }
    assert "ok" in result


@respx.mock
async def test_set_light_normalizes_on_off():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    await light_command("bright_room_light", "ON")
    await light_command("kitchen_light", "off")
    values = [json.loads(c.request.content)["command"]["value"] for c in route.calls]
    assert values == ["on", "off"]


@respx.mock
async def test_set_dimmer_value_and_clamping():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    await dimmer_command("night_light", 80)
    await dimmer_command("night_light", 150)
    await dimmer_command("night_light", -5)
    values = [json.loads(c.request.content)["command"]["value"] for c in route.calls]
    assert values == ["80", "100", "0"]


@respx.mock
async def test_set_climate_number_and_off():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    await climate_command("room_ac", "22")
    await climate_command("room_ac", "off")
    values = [json.loads(c.request.content)["command"]["value"] for c in route.calls]
    assert values == ["22", "off"]


@respx.mock
async def test_set_lock_and_switch_normalize():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    await lock_command("main_lock", "unlock")
    await switch_command("monitors", "on")
    values = [json.loads(c.request.content)["command"]["value"] for c in route.calls]
    assert values == ["unlock", "on"]


@respx.mock
async def test_invalid_device_id_does_not_post():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    out = await light_command("not_a_device", "on")
    assert out.startswith("error")
    assert route.call_count == 0


@respx.mock
async def test_invalid_state_does_not_post():
    route = respx.post(cfg.smarthome_url).mock(return_value=httpx.Response(200))
    out = await light_command("bright_room_light", "blink")
    assert out.startswith("error")
    assert route.call_count == 0
