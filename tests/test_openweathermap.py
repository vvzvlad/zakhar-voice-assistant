import httpx
import respx

from src.openweathermap import OPENWEATHERMAP_URL, get_weather_summary


SAMPLE_PAYLOAD = {
    "main": {"temp": 12.6},
    "wind": {"speed": 3.0},
    "weather": [{"description": "облачно с прояснениями"}],
}


@respx.mock
async def test_get_weather_summary_formats_russian_string():
    respx.get(OPENWEATHERMAP_URL).mock(return_value=httpx.Response(200, json=SAMPLE_PAYLOAD))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_summary(client, "Moscow", "key")
    # round(12.6) == 13; wind int(3.0) == 3.
    assert summary == "13 градусов, облачно с прояснениями, ветер 3 метров в секунду"


@respx.mock
async def test_get_weather_summary_non_200_returns_none():
    respx.get(OPENWEATHERMAP_URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_summary(client, "Moscow", "key")
    assert summary is None


@respx.mock
async def test_get_weather_summary_transport_error_returns_none():
    respx.get(OPENWEATHERMAP_URL).mock(side_effect=httpx.ConnectError("boom"))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_summary(client, "Moscow", "key")
    assert summary is None
