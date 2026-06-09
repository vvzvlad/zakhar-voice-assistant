from datetime import datetime, timedelta

import httpx
import respx

from src.openweathermap import (
    OPENWEATHERMAP_FORECAST_URL,
    OPENWEATHERMAP_URL,
    get_weather_forecast,
    get_weather_summary,
)


def _ts(dt):
    # Local datetime -> unix timestamp int (matches datetime.fromtimestamp in src).
    return int(dt.timestamp())


def _forecast_item(dt, temp=None, wind=None, desc=None, pop=None):
    item = {"dt": _ts(dt)}
    if temp is not None:
        item["main"] = {"temp": temp}
    if wind is not None:
        item["wind"] = {"speed": wind}
    if desc is not None:
        item["weather"] = [{"description": desc}]
    if pop is not None:
        item["pop"] = pop
    return item


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


@respx.mock
async def test_get_weather_forecast_aggregates_target_day():
    # Tomorrow is inside the ~5 day window; derive the target from the local "now"
    # and build dt values from the SAME local day so the test is timezone-agnostic.
    # Single `now` snapshot so target/other can never straddle a midnight boundary.
    now = datetime.now()
    target = (now + timedelta(days=1)).date()
    other = (now + timedelta(days=2)).date()

    payload = {
        "list": [
            _forecast_item(datetime(target.year, target.month, target.day, 9), temp=8, wind=2, desc="облачно", pop=0.1),
            _forecast_item(datetime(target.year, target.month, target.day, 12), temp=12, wind=4, desc="ясно", pop=0.0),
            _forecast_item(datetime(target.year, target.month, target.day, 15), temp=15, wind=5, desc="дождь", pop=0.6),
            # Different day: must be filtered out.
            _forecast_item(datetime(other.year, other.month, other.day, 12), temp=30, wind=9, desc="жара", pop=0.0),
        ]
    }
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)

    assert summary is not None
    assert summary.startswith(f"Прогноз на {target.isoformat()}: ")
    assert "от 8 до 15 градусов" in summary
    # 12:00 is closest to midday (13:00), so "ясно" wins the representative pick.
    assert "ясно" in summary
    assert "ветер до 5 метров в секунду" in summary
    assert "вероятность осадков 60 процентов" in summary
    # The other-day item must not leak in.
    assert "30" not in summary


@respx.mock
async def test_get_weather_forecast_out_of_range_returns_message():
    now = datetime.now()
    today = now.date()
    far = (now + timedelta(days=10)).date()
    payload = {
        "list": [
            _forecast_item(datetime(today.year, today.month, today.day, 12), temp=10, wind=3, desc="ясно", pop=0.0),
        ]
    }
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", far)

    assert summary is not None
    assert summary == (
        f"Прогноз на {far.isoformat()} недоступен: "
        f"прогноз есть только на ближайшие 5 дней."
    )


@respx.mock
async def test_single_temp_when_min_equals_max():
    # All matched entries share the same temperature -> single-temp branch.
    target = (datetime.now() + timedelta(days=1)).date()
    payload = {
        "list": [
            _forecast_item(datetime(target.year, target.month, target.day, 9), temp=7, wind=2, desc="облачно", pop=0.0),
            _forecast_item(datetime(target.year, target.month, target.day, 12), temp=7, wind=3, desc="ясно", pop=0.0),
            _forecast_item(datetime(target.year, target.month, target.day, 15), temp=7, wind=4, desc="дождь", pop=0.0),
        ]
    }
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)

    assert summary is not None
    assert "7 градусов" in summary
    assert "от " not in summary


@respx.mock
async def test_robust_to_missing_fields():
    # Among valid entries, include partial/malformed ones; defensive skips must hold.
    target = (datetime.now() + timedelta(days=1)).date()
    payload = {
        "list": [
            # Fully valid entry.
            _forecast_item(datetime(target.year, target.month, target.day, 12), temp=11, wind=4, desc="ясно", pop=0.2),
            # Missing `main`.
            {"dt": _ts(datetime(target.year, target.month, target.day, 9)), "wind": {"speed": 2}, "weather": [{"description": "облачно"}], "pop": 0.1},
            # Missing `weather`.
            {"dt": _ts(datetime(target.year, target.month, target.day, 15)), "main": {"temp": 9}, "wind": {"speed": 3}, "pop": 0.1},
            # Missing `wind`.
            {"dt": _ts(datetime(target.year, target.month, target.day, 18)), "main": {"temp": 8}, "weather": [{"description": "дождь"}], "pop": 0.1},
            # Non-numeric values for `main`/`wind`/`pop`.
            {"dt": _ts(datetime(target.year, target.month, target.day, 21)), "main": {"temp": "x"}, "wind": {"speed": "x"}, "pop": "x"},
        ]
    }
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)

    assert summary is not None
    assert summary.startswith(f"Прогноз на {target.isoformat()}: ")
    # The valid entry's temperature (11) must be present.
    assert "11" in summary


@respx.mock
async def test_pop_below_threshold_omitted():
    # All matched pop values are below the 0.1 threshold -> no precipitation phrase.
    target = (datetime.now() + timedelta(days=1)).date()
    payload = {
        "list": [
            _forecast_item(datetime(target.year, target.month, target.day, 9), temp=8, wind=2, desc="облачно", pop=0.0),
            _forecast_item(datetime(target.year, target.month, target.day, 12), temp=12, wind=4, desc="ясно", pop=0.05),
        ]
    }
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)

    assert summary is not None
    assert "вероятность осадков" not in summary


@respx.mock
async def test_empty_list_returns_none():
    target = (datetime.now() + timedelta(days=1)).date()
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(200, json={"list": []}))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)
    assert summary is None


@respx.mock
async def test_get_weather_forecast_non_200_returns_none():
    target = (datetime.now() + timedelta(days=1)).date()
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(return_value=httpx.Response(401, text="unauthorized"))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)
    assert summary is None


@respx.mock
async def test_get_weather_forecast_transport_error_returns_none():
    target = (datetime.now() + timedelta(days=1)).date()
    respx.get(OPENWEATHERMAP_FORECAST_URL).mock(side_effect=httpx.ConnectError("boom"))
    async with httpx.AsyncClient(verify=False) as client:
        summary = await get_weather_forecast(client, "Moscow", "key", target)
    assert summary is None
