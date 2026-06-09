"""Current weather and date forecast via OpenWeatherMap."""

from datetime import date, datetime

import httpx
from loguru import logger

OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


async def get_weather_summary(
    client: httpx.AsyncClient, city_name: str, api_key: str
) -> str | None:
    """Return short current weather summary via OpenWeatherMap.

    The injected client carries the proxy/verify configuration (the original
    routed this call through the optional proxy); None is returned on any
    transport or parse error.
    """
    try:
        params = {"q": city_name, "appid": api_key, "units": "metric", "lang": "ru"}
        response = await client.get(OPENWEATHERMAP_URL, params=params, timeout=8)
        logger.info(f"OpenWeatherMap response status for city '{city_name}': {response.status_code}")
        if response.status_code != 200:
            logger.error(f"OpenWeatherMap error for city '{city_name}': {response.status_code} - {response.text}")
            return None

        data = response.json()
        temp = data.get("main", {}).get("temp")
        wind_speed = data.get("wind", {}).get("speed")
        description = None
        weather_list = data.get("weather")
        if isinstance(weather_list, list) and weather_list:
            description = weather_list[0].get("description")

        parts = []
        if isinstance(temp, (int, float)):
            parts.append(f"{int(round(float(temp)))} градусов")
        if description:
            parts.append(description)
        if isinstance(wind_speed, (int, float)):
            wind_raw_str = f"{int(wind_speed)}".replace(".0", "")
            parts.append(f"ветер {wind_raw_str} метров в секунду")

        return ", ".join(parts) if parts else None
    except httpx.HTTPError as req_err:
        logger.error(f"OpenWeatherMap request error for city '{city_name}': {str(req_err)}")
        return None
    except (ValueError, TypeError) as parse_err:
        logger.error(f"OpenWeatherMap parse error for city '{city_name}': {str(parse_err)}")
        return None


async def get_weather_forecast(
    client: httpx.AsyncClient, city_name: str, api_key: str, target_date: date
) -> str | None:
    """Return a short Russian weather summary for `target_date` via the OWM
    5 day / 3 hour forecast API, or None on transport/parse failure.

    `target_date` is a datetime.date. Forecast timestamps (`dt`, unix epoch seconds) are
    converted to LOCAL time with datetime.fromtimestamp so the day boundaries
    match the local "today" the system prompt shows the model — keeping phrases
    like "завтра" consistent. Only the free /forecast endpoint is used (same key
    as current weather), so coverage is ~5 days; a date outside that window
    returns an explanatory string (NOT None), since the request itself succeeded.
    """
    try:
        params = {"q": city_name, "appid": api_key, "units": "metric", "lang": "ru"}
        response = await client.get(OPENWEATHERMAP_FORECAST_URL, params=params, timeout=8)
        logger.info(f"OpenWeatherMap forecast response status for city '{city_name}': {response.status_code}")
        if response.status_code != 200:
            logger.error(f"OpenWeatherMap forecast error for city '{city_name}': {response.status_code} - {response.text}")
            return None

        data = response.json()
        forecast_list = data.get("list")
        if not isinstance(forecast_list, list) or not forecast_list:
            return None

        # Keep forecast entries that fall on the requested local day, remembering
        # each entry's local hour so we can pick a midday-representative description.
        matched = []
        for item in forecast_list:
            dt = item.get("dt")
            if not isinstance(dt, (int, float)):
                continue
            local_dt = datetime.fromtimestamp(dt)
            if local_dt.date() == target_date:
                matched.append((local_dt.hour, item))

        if not matched:
            # The request succeeded but the date is past or beyond the ~5 day window.
            return (
                f"Прогноз на {target_date.isoformat()} недоступен: "
                f"прогноз есть только на ближайшие 5 дней."
            )

        temps = []
        winds = []
        pops = []
        for _hour, item in matched:
            temp = item.get("main", {}).get("temp")
            if isinstance(temp, (int, float)):
                temps.append(float(temp))
            wind_speed = item.get("wind", {}).get("speed")
            if isinstance(wind_speed, (int, float)):
                winds.append(float(wind_speed))
            pop = item.get("pop")
            if isinstance(pop, (int, float)):
                pops.append(float(pop))

        # Representative description: the entry whose local hour is closest to midday
        # (13:00); ties resolve to the earliest entry in matched order.
        description = None
        best_distance = None
        for hour, item in matched:
            weather_list = item.get("weather")
            if not (isinstance(weather_list, list) and weather_list):
                continue
            candidate = weather_list[0].get("description")
            if not candidate:
                continue
            distance = abs(hour - 13)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                description = candidate

        parts = []
        if temps:
            min_t = int(round(min(temps)))
            max_t = int(round(max(temps)))
            if min_t == max_t:
                parts.append(f"{min_t} градусов")
            else:
                parts.append(f"от {min_t} до {max_t} градусов")
        if description:
            parts.append(description)
        if pops:
            max_pop = max(pops)
            if max_pop >= 0.1:
                parts.append(f"вероятность осадков {int(round(max_pop * 100))} процентов")
        if winds:
            max_wind = max(winds)
            parts.append(f"ветер до {int(round(max_wind))} метров в секунду")

        if not parts:
            return None

        return f"Прогноз на {target_date.isoformat()}: " + ", ".join(parts)
    except httpx.HTTPError as req_err:
        logger.error(f"OpenWeatherMap forecast request error for city '{city_name}': {str(req_err)}")
        return None
    except (ValueError, TypeError) as parse_err:
        logger.error(f"OpenWeatherMap forecast parse error for city '{city_name}': {str(parse_err)}")
        return None
