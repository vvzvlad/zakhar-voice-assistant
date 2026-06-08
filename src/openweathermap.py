"""Current weather summary via OpenWeatherMap."""

import httpx
from loguru import logger

OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"


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
