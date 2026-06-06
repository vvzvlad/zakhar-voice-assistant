"""System prompt loading and assembly."""

import os
from datetime import datetime

import httpx
from loguru import logger

from src.settings import settings
from src.weather import get_weather_summary

DEFAULT_PROMPT_PATH = "templates/default_prompt.md"


def load_system_prompt() -> str:
    """Load system prompt from settings.system_prompt_path or create it from the default.

    If the data file is missing, copies default content into the data file and returns it.
    """
    prompt_path = settings.system_prompt_path

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"System prompt loaded from {prompt_path}")
            return content

    # Fallback: read default and create data file
    with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as df:
        default_content = df.read()

    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as pf:
        pf.write(default_content)
    logger.info(f"System prompt file created at {prompt_path} from {DEFAULT_PROMPT_PATH}")
    return default_content


async def build_system_prompt(client: httpx.AsyncClient) -> str:
    """Prefix SYSTEM_PROMPT with current time-of-day, date, and current weather."""
    now = datetime.now()
    date_time_text = now.strftime("%Y-%m-%d, %H:%M")  # 2025-09-18, 14:05
    week_day = now.strftime("%A")  # Tuesday
    day_time = now.strftime("%p")
    prefix = f"Сейчас (дата и время): {date_time_text}, {day_time}, {week_day}.\n"

    weather_summary = await get_weather_summary(
        client, settings.weather_city, settings.weather_api_key
    )
    if weather_summary is not None:
        prefix += f"Погода в {settings.weather_city}: {weather_summary}.\n"

    system_prompt = load_system_prompt()
    system_prompt = system_prompt.replace("<<<<<TDW>>>>>", prefix)

    return system_prompt
