"""System prompt loading and assembly."""

import os
from datetime import datetime

from loguru import logger

from src.core_config import CoreConfig

DEFAULT_PROMPT_PATH = "templates/default_prompt.md"


def load_system_prompt(prompt_path: str) -> str:
    """Load the system prompt from `prompt_path` or create it from the default.

    If the data file is missing, copies default content into the data file and returns it.
    """
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


def save_system_prompt(prompt_path: str, text: str) -> None:
    """Write `text` to `prompt_path` (utf-8), creating the parent dir if needed."""
    parent = os.path.dirname(prompt_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"System prompt saved to {prompt_path}")


def build_system_prompt(core: CoreConfig) -> str:
    """Prefix SYSTEM_PROMPT with the current time-of-day and date.

    Weather used to be injected here; it is now an on-demand MCP tool (see
    src.builtin_mcp.weather), so the prompt no longer makes a network call and is sync.
    """
    now = datetime.now()
    date_time_text = now.strftime("%Y-%m-%d, %H:%M")  # 2025-09-18, 14:05
    week_day = now.strftime("%A")  # Tuesday
    day_time = now.strftime("%p")
    prefix = f"Сейчас (дата и время): {date_time_text}, {day_time}, {week_day}.\n"

    system_prompt = load_system_prompt(core.prompt.system_prompt_path)
    system_prompt = system_prompt.replace("<<<<<TDW>>>>>", prefix)

    # Append each external MCP server's non-empty prompt so the model learns what
    # those servers' tools do (one block per server, blank-line separated).
    extra = [s.prompt.strip() for s in core.mcp_servers if s.prompt.strip()]
    if extra:
        system_prompt = system_prompt + "\n\n" + "\n\n".join(extra)

    return system_prompt
