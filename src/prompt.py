"""System prompt assembly."""

from datetime import datetime

from src.core_config import CoreConfig


def build_system_prompt(core: CoreConfig, prompt_store) -> str:
    """Prefix the active profile's text with the current time-of-day and date.

    The prompt body is the active profile of `prompt_store` (src.prompt_store
    .PromptStore over data/prompts.db). Weather DATA is an on-demand MCP tool (see
    src.builtin_mcp.openweathermap), so the prompt makes no network call and is
    sync. What we optionally append here is each tool source's descriptive
    `prompt` block — external MCP servers plus the built-in weather/calendar
    sources — same as for external MCP servers.
    """
    now = datetime.now()
    date_time_text = now.strftime("%Y-%m-%d, %H:%M")  # 2025-09-18, 14:05
    week_day = now.strftime("%A")  # Tuesday
    day_time = now.strftime("%p")
    prefix = f"Сейчас (дата и время): {date_time_text}, {day_time}, {week_day}.\n"

    system_prompt = prompt_store.active_text()
    system_prompt = system_prompt.replace("<<<<<TDW>>>>>", prefix)

    # Append each tool source's non-empty prompt so the model learns what those
    # tools do (one block per source, blank-line separated): external MCP servers
    # first, then the built-in weather/calendar sources. Disabled sources are
    # skipped — their tools are not in the ToolHub, so the model must not learn them.
    extra = [s.prompt.strip() for s in core.mcp_servers if s.enabled and s.prompt.strip()]
    for builtin in (core.openweathermap, core.calendar):
        if builtin.enabled and builtin.prompt.strip():
            extra.append(builtin.prompt.strip())
    if extra:
        system_prompt = system_prompt + "\n\n" + "\n\n".join(extra)

    return system_prompt
