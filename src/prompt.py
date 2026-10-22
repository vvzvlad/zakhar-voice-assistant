"""System prompt assembly."""

from datetime import datetime

from src.core_config import CoreConfig
from src.voice_marker import strip_voice_marker


def build_system_prompt(core: CoreConfig, prompt_store, wake_word: str | None = None,
                        memory: list[str] | None = None) -> str:
    """The system layer (core.prompt.system), then the active profile's text, then the
    tool sources' prompts.

    The profile is the active one of `prompt_store` (src.prompt_store.PromptStore over
    data/prompts.db). In both the system layer and the profile <<<<<TDW>>>>> becomes the
    current date and time, <<<<<WAKE_WORD>>>>> becomes `wake_word` (empty when it is
    unknown) and <<<<<MEMORY>>>>> becomes the `memory` facts (src.memory.MemoryStore),
    one «- fact» line each, or «(пусто)» when there are none. An empty system layer
    leaves the profile as the prompt's start. Weather DATA is an on-demand MCP tool
    (see src.builtin_mcp.openweathermap), so the prompt makes no network call and is sync. What we optionally append here is each tool
    source's descriptive `prompt` block — external MCP servers plus the built-in
    weather/calendar sources — same as for external MCP servers.
    """
    now = datetime.now()
    date_time_text = now.strftime("%Y-%m-%d, %H:%M")  # 2025-09-18, 14:05
    week_day = now.strftime("%A")  # Tuesday
    day_time = now.strftime("%p")
    prefix = f"Сейчас (дата и время): {date_time_text}, {day_time}, {week_day}.\n"
    memory_text = "\n".join(f"- {fact}" for fact in memory) if memory else "(пусто)"

    def expand(text: str) -> str:
        return (text.replace("<<<<<TDW>>>>>", prefix)
                .replace("<<<<<WAKE_WORD>>>>>", wake_word or "")
                .replace("<<<<<MEMORY>>>>>", memory_text))

    system_prompt = expand(prompt_store.active_text())
    # Drop the per-profile preferred-voice marker so it never reaches the model.
    system_prompt = strip_voice_marker(system_prompt)
    system = expand(core.prompt.system).strip()
    if system:
        system_prompt = system + "\n\n" + system_prompt

    # Append each tool source's non-empty prompt so the model learns what those
    # tools do (one block per source, blank-line separated): external MCP servers
    # first, then the built-in weather/calendar/assistant-tasks sources. Disabled sources are
    # skipped — their tools are not in the ToolHub, so the model must not learn them.
    extra = [s.prompt.strip() for s in core.mcp_servers if s.enabled and s.prompt.strip()]
    for builtin in (core.openweathermap, core.calendar, core.assistant_tasks):
        if builtin.enabled and builtin.prompt.strip():
            extra.append(builtin.prompt.strip())
    if extra:
        system_prompt = system_prompt + "\n\n" + "\n\n".join(extra)

    return system_prompt
