from src.core_config import (
    AssistantTasksConfig,
    CalendarConfig,
    CoreConfig,
    McpServerConfig,
    OpenWeatherMapConfig,
    PromptConfig,
)
from src.prompt import build_system_prompt
from src.prompt_store import PromptStore


def _store(tmp_path, body):
    """A real PromptStore on tmp_path seeded from a legacy file holding `body`."""
    legacy = tmp_path / "system_prompt.md"
    legacy.write_text(body, encoding="utf-8")
    return PromptStore(str(tmp_path / "prompts.db"), seed_path=str(legacy))


def test_build_system_prompt_replaces_marker_and_includes_body(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    core = CoreConfig()

    out = build_system_prompt(core, store)

    # The <<<<<TDW>>>>> marker is replaced in-place with the time/date prefix; the
    # active profile's body is included. Weather is not injected (it is a tool).
    assert "<<<<<TDW>>>>>" not in out
    assert out.startswith("BODY ")
    assert "Сейчас (дата и время):" in out
    assert "Погода" not in out
    store.close()


def test_build_system_prompt_appends_mcp_server_prompts(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    core = CoreConfig(
        mcp_servers=[
            McpServerConfig(name="home", url="http://ha/sse", prompt="Controls the lights."),
            McpServerConfig(name="empty", url="http://other", prompt=""),
        ],
    )

    out = build_system_prompt(core, store)

    # The configured server's non-empty prompt is appended; the empty one contributes
    # nothing, so there is no trailing blank block from it.
    assert "Controls the lights." in out
    assert out.rstrip().endswith("Controls the lights.")
    store.close()


def test_build_system_prompt_appends_builtin_weather_and_calendar_prompts(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    core = CoreConfig(
        openweathermap=OpenWeatherMapConfig(prompt="WEATHER-BLOCK"),
        calendar=CalendarConfig(prompt="CAL-BLOCK"),
        assistant_tasks=AssistantTasksConfig(prompt="TASKS-BLOCK"),
    )

    out = build_system_prompt(core, store)

    # Every built-in source's non-empty prompt is appended.
    assert "WEATHER-BLOCK" in out
    assert "CAL-BLOCK" in out
    assert "TASKS-BLOCK" in out
    store.close()


def test_build_system_prompt_skips_empty_builtin_prompts(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    # Weather has a prompt, calendar is empty: the empty one contributes no block, so the
    # output ends on the weather block with no trailing blank block from the calendar.
    core = CoreConfig(
        openweathermap=OpenWeatherMapConfig(prompt="WEATHER-BLOCK"),
        calendar=CalendarConfig(prompt=""),
    )

    out = build_system_prompt(core, store)

    assert "WEATHER-BLOCK" in out
    assert out.rstrip().endswith("WEATHER-BLOCK")
    store.close()


def test_build_system_prompt_skips_disabled_mcp_server_prompts(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    # A disabled server's prompt block is dropped (its tools are not in the
    # ToolHub); the enabled server's block is still appended.
    core = CoreConfig(
        mcp_servers=[
            McpServerConfig(name="off", url="http://off", prompt="OFF-BLOCK", enabled=False),
            McpServerConfig(name="on", url="http://on", prompt="ON-BLOCK"),
        ],
    )

    out = build_system_prompt(core, store)

    assert "OFF-BLOCK" not in out
    assert "ON-BLOCK" in out
    store.close()


def test_build_system_prompt_skips_disabled_builtin_prompts(tmp_path):
    store = _store(tmp_path, "BODY <<<<<TDW>>>>>")
    # Disabled built-in sources contribute no prompt block even when set.
    core = CoreConfig(
        openweathermap=OpenWeatherMapConfig(prompt="WEATHER-BLOCK", enabled=False),
        calendar=CalendarConfig(prompt="CAL-BLOCK", enabled=False),
    )

    out = build_system_prompt(core, store)

    assert "WEATHER-BLOCK" not in out
    assert "CAL-BLOCK" not in out
    store.close()


def test_build_system_prompt_strips_voice_marker(tmp_path):
    # The per-profile preferred-voice marker is metadata for the panel only and
    # must NEVER reach the model: build_system_prompt strips it while keeping the
    # rest of the body and the TDW substitution intact.
    store = _store(
        tmp_path,
        "BODY <<<<<TDW>>>>>\n<<<<<VOICE provider=yandex voice=zahar>>>>>\nTAIL",
    )
    core = CoreConfig()

    out = build_system_prompt(core, store)

    assert "<<<<<VOICE" not in out
    assert "provider=yandex" not in out
    assert out.startswith("BODY ")
    assert "Сейчас (дата и время):" in out
    assert "TAIL" in out
    store.close()


def test_build_system_prompt_uses_active_profile_not_seed_file(tmp_path):
    # The prompt body comes from the ACTIVE store profile, not from the legacy
    # seed file: after switching profiles the new text is what gets assembled.
    store = _store(tmp_path, "OLD BODY")
    other = store.create("other", "NEW BODY")
    store.activate(other["id"])

    out = build_system_prompt(CoreConfig(), store)

    assert "NEW BODY" in out
    assert "OLD BODY" not in out
    store.close()


def test_build_system_prompt_puts_the_system_layer_before_the_profile(tmp_path):
    # The system layer opens the prompt, a blank line before the profile; both get the
    # date/time and the wake word substituted.
    store = _store(tmp_path, "PROFILE <<<<<WAKE_WORD>>>>> <<<<<TDW>>>>>")
    core = CoreConfig(prompt=PromptConfig(system="SYSTEM «<<<<<WAKE_WORD>>>>>» <<<<<TDW>>>>>\n"))

    out = build_system_prompt(core, store, "Захар")

    system, profile = out.split("\n\n", 1)
    assert system.startswith("SYSTEM «Захар» Сейчас (дата и время):")
    assert profile.startswith("PROFILE Захар Сейчас (дата и время):")
    assert "<<<<<" not in out
    store.close()


def test_build_system_prompt_an_unknown_wake_word_is_empty(tmp_path):
    store = _store(tmp_path, "PROFILE «<<<<<WAKE_WORD>>>>>»")
    core = CoreConfig(prompt=PromptConfig(system="SYSTEM «<<<<<WAKE_WORD>>>>>»"))

    assert build_system_prompt(core, store) == "SYSTEM «»\n\nPROFILE «»"
    store.close()


def test_build_system_prompt_lists_the_memory_facts(tmp_path):
    # In both texts <<<<<MEMORY>>>>> becomes one «- fact» line per saved fact.
    store = _store(tmp_path, "PROFILE\n<<<<<MEMORY>>>>>")
    core = CoreConfig(prompt=PromptConfig(system="SYSTEM\n<<<<<MEMORY>>>>>"))

    out = build_system_prompt(core, store, memory=["Пользователь — мужчина", "Кота зовут Барсик"])

    facts = "- Пользователь — мужчина\n- Кота зовут Барсик"
    assert out == f"SYSTEM\n{facts}\n\nPROFILE\n{facts}"
    store.close()


def test_build_system_prompt_no_memory_reads_empty(tmp_path):
    store = _store(tmp_path, "PROFILE <<<<<MEMORY>>>>>")
    core = CoreConfig()

    assert build_system_prompt(core, store, memory=[]) == "PROFILE (пусто)"
    assert build_system_prompt(core, store) == "PROFILE (пусто)"
    store.close()
