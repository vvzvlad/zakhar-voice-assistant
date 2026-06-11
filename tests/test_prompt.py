from src.core_config import (
    CalendarConfig,
    CoreConfig,
    McpServerConfig,
    OpenWeatherMapConfig,
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
    # active profile's body is included. Weather is no longer injected (it is a tool now).
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
    )

    out = build_system_prompt(core, store)

    # Both built-in sources' non-empty prompts are appended.
    assert "WEATHER-BLOCK" in out
    assert "CAL-BLOCK" in out
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
