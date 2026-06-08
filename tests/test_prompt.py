from src.core_config import CoreConfig, McpServerConfig, PromptConfig
from src.prompt import build_system_prompt, load_system_prompt


def test_load_system_prompt_reads_existing(tmp_path):
    path = tmp_path / "system_prompt.md"
    path.write_text("hello prompt", encoding="utf-8")
    assert load_system_prompt(str(path)) == "hello prompt"


def test_load_system_prompt_creates_from_default(tmp_path):
    path = tmp_path / "nested" / "system_prompt.md"
    content = load_system_prompt(str(path))
    # The data file is created (copied from templates/default_prompt.md) and non-empty.
    assert path.exists()
    assert content == path.read_text(encoding="utf-8")
    assert content


def test_build_system_prompt_replaces_marker_and_includes_body(tmp_path):
    path = tmp_path / "system_prompt.md"
    path.write_text("BODY <<<<<TDW>>>>>", encoding="utf-8")
    core = CoreConfig(prompt=PromptConfig(system_prompt_path=str(path)))

    out = build_system_prompt(core)

    # The <<<<<TDW>>>>> marker is replaced in-place with the time/date prefix; the
    # prompt file body is included. Weather is no longer injected (it is a tool now).
    assert "<<<<<TDW>>>>>" not in out
    assert out.startswith("BODY ")
    assert "Сейчас (дата и время):" in out
    assert "Погода" not in out


def test_build_system_prompt_appends_mcp_server_prompts(tmp_path):
    path = tmp_path / "system_prompt.md"
    path.write_text("BODY <<<<<TDW>>>>>", encoding="utf-8")
    core = CoreConfig(
        prompt=PromptConfig(system_prompt_path=str(path)),
        mcp_servers=[
            McpServerConfig(name="home", url="http://ha/sse", prompt="Controls the lights."),
            McpServerConfig(name="empty", url="http://other", prompt=""),
        ],
    )

    out = build_system_prompt(core)

    # The configured server's non-empty prompt is appended; the empty one contributes
    # nothing, so there is no trailing blank block from it.
    assert "Controls the lights." in out
    assert out.rstrip().endswith("Controls the lights.")
