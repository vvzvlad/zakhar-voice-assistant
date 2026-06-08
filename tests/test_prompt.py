import httpx
import respx

from src.core_config import CoreConfig, PromptConfig, WeatherConfig
from src.prompt import build_system_prompt, load_system_prompt
from src.weather import OPENWEATHERMAP_URL


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


@respx.mock
async def test_build_system_prompt_includes_weather_and_prefix(tmp_path):
    path = tmp_path / "system_prompt.md"
    path.write_text("BODY <<<<<TDW>>>>>", encoding="utf-8")
    core = CoreConfig(
        prompt=PromptConfig(system_prompt_path=str(path)),
        weather=WeatherConfig(api_key="k", city="Moscow"),
    )
    respx.get(OPENWEATHERMAP_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "main": {"temp": 5.0},
                "wind": {"speed": 3},
                "weather": [{"description": "ясно"}],
            },
        )
    )

    async with httpx.AsyncClient() as client:
        out = await build_system_prompt(client, core)

    # The <<<<<TDW>>>>> marker is replaced in-place with the time/date/weather prefix.
    assert "<<<<<TDW>>>>>" not in out
    assert out.startswith("BODY ")
    assert "Сейчас (дата и время):" in out
    assert "Погода в Moscow:" in out


@respx.mock
async def test_build_system_prompt_omits_weather_on_error(tmp_path):
    path = tmp_path / "system_prompt.md"
    path.write_text("BODY <<<<<TDW>>>>>", encoding="utf-8")
    core = CoreConfig(
        prompt=PromptConfig(system_prompt_path=str(path)),
        weather=WeatherConfig(api_key="k", city="Moscow"),
    )
    respx.get(OPENWEATHERMAP_URL).mock(return_value=httpx.Response(404, json={}))

    async with httpx.AsyncClient() as client:
        out = await build_system_prompt(client, core)

    assert "Погода" not in out
    assert out.startswith("BODY ")
    assert "Сейчас (дата и время):" in out
