"""Built-in speaker-volume MCP server (in-process FastMCP).

Lets the model read and change the volume when the user says e.g. "какая
громкость", "сделай потише" or "громкость на 30". The tools act only on the
speaker the phrase came from: the speaker is read from the per-run ContextVar
(never an LLM-visible argument). The DeviceManager is resolved at call time
through `get_manager`, because the tool source is built before the manager
exists. The tools catch their exceptions and return a human-readable Russian
string (they never raise), mirroring the reminders tool contract.
"""

from mcp.server.fastmcp import FastMCP

from src.run_context import current_device


def build_volume_server(get_manager) -> FastMCP:
    """Build a FastMCP server exposing the volume tools (get_volume, set_volume);
    `get_manager` is a zero-arg callable returning the live DeviceManager."""
    mcp = FastMCP("volume")

    @mcp.tool(
        name="get_volume",
        description=(
            "Узнать текущую громкость колонки, с которой говорит пользователь, в процентах."
        ),
    )
    async def get_volume() -> str:
        device = current_device.get()
        if device is None:
            return "Не знаю, с какой колонки пришла фраза."
        try:
            volume = get_manager().find(device).get_volume()
        except Exception as e:
            return f"Не удалось узнать громкость: {e}"
        return f"Громкость колонки: {volume}%."

    @mcp.tool(
        name="set_volume",
        description=(
            "Изменить громкость колонки, с которой говорит пользователь, — громкость её "
            "собственного голоса. Громкость телевизора, музыки и других устройств этим "
            "инструментом не меняется. level — новая "
            "громкость в процентах 0–100 («громкость на 30» → level=30, «на максимум» → "
            "100). change — изменить на столько процентов: плюс громче, минус тише; "
            "«погромче» / «потише» без числа — change=10 / change=-10. Укажи ровно одно "
            "из двух."
        ),
    )
    async def set_volume(level: int | None = None, change: int | None = None) -> str:
        device = current_device.get()
        if device is None:
            return "Не знаю, с какой колонки пришла фраза, — громкость не изменена."
        try:
            volume = get_manager().find(device).set_volume(level=level, change=change)
        except Exception as e:
            return f"Не удалось изменить громкость: {e}"
        return f"Громкость колонки: {volume}%."

    return mcp
