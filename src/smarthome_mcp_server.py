"""Thin smart-home MCP server (FastMCP, Streamable HTTP).

This is a standalone process. It exposes smart-home control TOOLS over MCP and
forwards each tool call to the EXISTING Node-RED endpoint (so Node-RED flows stay
unchanged). It must be runnable WITHOUT the app's heavy Settings, so it defines
its own tiny config and does NOT import src.settings (which would require
GROQ_API_KEY etc.).

The Node-RED payload shape is preserved EXACTLY as the old homemade protocol used:
    {"command": {"device_id": ..., "value": ...}}

Design note: the @mcp.tool()-decorated functions are thin wrappers over the
unit-testable helpers (validate_* / normalize_* / dispatch), so tests can exercise
the validation and normalization logic directly without an MCP round-trip.
"""

import httpx
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_errors import load_settings_or_exit


class McpServerSettings(BaseSettings):
    """Self-contained config for the MCP server (independent of the app Settings)."""

    smarthome_url: str                   # Node-RED endpoint (required, no default)
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8201

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


cfg = load_settings_or_exit(McpServerSettings)


# --- Device catalog (the new source of truth for IDs) ------------------------
# Each entry: id -> (room, human description). Drives list_devices() and the
# per-tool validation (which IDs each tool may target).
LIGHTS = {
    "bright_room_light": ("Зал", "Яркий свет"),
    "low_room_light": ("Зал", "Слабый свет"),
    "table_light": ("Зал", "Лампа на столе"),
    "kitchen_light": ("Кухня", "Лампа"),
    "toilet_light": ("Туалет", "Лампа"),
    "bathroom_light": ("Ванная", "Лампа"),
    "corridor_light": ("Коридор", "Лампа"),
}
DIMMERS = {
    "night_light": ("Зал", "Ночник (0-100)"),
}
CLIMATE = {
    "room_ac": ("Зал", "Кондиционер"),
}
SWITCHES = {
    "monitors": ("Зал", "Мониторы"),
}
LOCKS = {
    "main_lock": ("Коридор", "Замок"),
}

# Assistant's physical location (mirrors the system prompt).
ASSISTANT_ROOM = "Зал"


async def dispatch(device_id: str, value: str) -> str:
    """Forward a normalized command to Node-RED. Returns a short status string for the model."""
    payload = {"command": {"device_id": device_id, "value": value}}
    async with httpx.AsyncClient(verify=False) as client:
        try:
            resp = await client.post(
                cfg.smarthome_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            return f"ok ({resp.status_code}) {device_id}={value}"
        except httpx.HTTPError as e:
            return f"error sending {device_id}={value}: {e}"


# --- Normalization / validation helpers (unit-testable, no MCP needed) -------
async def light_command(device_id: str, state: str) -> str:
    """Validate + normalize an on/off light command, then dispatch."""
    if device_id not in LIGHTS:
        return f"error: unknown light device_id {device_id!r}"
    norm = state.strip().lower()
    if norm not in ("on", "off"):
        return f"error: state must be 'on' or 'off', got {state!r}"
    return await dispatch(device_id, norm)


async def dimmer_command(device_id: str, brightness: int) -> str:
    """Validate + clamp a 0..100 brightness command, then dispatch."""
    if device_id not in DIMMERS:
        return f"error: unknown dimmer device_id {device_id!r}"
    try:
        value = int(brightness)
    except (TypeError, ValueError):
        return f"error: brightness must be an integer, got {brightness!r}"
    value = max(0, min(100, value))  # clamp to 0..100
    return await dispatch(device_id, str(value))


async def climate_command(device_id: str, temperature: str) -> str:
    """Validate + normalize a climate command (a number like '22' or literal 'off'), then dispatch."""
    if device_id not in CLIMATE:
        return f"error: unknown climate device_id {device_id!r}"
    norm = str(temperature).strip().lower()
    if norm == "off":
        return await dispatch(device_id, "off")
    try:
        int(norm)
    except ValueError:
        return f"error: temperature must be a number or 'off', got {temperature!r}"
    return await dispatch(device_id, norm)


async def switch_command(device_id: str, state: str) -> str:
    """Validate + normalize an on/off switch command (not a light), then dispatch."""
    if device_id not in SWITCHES:
        return f"error: unknown switch device_id {device_id!r}"
    norm = state.strip().lower()
    if norm not in ("on", "off"):
        return f"error: state must be 'on' or 'off', got {state!r}"
    return await dispatch(device_id, norm)


async def lock_command(device_id: str, action: str) -> str:
    """Validate + normalize a lock/unlock command, then dispatch."""
    if device_id not in LOCKS:
        return f"error: unknown lock device_id {device_id!r}"
    norm = action.strip().lower()
    if norm not in ("lock", "unlock"):
        return f"error: action must be 'lock' or 'unlock', got {action!r}"
    return await dispatch(device_id, norm)


def render_catalog() -> str:
    """Human-readable catalog (rooms -> devices -> IDs) for the model."""
    # Build rooms -> list of "description (kind): id" lines.
    rooms: dict[str, list[str]] = {}
    for group, kind in (
        (LIGHTS, "light on/off"),
        (DIMMERS, "dimmer 0-100"),
        (CLIMATE, "climate temp/off"),
        (SWITCHES, "switch on/off"),
        (LOCKS, "lock/unlock"),
    ):
        for device_id, (room, desc) in group.items():
            rooms.setdefault(room, []).append(f"  - {desc} [{kind}] id={device_id}")
    lines = [f"Assistant is located in: {ASSISTANT_ROOM}.", "Devices by room:"]
    for room in rooms:
        lines.append(f"{room}:")
        lines.extend(rooms[room])
    return "\n".join(lines)


# --- FastMCP instance + tools (module scope; host/port read from cfg) --------
mcp = FastMCP(
    "smarthome",
    host=cfg.mcp_server_host,
    port=cfg.mcp_server_port,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def set_light(device_id: str, state: str) -> str:
    """Turn a light on or off. state is "on" or "off".

    Allowed device_id: bright_room_light (Зал, яркий свет), low_room_light
    (Зал, слабый свет), table_light (Зал, лампа на столе), kitchen_light (Кухня),
    toilet_light (Туалет), bathroom_light (Ванная), corridor_light (Коридор).
    """
    return await light_command(device_id, state)


@mcp.tool()
async def set_dimmer(device_id: str, brightness: int) -> str:
    """Set a dimmable light brightness (0..100, clamped). 0 = off, 100 = max.

    Allowed device_id: night_light (Зал, ночник).
    """
    return await dimmer_command(device_id, brightness)


@mcp.tool()
async def set_climate(device_id: str, temperature: str) -> str:
    """Set climate target temperature (a number like "22") or turn it off ("off").

    Allowed device_id: room_ac (Зал, кондиционер).
    """
    return await climate_command(device_id, temperature)


@mcp.tool()
async def set_switch(device_id: str, state: str) -> str:
    """Turn a switch on or off (not a light). state is "on" or "off".

    Allowed device_id: monitors (Зал, мониторы).
    """
    return await switch_command(device_id, state)


@mcp.tool()
async def set_lock(device_id: str, action: str) -> str:
    """Lock or unlock a lock. action is "lock" or "unlock".

    Allowed device_id: main_lock (Коридор).
    """
    return await lock_command(device_id, action)


@mcp.tool()
def list_devices() -> str:
    """Return the smart-home catalog (rooms -> devices -> IDs) so you can map
    natural-language requests to the correct device_id for the other tools."""
    return render_catalog()


def main() -> None:
    """Run the MCP server over Streamable HTTP (serves at /mcp)."""
    logger.info(
        f"Starting smart-home MCP server on {cfg.mcp_server_host}:{cfg.mcp_server_port} "
        f"-> Node-RED {cfg.smarthome_url}"
    )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
