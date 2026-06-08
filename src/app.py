"""Application composition root: boots from data/config.json and runs forever."""

import asyncio
import json

import httpx
import zeroconf
from loguru import logger

import src.plugins  # noqa: F401  triggers provider registration
from src import config_store
from src.audio_server import AudioServer
from src.builtin_mcp.weather import build_weather_server
from src.config_service import ConfigDoc, ConfigService
from src.esphome_client import DeviceManager
from src.mcp_client import McpToolHub
from src.plugins.base import Deps
from src.tool_hub import BuiltinMcpSource, HttpMcpSource, ToolHub
from src.version import __version__


def load_or_create_config() -> dict:
    """Return the config document, creating data/config.json from the template on first boot."""
    doc = config_store.load()
    if not doc:
        with open("templates/default_config.json", encoding="utf-8") as f:
            doc = json.load(f)
        config_store.save(doc, config_store.DEFAULT_PATH)
        logger.info("created default config at data/config.json")
    return doc


async def main() -> None:
    """Build shared dependencies, start all speakers, and run until cancelled."""
    doc = load_or_create_config()
    core = ConfigDoc(**doc).core  # parse once to read proxy/timeout for Deps

    # client_ext (proxied) -> cloud STT/LLM + weather; client_local -> local TTS.
    client_ext = httpx.AsyncClient(proxy=(core.network.external_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)
    deps = Deps(http_cloud=client_ext, http_local=client_local, tts_timeout=core.tts_timeout)

    svc = ConfigService(doc, deps)
    core = svc.core
    stt_backend = svc.create("stt")
    tts_backend = svc.create("tts")
    llm_backend = svc.create("llm")
    llm_cfg = svc.get("llm")

    audio_server = AudioServer(core.audio.host, core.audio.port, core.audio.ttl)
    await audio_server.start()

    # Multi-source tool hub: the external smart-home MCP server (HttpMcpSource) plus
    # in-process built-in MCP servers (weather first). Built only here, in the same
    # task as stop(), per anyio cancel-scope rules. A source failing to start is
    # handled gracefully inside ToolHub.start(). The weather tool is gated on an OWM
    # api key; built-in weather uses the proxied client_ext for its OWM call.
    sources = []
    if core.mcp.url:
        sources.append(HttpMcpSource("home", McpToolHub(core.mcp.url, core.mcp.token or None)))
    if core.weather.api_key:
        sources.append(
            BuiltinMcpSource(
                "weather", build_weather_server(client_ext, core.weather.api_key, core.weather.city)
            )
        )
    hub = ToolHub(sources)
    await hub.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(
        zc, hub, stt_backend, llm_backend, tts_backend, audio_server,
        client_ext, core, llm_cfg,
    )

    try:
        await manager.start()
        await asyncio.Event().wait()  # run forever until cancelled
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        await manager.stop()
        await hub.stop()
        await audio_server.stop()
        await client_ext.aclose()
        await client_local.aclose()
        zc.close()
