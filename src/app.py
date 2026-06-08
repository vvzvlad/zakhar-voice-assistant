"""Application composition root: boots from data/config.json and runs forever."""

import asyncio
import json

import httpx
import zeroconf
from loguru import logger

import src.plugins  # noqa: F401  triggers provider registration
from src import config_store
from src.audio_server import AudioServer
from src.config_service import ConfigDoc, ConfigService
from src.esphome_client import DeviceManager
from src.mcp_client import McpToolHub
from src.plugins.base import Deps


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
    max_tool_rounds = svc.get("llm").max_tool_rounds

    audio_server = AudioServer(core.audio.host, core.audio.port, core.audio.ttl)
    await audio_server.start()

    # Smart-home MCP client: connect once here (same task as stop(), per anyio
    # cancel-scope rules). A connect failure is handled gracefully inside start().
    hub = McpToolHub(core.mcp.url, core.mcp.token or None)
    await hub.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(
        zc, hub, stt_backend, llm_backend, tts_backend, audio_server,
        client_ext, core, max_tool_rounds,
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
