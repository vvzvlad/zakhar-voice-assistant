"""Application composition root: wires shared deps and runs forever."""

import asyncio

import httpx
import zeroconf
from loguru import logger

from src.audio_server import AudioServer
from src.esphome_client import DeviceManager
from src.mcp_client import McpToolHub
from src.settings import settings
from src.stt import make_stt_backend
from src.tts import make_tts_backend


async def main() -> None:
    """Build shared dependencies, start all speakers, and run until cancelled."""
    # client_ext (proxied) -> cloud STT/intent + weather; client_local -> local TTS.
    client_ext = httpx.AsyncClient(proxy=(settings.external_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)

    # STT runs over client_ext (the proxied client GroqSttBackend uses).
    stt_backend = make_stt_backend(settings.stt_provider, client_ext, settings)
    # Yandex SpeechKit is a cloud API -> route it through the proxied client (like
    # STT/intent). Local TTS backends (teratts/piper) use the direct client.
    tts_client = client_ext if settings.tts_backend == "yandex" else client_local
    tts_backend = make_tts_backend(
        settings.tts_backend, settings.tts_base_url, tts_client, settings.tts_timeout
    )
    audio_server = AudioServer(settings.audio_host, settings.audio_port, settings.audio_ttl)
    await audio_server.start()

    # Smart-home MCP client: connect once here (same task as stop(), per anyio
    # cancel-scope rules). A connect failure is handled gracefully inside start().
    hub = McpToolHub(settings.mcp_smarthome_url, settings.mcp_smarthome_token or None)
    await hub.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(
        zc, client_ext, hub, stt_backend, tts_backend, audio_server
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
