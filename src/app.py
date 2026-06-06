"""Application composition root: wires shared deps and runs forever."""

import asyncio

import httpx
import zeroconf
from loguru import logger

from src.audio_server import AudioServer
from src.esphome_client import DeviceManager
from src.settings import settings
from src.tts import make_tts_backend


async def main() -> None:
    """Build shared dependencies, start all speakers, and run until cancelled."""
    # client_ext (proxied) -> Groq chat/STT + weather; client_local -> smart-home + TTS.
    client_ext = httpx.AsyncClient(proxy=(settings.groq_proxy or None), verify=False)
    client_local = httpx.AsyncClient(verify=False)

    tts_backend = make_tts_backend(
        settings.tts_backend, settings.tts_base_url, client_local, settings.tts_timeout
    )
    audio_server = AudioServer(settings.audio_host, settings.audio_port, settings.audio_ttl)
    await audio_server.start()

    zc = zeroconf.Zeroconf()
    manager = DeviceManager(zc, client_ext, client_local, tts_backend, audio_server)

    try:
        await manager.start()
        await asyncio.Event().wait()  # run forever until cancelled
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        await manager.stop()
        await audio_server.stop()
        await client_ext.aclose()
        await client_local.aclose()
        zc.close()
