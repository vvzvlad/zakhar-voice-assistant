"""ESPHome Native API client per speaker, plus a manager for all speakers.

We are the CLIENT side: we connect TO each speaker (a TCP server on :6053),
drive its voice_assistant, and route events through a per-device Pipeline.
"""

from aioesphomeapi import APIClient, ReconnectLogic
from loguru import logger

from src.pipeline import Pipeline
from src.settings import DeviceConfig, settings


class DeviceClient:
    """One ESPHome speaker: connection lifecycle + voice_assistant wiring."""

    def __init__(self, cfg: DeviceConfig, zc, client_ext, client_local, tts_backend, audio_server):
        self.cfg = cfg
        self.pipeline = Pipeline(
            cfg.name,
            client_ext,
            client_local,
            tts_backend,
            audio_server,
            settings.public_base_url,
            settings.context_dir,
        )
        self.cli = APIClient(
            cfg.host, settings.esphome_port, None, noise_psk=cfg.psk, zeroconf_instance=zc
        )
        self.reconnect = ReconnectLogic(
            client=self.cli,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            zeroconf_instance=zc,
            name=cfg.name,
        )
        self._unsub = None

    async def _on_connect(self) -> None:
        """Re-runs on every (re)connection: log device, wire & subscribe."""
        try:
            info = await self.cli.device_info()
            logger.info(
                f"connected {self.cfg.name}: {info.name} (esphome {info.esphome_version})"
            )
        except Exception as e:
            logger.warning(f"{self.cfg.name}: device_info failed: {e}")

        # Bind the pipeline's emitters to this live connection.
        self.pipeline.send_event = self.cli.send_voice_assistant_event
        self.pipeline.send_audio = self.cli.send_voice_assistant_audio

        self._unsub = self.cli.subscribe_voice_assistant(
            handle_start=self._handle_start,
            handle_stop=self._handle_stop,
            handle_audio=self._handle_audio,
        )
        logger.info(f"subscribed voice_assistant for {self.cfg.name}")

    async def _on_disconnect(self, expected: bool) -> None:
        logger.info(f"disconnected {self.cfg.name} (expected={expected})")
        # Subscription is re-created on the next on_connect.
        self._unsub = None

    async def _handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        return await self.pipeline.on_start(
            conversation_id, flags, audio_settings, wake_word_phrase
        )

    async def _handle_audio(self, data, data2=None):
        await self.pipeline.on_audio(data, data2)

    async def _handle_stop(self, abort):
        await self.pipeline.on_stop(abort)

    async def start(self) -> None:
        await self.reconnect.start()

    async def stop(self) -> None:
        await self.reconnect.stop()
        try:
            await self.cli.disconnect(force=True)
        except Exception:
            pass


class DeviceManager:
    """Owns one DeviceClient per configured speaker; starts/stops them all."""

    def __init__(self, zc, client_ext, client_local, tts_backend, audio_server):
        self.clients = [
            DeviceClient(cfg, zc, client_ext, client_local, tts_backend, audio_server)
            for cfg in settings.devices
        ]

    async def start(self) -> None:
        for c in self.clients:
            await c.start()
        logger.info(f"started {len(self.clients)} device client(s)")

    async def stop(self) -> None:
        for c in self.clients:
            await c.stop()
