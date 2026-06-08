"""ESPHome Native API client per speaker, plus a manager for all speakers.

We are the CLIENT side: we connect TO each speaker (a TCP server on :6053),
drive its voice_assistant, and route events through a per-device Pipeline.
"""

from aioesphomeapi import APIClient, ReconnectLogic
from loguru import logger

from src.core_config import DeviceConfig
from src.pipeline import Pipeline


class DeviceClient:
    """One ESPHome speaker: connection lifecycle + voice_assistant wiring."""

    def __init__(
        self,
        cfg: DeviceConfig,
        zc,
        hub,
        stt_backend,
        llm_backend,
        tts_backend,
        audio_server,
        core,
        llm_cfg,
        runs_store=None,
    ):
        self.cfg = cfg
        self.pipeline = Pipeline(
            cfg.name,
            hub,
            stt_backend,
            llm_backend,
            tts_backend,
            audio_server,
            core,
            llm_cfg,
            runs_store=runs_store,
        )
        self.cli = APIClient(
            cfg.host, core.esphome.port, None, noise_psk=cfg.psk, zeroconf_instance=zc
        )
        self.reconnect = ReconnectLogic(
            client=self.cli,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            zeroconf_instance=zc,
            name=cfg.name,
        )
        self._unsub = None
        self.online = False

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
        self.online = True

    async def _on_disconnect(self, expected: bool) -> None:
        logger.info(f"disconnected {self.cfg.name} (expected={expected})")
        # Subscription is re-created on the next on_connect.
        self._unsub = None
        self.online = False

    async def _handle_start(self, conversation_id, flags, audio_settings, wake_word_phrase):
        return await self.pipeline.on_start(
            conversation_id, flags, audio_settings, wake_word_phrase
        )

    async def _handle_audio(self, data, data2=None):
        await self.pipeline.on_audio(data, data2)

    async def _handle_stop(self, abort):
        await self.pipeline.on_stop(abort)

    async def announce(self, text: str) -> None:
        """Proactively speak `text` on this speaker via the assist-satellite announce path."""
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        # Synthesize at fire time so the audio-cache TTL never matters (URL is fresh).
        mime, audio = await self.pipeline.tts_backend.synthesize(text, "ru")
        audio_id = self.pipeline.audio_server.put(audio, mime)
        ext = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac"}.get(mime, "mp3")
        url = f"{self.pipeline.public_base_url.rstrip('/')}/tts/{audio_id}.{ext}"
        logger.info(f"{self.cfg.name}: 🔔 announce: {text!r} -> {url}")
        # Assist-satellite announce ducks any current audio and plays while idle.
        await self.cli.send_voice_assistant_announcement_await_response(
            media_id=url, timeout=30.0, text=text,
        )

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

    def __init__(
        self,
        zc,
        hub,
        stt_backend,
        llm_backend,
        tts_backend,
        audio_server,
        core,
        llm_cfg,
        runs_store=None,
    ):
        self.clients = [
            DeviceClient(
                cfg,
                zc,
                hub,
                stt_backend,
                llm_backend,
                tts_backend,
                audio_server,
                core,
                llm_cfg,
                runs_store=runs_store,
            )
            for cfg in core.devices
        ]

    def statuses(self) -> list[dict]:
        """Live connection status for every configured speaker (for the panel API)."""
        return [
            {"name": c.cfg.name, "host": c.cfg.host, "online": c.online}
            for c in self.clients
        ]

    async def announce(self, device_name: str | None, text: str) -> None:
        """Route a reminder to its originating speaker; drop if unavailable."""
        target = None
        if device_name is not None:
            target = next((c for c in self.clients if c.cfg.name == device_name), None)
        else:
            # No device recorded (shouldn't happen via the pipeline): use the first online one.
            target = next((c for c in self.clients if c.online), None)
        if target is None or not target.online:
            logger.warning(f"reminder target {device_name!r} unavailable; dropping")
            return
        await target.announce(text)

    async def start(self) -> None:
        for c in self.clients:
            await c.start()
        logger.info(f"started {len(self.clients)} device client(s)")

    async def stop(self) -> None:
        for c in self.clients:
            await c.stop()
