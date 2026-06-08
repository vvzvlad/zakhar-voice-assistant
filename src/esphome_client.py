"""ESPHome Native API client per speaker, plus a manager for all speakers.

We are the CLIENT side: we connect TO each speaker (a TCP server on :6053),
drive its voice_assistant, and route events through a per-device Pipeline.
"""

from aioesphomeapi import APIClient, ReconnectLogic
from loguru import logger

from src.audio_server import tts_url
from src.core_config import DeviceConfig
from src.pipeline import Pipeline

# Native API object_ids of the manual-capture template entities. The firmware
# transmits object_id = slugify(name) over the API (NOT the YAML `id:` field), so
# these MUST equal slugify(name) from esphome/zakhar-voice.yaml — i.e. the entities
# named "Zakhar Capture Seconds" / "Zakhar Capture Sample".
CAPTURE_SECONDS_OBJECT_ID = "zakhar_capture_seconds"
CAPTURE_SAMPLE_OBJECT_ID = "zakhar_capture_sample"


class DeviceClient:
    """One ESPHome speaker: connection lifecycle + voice_assistant wiring."""

    def __init__(self, cfg: DeviceConfig, zc, runtime):
        self.cfg = cfg
        self.rt = runtime
        self.pipeline = Pipeline(cfg.name, runtime)
        self.cli = APIClient(
            cfg.host, runtime.core.esphome.port, None,
            noise_psk=cfg.psk, zeroconf_instance=zc,
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
        # Native API entity keys for the manual-capture template entities, discovered
        # on connect by object_id. None when the firmware predates these entities.
        self._capture_button_key = None
        self._capture_seconds_key = None

    async def _on_connect(self) -> None:
        """Re-runs on every (re)connection: log device, discover entities, wire & subscribe."""
        # Combined call: device info + the entity list in one round-trip. We need the
        # entity list to map the capture template entities (by object_id) to their
        # Native API keys for number_command/button_command.
        try:
            info, entities, _services = await self.cli.device_info_and_list_entities()
            logger.info(
                f"connected {self.cfg.name}: {info.name} (esphome {info.esphome_version})"
            )
            self._discover_capture_keys(entities)
        except Exception as e:
            logger.warning(f"{self.cfg.name}: device_info/list_entities failed: {e}")

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

    def _discover_capture_keys(self, entities) -> None:
        """Map the manual-capture template entities to Native API keys by object_id.

        Sets _capture_button_key / _capture_seconds_key from the entity list; leaves
        them None when the firmware does not expose the entities (older flash).
        """
        self._capture_button_key = None
        self._capture_seconds_key = None
        for ent in entities:
            object_id = getattr(ent, "object_id", None)
            if object_id == CAPTURE_SAMPLE_OBJECT_ID:
                self._capture_button_key = ent.key
            elif object_id == CAPTURE_SECONDS_OBJECT_ID:
                self._capture_seconds_key = ent.key
        if self._capture_button_key is None or self._capture_seconds_key is None:
            logger.info(
                f"{self.cfg.name}: manual-capture entities not found "
                f"(button={self._capture_button_key}, seconds={self._capture_seconds_key}); "
                f"flash the firmware with the capture entities to enable it"
            )

    async def capture(self, seconds: int) -> None:
        """Record `seconds` of mic audio on this speaker in capture-only mode.

        Arms the pipeline for a capture-only run FIRST (so the flag is set before the
        device's voice_assistant.start arrives), sets the device-side duration, then
        presses the device button. The device then streams audio for `seconds` and
        stops itself; the pipeline writes the PCM to a WAV (no STT/LLM/TTS). Raises a
        clear error when the speaker is offline or lacks the capture entities.
        """
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        if self._capture_button_key is None or self._capture_seconds_key is None:
            raise RuntimeError(
                f"{self.cfg.name} has no manual-capture entities "
                f"(firmware needs the zakhar_capture_sample/seconds template entities)"
            )
        # Arm BEFORE pressing so the resulting on_start is treated as capture-only.
        self.pipeline.arm_capture(seconds)
        logger.info(f"{self.cfg.name}: ⏺️ manual capture {seconds}s")
        # number_command / button_command are sync (they just queue a protobuf send).
        self.cli.number_command(self._capture_seconds_key, float(seconds))
        self.cli.button_command(self._capture_button_key)

    async def announce(self, text: str) -> None:
        """Proactively speak `text` on this speaker via the assist-satellite announce path."""
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        # Synthesize at fire time so the audio-cache TTL never matters (URL is fresh).
        mime, audio = await self.pipeline.tts_backend.synthesize(text, "ru")
        audio_id = self.pipeline.audio_server.put(audio, mime)
        _ext, url = tts_url(self.pipeline.public_base_url, audio_id, mime)
        logger.info(f"{self.cfg.name}: 🔔 announce: {text!r} -> {url}")
        # Assist-satellite announce ducks any current audio and plays while idle.
        await self.cli.send_voice_assistant_announcement_await_response(
            media_id=url, timeout=30.0, text=text,
        )

    async def start(self) -> None:
        await self.reconnect.start()

    async def stop(self) -> None:
        try:
            await self.reconnect.stop()
        except Exception as e:
            logger.warning(f"{self.cfg.name}: reconnect.stop failed: {e}")
        try:
            await self.cli.disconnect(force=True)
        except Exception:
            pass


class DeviceManager:
    """Owns one DeviceClient per configured speaker; starts/stops them all."""

    def __init__(self, zc, runtime):
        self.zc = zc
        self.rt = runtime
        # The esphome port is global (not per-device); track the value the live
        # clients were built with so reconfigure() can detect a port change.
        self._esphome_port = runtime.core.esphome.port
        self.clients = [
            DeviceClient(cfg, zc, runtime) for cfg in runtime.core.devices
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

    async def capture(self, device_name: str, seconds: int) -> None:
        """Trigger a manual capture-only recording on the named speaker.

        Routes to the matching online client (mirrors announce routing). Raises a
        clear error when the device is unknown or offline so the caller (panel API)
        can return the right status code.
        """
        target = next(
            (c for c in self.clients if c.cfg.name == device_name), None
        )
        if target is None:
            raise LookupError(f"unknown device {device_name!r}")
        if not target.online:
            raise RuntimeError(f"{device_name} is offline")
        await target.capture(seconds)

    async def start(self) -> None:
        for c in self.clients:
            try:
                await c.start()
            except Exception as e:
                logger.error(f"failed to start device client {c.cfg.name}: {e}")
        logger.info(f"started {len(self.clients)} device client(s)")

    async def stop(self) -> None:
        for c in self.clients:
            try:
                await c.stop()
            except Exception as e:
                logger.error(f"failed to stop device client {c.cfg.name}: {e}")

    async def reconfigure(self) -> None:
        """Reconcile running device clients with the current config (hot).

        A device is keyed by (name, host, psk); changed keys are stopped+recreated.
        A global esphome.port change rebuilds every client (the port is not per-device).
        `self.clients` is mutated in place so the panel/scheduler keep their bound
        `statuses`/`announce` methods (we never replace the manager object)."""
        core = self.rt.core
        port = core.esphome.port
        if port != self._esphome_port:
            await self.stop()
            self._esphome_port = port
            self.clients = [DeviceClient(cfg, self.zc, self.rt) for cfg in core.devices]
            await self.start()
            return
        desired = {(d.name, d.host, d.psk): d for d in core.devices}
        current = {(c.cfg.name, c.cfg.host, c.cfg.psk): c for c in self.clients}
        for key, client in list(current.items()):
            if key not in desired:
                await client.stop()
                self.clients.remove(client)
        for key, cfg in desired.items():
            if key not in current:
                client = DeviceClient(cfg, self.zc, self.rt)
                self.clients.append(client)
                await client.start()
