"""ESPHome Native API client per speaker, plus a manager for all speakers.

We are the CLIENT side: we connect TO each speaker (a TCP server on :6053),
drive its voice_assistant, and route events through a per-device Pipeline.
"""

import asyncio

from aioesphomeapi import APIClient, ReconnectLogic
from loguru import logger

from src.audio_server import tts_url
from src.core_config import DeviceConfig
from src.pipeline import CAPTURE_MAX_SECONDS, Pipeline, build_ack_clip

# Extra wall-clock margin on top of the requested capture seconds when waiting for
# the recorded WAV: covers the press -> voice_assistant.start round-trip plus the
# device's own self-stop, which both run inside the requested window on the device.
CAPTURE_WAIT_MARGIN = 8.0

# Native API object_ids of the manual-capture template entities. The firmware
# transmits object_id = slugify(name) over the API (NOT the YAML `id:` field), so
# these MUST equal slugify(name) from esphome/zakhar-voice.yaml — i.e. the entities
# named "Capture Seconds" / "Capture Sample".
CAPTURE_SECONDS_OBJECT_ID = "capture_seconds"
CAPTURE_SAMPLE_OBJECT_ID = "capture_sample"


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
        # Early-filler announcements (see Pipeline._deliver_filler) use the same
        # assist-satellite announce path as DeviceClient.announce(): it ducks current
        # audio and plays while the run is still working. Rebound on every (re)connect.
        self.pipeline.send_announcement = self.cli.send_voice_assistant_announcement_await_response

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

    async def capture(self, seconds: int) -> bytes:
        """Record `seconds` of mic audio on this speaker and RETURN it as WAV bytes.

        Arms the pipeline for a capture-only run FIRST (so the flag is set before the
        device's voice_assistant.start arrives), sets the device-side duration, then
        presses the device button. The device streams audio for `seconds` and stops
        itself; the pipeline (capture-only, no STT/LLM/TTS) resolves the armed Future
        with the recorded WAV bytes, which we await and return — the capture is
        EPHEMERAL, nothing is written to the server. Raises a clear error when the
        speaker is offline, lacks the capture entities, or the recording times out.
        Raises CaptureBusyError (from arm_capture) when a capture is already armed /
        in-flight on this device, so concurrent captures are rejected rather than
        racing each other's Futures.
        """
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        if self._capture_button_key is None or self._capture_seconds_key is None:
            raise RuntimeError(
                f"{self.cfg.name} has no manual-capture entities "
                f"(firmware needs the capture_sample/seconds template entities)"
            )
        # Defensive clamp to the supported range (the panel API validates 1..MAX, but
        # this guards any other caller). The device-side template number caps at the
        # same CAPTURE_MAX_SECONDS, and the wait_for timeout below scales with seconds.
        seconds = max(1, min(int(seconds), CAPTURE_MAX_SECONDS))
        # Arm BEFORE pressing so the resulting on_start is treated as capture-only.
        # arm_capture hands back the Future the capture run resolves with WAV bytes.
        future = self.pipeline.arm_capture(seconds)
        logger.info(f"{self.cfg.name}: ⏺️ manual capture {seconds}s")
        # number_command / button_command are sync (they just queue a protobuf send).
        self.cli.number_command(self._capture_seconds_key, float(seconds))
        self.cli.button_command(self._capture_button_key)
        try:
            return await asyncio.wait_for(future, timeout=seconds + CAPTURE_WAIT_MARGIN)
        except asyncio.TimeoutError:
            # The recording never arrived (lost press / device never streamed). Clear
            # the armed state so a later run isn't hijacked, and surface a clear error.
            self.pipeline.disarm_capture()
            raise TimeoutError(
                f"{self.cfg.name} capture timed out after {seconds + CAPTURE_WAIT_MARGIN:.0f}s"
            )

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

    async def play_media(self, audio: bytes, mime: str) -> None:
        """Play a ready audio clip on this speaker via the assist-satellite announce path.

        Mirrors announce() but takes pre-built audio bytes instead of synthesizing text —
        used by the panel's chime preview. Ducks any current audio and plays while idle.
        """
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        audio_id = self.pipeline.audio_server.put(audio, mime)
        _ext, url = tts_url(self.pipeline.public_base_url, audio_id, mime)
        logger.info(f"{self.cfg.name}: 🔔 play media -> {url}")
        await self.cli.send_voice_assistant_announcement_await_response(
            media_id=url, timeout=30.0, text="",
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

    async def capture(self, device_name: str, seconds: int) -> bytes:
        """Trigger a manual capture-only recording on the named speaker, return WAV bytes.

        Routes to the matching online client (mirrors announce routing) and returns
        the recorded audio as WAV bytes (the capture is ephemeral — nothing is kept
        on the server). Raises a clear error when the device is unknown or offline so
        the caller (panel API) can return the right status code.
        """
        target = next(
            (c for c in self.clients if c.cfg.name == device_name), None
        )
        if target is None:
            raise LookupError(f"unknown device {device_name!r}")
        if not target.online:
            raise RuntimeError(f"{device_name} is offline")
        return await target.capture(seconds)

    async def play_chime(self, sound_path: str, device_name: str | None = None) -> dict:
        """Play the given end-of-phrase chime on the speaker(s) for an operator preview.

        Builds the clip ONCE off the event loop (build_ack_clip does file IO / a WAV
        transcode), then plays it via the announce path on the named device, or on EVERY
        online device when device_name is None. Offline targets (and any per-device
        failure) are reported, never raised. Returns {"played": [names], "offline": [names]}.
        Raises LookupError for an unknown named device so the API can return 404.
        """
        mime, audio = await asyncio.to_thread(build_ack_clip, sound_path)
        if device_name is not None:
            target = next((c for c in self.clients if c.cfg.name == device_name), None)
            if target is None:
                raise LookupError(f"unknown device {device_name!r}")
            targets = [target]
        else:
            targets = list(self.clients)
        played: list[str] = []
        offline: list[str] = []
        for c in targets:
            if not c.online:
                offline.append(c.cfg.name)
                continue
            try:
                await c.play_media(audio, mime)
                played.append(c.cfg.name)
            except Exception as e:
                logger.warning(f"{c.cfg.name}: chime preview failed: {e}")
                offline.append(c.cfg.name)
        return {"played": played, "offline": offline}

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
