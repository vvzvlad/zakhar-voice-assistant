"""Device server: the WebSocket endpoint voice devices connect to (docs/device-protocol.md).

One aiohttp app on core.audio.host/port serves GET /device (the device protocol) and
GET /health. DeviceManager keeps one Device per name; a Device keeps its Pipeline
across reconnects and binds the pipeline's transport hooks (send_event, play,
has_speaker) to its live Connection. This module alone maps StageEvents onto wire
messages and converts outgoing audio to what the device plays.
"""

import asyncio
import io
import itertools
import json
import wave
from collections import deque

import numpy as np
import soxr
from aiohttp import WSCloseCode, WSMsgType, web
from loguru import logger

from src.audio_prep import pcm_to_wav_bytes
from src.chime import build_ack_clip
from src.pipeline import Pipeline
from src.pipeline_events import StageEvent
from src.tts import pcm_rate

# Binary frame kinds (first byte).
MIC = 0x01
PLAY = 0x02
CLOSE_DISABLED = 4000
CLOSE_REPLACED = 4001
HEARTBEAT_S = 10.0
# Largest playback payload per binary frame.
PLAY_FRAME_BYTES = 4096
# The server keeps at most buffer_ms minus this much audio ahead of the device's playback.
PACING_MARGIN_MS = 500
# The wait for `played` gives up after the stream's duration + buffer_ms + this much.
PLAYED_WAIT_MARGIN_S = 5.0

# Maximum duration (seconds) of a manual "record X seconds" capture; the panel API
# validates against it.
CAPTURE_MAX_SECONDS = 300
# Wall-clock margin on top of the requested capture seconds when waiting for audio_end.
CAPTURE_WAIT_MARGIN = 8.0

# Panel control ids.
WAKE_CUTOFF = "wake_probability_cutoff"
VAD_CUTOFF = "vad_cutoff"
SPEAKER_VOLUME = "speaker_volume"
_CUTOFF_FIELDS = {WAKE_CUTOFF: "wake_cutoff", VAD_CUTOFF: "vad_cutoff"}

# StageEvents that map onto a plain `state` message; TTS_START, STT_END and ERROR
# carry text and are handled in Device._send_event, the rest send nothing.
_STATE_FOR_EVENT = {
    StageEvent.STT_VAD_START: "listening",
    StageEvent.STT_VAD_END: "transcribing",
    StageEvent.INTENT_START: "thinking",
    StageEvent.RUN_END: "idle",
}

_STREAM_IDS = itertools.count(1)

# Fire-and-forget tasks (closing a replaced or disabled connection), kept referenced
# until they finish.
_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


class CaptureBusyError(Exception):
    """A capture is already in progress on this device (the panel API answers 409)."""


class CaptureEmptyError(Exception):
    """A capture finished but the device streamed no audio."""


async def _one_chunk(data: bytes):
    yield data


async def _aclose(chunks) -> None:
    """Close a chunk iterator that will not be iterated further (a TTS backend
    releases its open response from the generator's finally)."""
    aclose = getattr(chunks, "aclose", None)
    if aclose is not None:
        await aclose()


class _PcmOut:
    """The delivery boundary for one play() stream: WAV (a whole clip) or raw PCM
    (pcm_mime, in chunks) in, s16le mono at the device's rate out, resampled with a
    streaming soxr resampler."""

    def __init__(self, mime: str, out_rate: int):
        self._wav = mime == "audio/wav"
        self._rate = pcm_rate(mime)
        if not self._wav and self._rate is None:
            raise ValueError(f"unsupported audio mime {mime!r}")
        self._out_rate = out_rate
        self._pending = b""  # WAV: the clip so far; PCM: an odd byte split across chunks
        self._resampler = None

    def feed(self, chunk: bytes) -> bytes:
        data = self._pending + chunk
        if self._wav:
            self._pending = data
            return b""
        n = len(data) - len(data) % 2
        self._pending = data[n:]
        return self._resample(np.frombuffer(data[:n], dtype="<i2"), last=False)

    def flush(self) -> bytes:
        if not self._wav:
            return self._resample(np.zeros(0, dtype="<i2"), last=True)
        with wave.open(io.BytesIO(self._pending), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"expected 16-bit WAV, got {w.getsampwidth() * 8}-bit")
            self._rate, channels = w.getframerate(), w.getnchannels()
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        mono = frames.reshape(-1, channels).mean(axis=1).astype("<i2")
        return self._resample(mono, last=True)

    def _resample(self, samples, last: bool) -> bytes:
        if self._rate == self._out_rate:
            return samples.tobytes()
        if self._resampler is None:
            self._resampler = soxr.ResampleStream(self._rate, self._out_rate, 1, dtype="int16")
        return self._resampler.resample_chunk(samples, last=last).astype("<i2").tobytes()


class _Stream:
    """One play() stream on the wire, from play_start until the device reports it."""

    def __init__(self, session):
        self.id = next(_STREAM_IDS)
        self.session = session
        self.sent_s = 0.0  # seconds of audio sent
        # Resolves to "played", "dropped" or "closed" (the connection went away).
        self.ended = asyncio.get_running_loop().create_future()

    @property
    def live(self) -> bool:
        return not self.ended.done()

    def end(self, how: str) -> None:
        if not self.ended.done():
            self.ended.set_result(how)


class Connection:
    """One live WebSocket of a device, from hello to close.

    Everything goes out through one ordered queue drained by a writer task, so the
    synchronous send_event hook and the paced audio of play() keep their order. Each
    queued item is tagged with its play stream (None for other messages), so a wake
    can purge the audio it flushes on the device."""

    def __init__(self, ws: web.WebSocketResponse, hello: dict):
        self.ws = ws
        self.name = hello["id"]
        self.fw = hello.get("fw")
        self.model = hello.get("model")
        self.speaker = bool(hello.get("caps", {}).get("speaker"))
        self.mic_channels = hello.get("mic", {}).get("channels", 1)
        play = hello.get("play", {})
        self.play_rate = play.get("rate")
        self.buffer_ms = play.get("buffer_ms")
        self.volume = hello.get("volume")
        self.wake_peak = None
        # The device's current session, and whether its run is still going (from
        # wake until the terminal state): plays in that window belong to the session.
        self.session = None
        self.session_open = False
        self.cutoffs = None  # (wake, vad) last sent in `config`
        self.closed = False
        self.capture_buf: bytearray | None = None  # mic audio while a capture runs
        self._capture_fut: asyncio.Future | None = None
        self._out: deque = deque()
        self._out_ready = asyncio.Event()
        self._streams: dict[int, _Stream] = {}
        self._play_lock = asyncio.Lock()
        # Loop time at which the audio sent so far will have played out.
        self._play_until = 0.0
        self._writer = asyncio.create_task(self._write_loop())

    async def _write_loop(self) -> None:
        try:
            while True:
                await self._out_ready.wait()
                self._out_ready.clear()
                while self._out:
                    _stream, data = self._out.popleft()
                    if isinstance(data, str):
                        await self.ws.send_str(data)
                    else:
                        await self.ws.send_bytes(data)
        except ConnectionError as e:
            logger.debug(f"{self.name}: send failed: {e}")

    def _enqueue(self, data, stream=None) -> None:
        if self.closed:
            return
        self._out.append((stream, data))
        self._out_ready.set()

    def send(self, msg: dict, stream: _Stream | None = None) -> None:
        self._enqueue(json.dumps(msg, ensure_ascii=False), stream)

    def send_session(self, msg: dict) -> None:
        """Send a message of the current session, stamped with it."""
        if self.session is None:
            return
        self.send({**msg, "session": self.session})

    def send_config(self, cfg) -> None:
        self.cutoffs = (cfg.wake_cutoff, cfg.vad_cutoff)
        self.send({"type": "config", "wake_cutoff": cfg.wake_cutoff, "vad_cutoff": cfg.vad_cutoff})

    def start_session(self, session) -> None:
        """A wake: the device flushed its playback, so every stream in flight and
        every queued audio frame is dropped here too, and `play_stop` goes out
        before anything else of the new session (it drops on the device a stream
        whose play_start left before the wake was seen)."""
        self.session = session
        self.session_open = True
        self._out = deque(item for item in self._out if item[0] is None)
        for stream in self._streams.values():
            stream.end("dropped")
        self._streams.clear()
        self._play_until = 0.0
        self.send({"type": "play_stop"})

    def on_played(self, stream_id, dropped: bool) -> None:
        stream = self._streams.pop(stream_id, None)
        if stream is not None:
            stream.end("dropped" if dropped else "played")

    def _stream_msg(self, kind: str, stream: _Stream) -> dict:
        msg = {"type": kind, "id": stream.id}
        if stream.session is not None:
            msg["session"] = stream.session
        return msg

    async def play(self, mime: str, audio) -> None:
        """Send one clip (bytes or an async chunk iterator) as a play stream, paced to
        the device's buffer, and return once the device reports it played (or dropped).
        Streams go out back to back in call order. The iterator is always closed."""
        chunks = _one_chunk(audio) if isinstance(audio, (bytes, bytearray)) else audio
        try:
            out = _PcmOut(mime, self.play_rate)
            async with self._play_lock:
                stream = _Stream(self.session if self.session_open else None)
                if self.closed:
                    stream.end("closed")
                else:
                    self._streams[stream.id] = stream
                    self.send(self._stream_msg("play_start", stream), stream)
                try:
                    async for chunk in chunks:
                        await self._send_pcm(stream, out.feed(chunk))
                        if not stream.live:
                            break
                    else:
                        await self._send_pcm(stream, out.flush())
                finally:
                    if stream.live:
                        self.send(self._stream_msg("play_end", stream), stream)
        finally:
            await _aclose(chunks)
        timeout = stream.sent_s + self.buffer_ms / 1000 + PLAYED_WAIT_MARGIN_S
        await asyncio.wait({stream.ended}, timeout=timeout)
        if stream.live:
            logger.warning(f"{self.name}: no `played` for stream {stream.id} in {timeout:.1f}s; "
                           f"taking it as dropped")
            self._streams.pop(stream.id, None)
            stream.end("dropped")
        if stream.ended.result() == "closed":
            raise ConnectionError(f"{self.name} disconnected during playback")

    async def _send_pcm(self, stream: _Stream, pcm: bytes) -> None:
        """Queue PCM frames, keeping at most buffer_ms - PACING_MARGIN_MS of audio
        ahead of the device, whose playback is taken to run at real time from the
        first frame sent."""
        loop = asyncio.get_running_loop()
        ahead = (self.buffer_ms - PACING_MARGIN_MS) / 1000
        for i in range(0, len(pcm), PLAY_FRAME_BYTES):
            frame = pcm[i:i + PLAY_FRAME_BYTES]
            seconds = len(frame) / 2 / self.play_rate
            wait = max(self._play_until, loop.time()) + seconds - loop.time() - ahead
            if wait > 0:
                await asyncio.sleep(wait)
            if not stream.live:
                return
            self._play_until = max(self._play_until, loop.time()) + seconds
            stream.sent_s += seconds
            self._enqueue(bytes([PLAY]) + frame, stream)

    async def capture(self, seconds: int) -> bytes:
        """Record `seconds` of mic audio on the device; return it as a mono WAV."""
        if self._capture_fut is not None:
            raise CaptureBusyError(f"{self.name} capture already in progress")
        self._capture_fut = asyncio.get_running_loop().create_future()
        self.capture_buf = bytearray()
        self.send({"type": "capture", "seconds": seconds})
        try:
            return await asyncio.wait_for(self._capture_fut, seconds + CAPTURE_WAIT_MARGIN)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"{self.name} capture timed out after {seconds + CAPTURE_WAIT_MARGIN:.0f}s"
            )
        finally:
            self._capture_fut = None
            self.capture_buf = None

    def on_audio_end(self, reason) -> None:
        fut, pcm = self._capture_fut, self.capture_buf
        if fut is None or fut.done():
            return
        if reason == "busy":
            fut.set_exception(RuntimeError(f"{self.name} is busy: a voice session is running"))
        elif reason == "capture":
            if pcm:
                fut.set_result(pcm_to_wav_bytes(bytes(pcm)))
            else:
                fut.set_exception(CaptureEmptyError(f"{self.name} capture produced no audio"))

    def close(self, code: int | None = None) -> None:
        """Stop sending, end every pending play and capture, and (with a code) close
        the socket in the background."""
        if self.closed:
            return
        self.closed = True
        self._writer.cancel()
        for stream in self._streams.values():
            stream.end("closed")
        self._streams.clear()
        if self._capture_fut is not None and not self._capture_fut.done():
            self._capture_fut.set_exception(ConnectionError(f"{self.name} disconnected"))
        if code is not None:
            _spawn(self.ws.close(code=code))


class Device:
    """One device name: its Pipeline (kept across reconnects) and its live Connection."""

    def __init__(self, name: str, runtime):
        self.name = name
        self.rt = runtime
        self.conn: Connection | None = None
        self.pipeline = Pipeline(name, runtime)
        self.pipeline.send_event = self._send_event
        self.pipeline.play = self.play

    @property
    def cfg(self):
        return next((d for d in self.rt.core.devices if d.name == self.name), None)

    @property
    def online(self) -> bool:
        return self.conn is not None

    def attach(self, conn: Connection) -> Connection | None:
        """Make `conn` the live connection; return the one it replaces, if any."""
        old, self.conn = self.conn, conn
        self.pipeline.has_speaker = conn.speaker
        return old

    async def detach(self, conn: Connection) -> None:
        conn.close()
        if self.conn is conn:
            self.conn = None
            logger.info(f"{self.name}: disconnected")
            await self.pipeline.on_disconnect()

    def _send_event(self, event, data) -> None:
        """The pipeline's send_event hook: map a StageEvent onto wire messages."""
        conn = self.conn
        if conn is None:
            return
        if event is StageEvent.STT_END:
            if data.get("text"):
                conn.send_session({"type": "transcript", "text": data["text"]})
        elif event is StageEvent.TTS_START:
            conn.send_session({"type": "reply", "text": data.get("text", "")})
            conn.send_session({"type": "state", "value": "speaking"})
        elif event is StageEvent.ERROR:
            conn.send_session({"type": "state", "value": "error", "text": data.get("message", "")})
        elif event is StageEvent.RUN_END and not conn.session_open:
            pass  # the session already ended with `state error`
        elif event in _STATE_FOR_EVENT:
            conn.send_session({"type": "state", "value": _STATE_FOR_EVENT[event]})
        if event in (StageEvent.RUN_END, StageEvent.ERROR):
            conn.session_open = False

    async def play(self, mime: str, audio) -> None:
        """The pipeline's play hook."""
        conn = self.conn
        if conn is None or not conn.speaker:
            if not isinstance(audio, (bytes, bytearray)):
                await _aclose(audio)
            raise RuntimeError(f"{self.name} cannot play: offline or no speaker")
        await conn.play(mime, audio)

    async def on_text(self, conn: Connection, msg: dict) -> None:
        if conn is not self.conn:
            return  # a replaced connection still draining
        kind = msg.get("type")
        if kind == "wake":
            # Synchronously, before any await: the new session, the dropped plays and
            # (at the top of on_start) the cancelled older run.
            conn.start_session(msg["session"])
            await self.pipeline.on_start(str(msg["session"]), msg.get("word"))
        elif kind == "played":
            conn.on_played(msg["id"], bool(msg.get("dropped")))
        elif kind == "volume":
            conn.volume = msg["value"]
        elif kind == "wake_peak":
            conn.wake_peak = msg["value"]
        elif kind == "audio_end":
            conn.on_audio_end(msg.get("reason"))

    async def on_binary(self, conn: Connection, data: bytes) -> None:
        if conn is not self.conn or data[0] != MIC:
            return
        payload = data[1:]
        if conn.mic_channels == 2:
            samples = np.frombuffer(payload, dtype="<i2")
            ch0, ch1 = samples[0::2].tobytes(), samples[1::2].tobytes()
        else:
            ch0, ch1 = payload, None
        if conn.capture_buf is not None and not conn.session_open:
            # A capture records the channel the pipeline would use. Frames of a running
            # session (the device answers the capture with `busy`) stay with the session.
            use_ch1 = self.rt.core.vad.mic_channel == 1 and ch1
            conn.capture_buf.extend(ch1 if use_ch1 else ch0)
            return
        await self.pipeline.on_audio(ch0, ch1)

    def versions(self) -> list[dict]:
        conn = self.conn
        return [
            {"id": "config_version", "name": "Firmware", "value": conn.fw},
            {"id": "model_version", "name": "Model", "value": conn.model},
        ]

    def controls(self) -> list[dict]:
        """Panel controls: the cutoffs (server config) and, on a connected device
        with a speaker, its volume."""
        cfg = self.cfg
        out = [
            {"id": WAKE_CUTOFF, "name": "Wake word cutoff", "value": cfg.wake_cutoff,
             "min": 0, "max": 100, "step": 1, "unit": "%"},
            {"id": VAD_CUTOFF, "name": "VAD cutoff", "value": cfg.vad_cutoff,
             "min": 0, "max": 100, "step": 1, "unit": "%"},
        ]
        conn = self.conn
        if conn is not None and conn.speaker:
            out.append({"id": SPEAKER_VOLUME, "name": "Speaker volume", "value": conn.volume,
                        "min": 0, "max": 100, "step": 1, "unit": "%"})
        return out

    def set_control(self, control_id: str, value: float) -> None:
        """A cutoff is written to the config (the reconfigure that follows sends
        `config` to a connected device); the volume goes to the device.
        Raises LookupError (unknown control / no speaker) or RuntimeError (offline)."""
        pct = max(0, min(100, round(value)))
        field = _CUTOFF_FIELDS.get(control_id)
        if field is not None:
            devices = [d.model_dump(mode="json") for d in self.rt.core.devices]
            for d in devices:
                if d["name"] == self.name:
                    d[field] = pct
            self.rt.svc.apply({"core": {"devices": devices}})
            return
        if control_id != SPEAKER_VOLUME:
            raise LookupError(f"{self.name} has no control {control_id!r}")
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        if not conn.speaker:
            raise LookupError(f"{self.name} has no speaker")
        conn.send({"type": "set_volume", "value": pct})
        conn.volume = pct

    async def announce(self, text: str) -> None:
        """Proactively say `text`: the text to the device as an announcement and, on
        a device with a speaker, the synthesized speech."""
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        conn.send({"type": "reply", "text": text})
        await self.pipeline.speak(text)

    async def capture(self, seconds: int) -> bytes:
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        seconds = max(1, min(int(seconds), CAPTURE_MAX_SECONDS))
        logger.info(f"{self.name}: ⏺️ manual capture {seconds}s")
        return await conn.capture(seconds)


class DeviceManager:
    """The device server: one aiohttp app (/device, /health) and one Device per name."""

    def __init__(self, runtime):
        self.rt = runtime
        self.host = runtime.core.audio.host
        self.port = runtime.core.audio.port
        self._devices: dict[str, Device] = {}
        self._runner: web.AppRunner | None = None

    def _device(self, name: str) -> Device:
        device = self._devices.get(name)
        if device is None:
            device = self._devices[name] = Device(name, self.rt)
        return device

    @property
    def clients(self) -> list[Device]:
        """The enabled configured devices."""
        return [self._device(cfg.name) for cfg in self.rt.core.devices if cfg.enabled]

    def _find(self, device_name: str) -> Device:
        target = next((c for c in self.clients if c.name == device_name), None)
        if target is None:
            raise LookupError(f"unknown device {device_name!r}")
        return target

    def statuses(self) -> list[dict]:
        """Live connection status for every configured device (for the panel API)."""
        out = []
        for cfg in self.rt.core.devices:
            device = self._devices.get(cfg.name)
            online = bool(device and device.online)
            out.append({
                "name": cfg.name,
                "enabled": cfg.enabled,
                "online": online,
                "versions": device.versions() if online else [],
            })
        return out

    async def announce(self, device_name: str | None, text: str) -> None:
        """Route a reminder to its originating device; drop if unavailable."""
        if device_name is not None:
            target = next((c for c in self.clients if c.name == device_name), None)
        else:
            target = next((c for c in self.clients if c.online), None)
        if target is None or not target.online:
            logger.warning(f"reminder target {device_name!r} unavailable; dropping")
            return
        await target.announce(text)

    async def capture(self, device_name: str, seconds: int) -> bytes:
        """Record a manual sample on the named device and return it as WAV bytes."""
        target = self._find(device_name)
        if not target.online:
            raise RuntimeError(f"{device_name} is offline")
        return await target.capture(seconds)

    def device_controls(self, device_name: str) -> dict:
        target = self._find(device_name)
        return {
            "device": device_name,
            "online": target.online,
            "controls": target.controls(),
            "versions": target.versions() if target.online else [],
        }

    def set_device_control(self, device_name: str, control_id: str, value: float) -> dict:
        self._find(device_name).set_control(control_id, value)
        return self.device_controls(device_name)

    def set_wake_prob_stream(self, device_name: str, enabled: bool) -> dict:
        target = self._find(device_name)
        conn = target.conn
        if conn is None:
            raise RuntimeError(f"{device_name} is offline")
        conn.send({"type": "wake_peak_stream", "on": bool(enabled)})
        if not enabled:
            conn.wake_peak = None
        return {"device": device_name, "online": True, "available": True, "value": conn.wake_peak}

    def wake_prob(self, device_name: str) -> dict:
        target = self._find(device_name)
        conn = target.conn
        return {
            "device": device_name,
            "online": conn is not None,
            "available": conn is not None,
            "value": conn.wake_peak if conn is not None else None,
        }

    async def play_chime(self, sound_path: str, device_name: str | None = None) -> dict:
        """Play the given chime on the named device, or on every device, for an
        operator preview. Returns {"played": [names], "offline": [names]}; a device
        that fails to play is reported offline. Raises LookupError for an unknown name."""
        mime, audio = await asyncio.to_thread(build_ack_clip, sound_path)
        targets = [self._find(device_name)] if device_name is not None else self.clients
        played: list[str] = []
        offline: list[str] = []
        for c in targets:
            if not c.online:
                offline.append(c.name)
                continue
            try:
                await c.play(mime, audio)
                played.append(c.name)
            except Exception as e:
                logger.warning(f"{c.name}: chime preview failed: {e}")
                offline.append(c.name)
        return {"played": played, "offline": offline}

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _handle_device(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT_S)
        await ws.prepare(request)
        device = conn = None
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if conn is not None:
                        await device.on_text(conn, data)
                    elif data.get("type") == "hello":
                        device, conn = await self._hello(ws, data)
                        if conn is None:
                            break
                elif msg.type == WSMsgType.BINARY and conn is not None:
                    await device.on_binary(conn, msg.data)
        finally:
            if conn is not None:
                await device.detach(conn)
        return ws

    async def _hello(self, ws, hello: dict):
        """Admit a device: add an unknown id to the config, refuse a disabled one,
        replace an older connection of the same id, then send `config`."""
        name = hello["id"]
        cfg = next((d for d in self.rt.core.devices if d.name == name), None)
        if cfg is None:
            devices = [d.model_dump(mode="json") for d in self.rt.core.devices]
            self.rt.svc.apply({"core": {"devices": [*devices, {"name": name}]}})
            cfg = next(d for d in self.rt.core.devices if d.name == name)
            logger.info(f"{name}: unknown device added to the config")
        if not cfg.enabled:
            logger.info(f"{name}: device is disabled in the config; refusing it")
            await ws.close(code=CLOSE_DISABLED)
            return None, None
        device = self._device(name)
        conn = Connection(ws, hello)
        old = device.attach(conn)
        if old is not None:
            old.close(CLOSE_REPLACED)
            await device.pipeline.on_disconnect()
        conn.send_config(cfg)
        logger.info(
            f"{name}: connected ({conn.fw}, model {conn.model}, "
            f"speaker={conn.speaker}{', replaced the older connection' if old else ''})"
        )
        return device, conn

    async def reconfigure(self) -> None:
        """Apply the device config to live connections: close a device that left the
        config or got disabled, and send `config` to one whose cutoffs changed."""
        names = {cfg.name for cfg in self.rt.core.devices}
        for name, device in self._devices.items():
            if name not in names and device.conn is not None:
                logger.info(f"{name}: removed from the config; closing its connection")
                device.conn.close(WSCloseCode.OK)
        for cfg in self.rt.core.devices:
            device = self._devices.get(cfg.name)
            conn = device.conn if device is not None else None
            if conn is None:
                continue
            if not cfg.enabled:
                logger.info(f"{cfg.name}: disabled; closing its connection")
                conn.close(CLOSE_DISABLED)
            elif conn.cutoffs != (cfg.wake_cutoff, cfg.vad_cutoff):
                conn.send_config(cfg)

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([
            web.get("/device", self._handle_device),
            web.get("/health", self._handle_health),
        ])
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()
        # When bound to an OS-assigned port (0), record the actual port.
        if self.port == 0 and self._runner.addresses:
            self.port = self._runner.addresses[0][1]
        logger.info(f"device server on {self.host}:{self.port}")

    async def stop(self) -> None:
        for device in self._devices.values():
            if device.conn is not None:
                device.conn.close(WSCloseCode.GOING_AWAY)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def rebind(self, host: str, port: int) -> None:
        """Re-bind the listening socket to a new host/port. No-op if unchanged."""
        if (host, port) == (self.host, self.port):
            return
        await self.stop()
        self.host, self.port = host, port
        await self.start()
