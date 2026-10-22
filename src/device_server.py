"""Device server: the WebSocket endpoint voice devices connect to (docs/device-protocol.md).

One aiohttp app on core.audio.host/port serves GET /device (the device protocol) and
GET /health. DeviceManager keeps one Device per name; a Device keeps its Pipeline and its
running capture across reconnects and binds the pipeline's transport hooks (send_event,
play, has_speaker, has_follow_up, take_session, end_wake) to its live Connection. This
module alone maps StageEvents onto wire messages and converts outgoing audio to what the
device plays.
"""

import asyncio
import io
import itertools
import json
import math
import re
import statistics
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import soxr
from aiohttp import WSCloseCode, WSMsgType, web
from loguru import logger

from src import captures, config_store, context, device_logs, telemetry
from src.chime import build_ack_clip
from src.pipeline import Pipeline
from src.pipeline_events import StageEvent
from src.reply_compressor import ReplyCompressor
from src.run_events import record_run
from src.runs_store import new_record
from src.tts import pcm_rate
from src.wake_arbiter import WakeArbiter

# Binary frame kinds (first byte).
MIC = 0x01
PLAY = 0x02
MODEL = 0x03
CAPTURE_AUDIO = 0x04
CLOSE_DISABLED = 4000
CLOSE_REPLACED = 4001
HEARTBEAT_S = 10.0
# Largest payload per binary frame, playback and model alike.
PLAY_FRAME_BYTES = 4096
# The release's default models (the Dockerfile copies every .tflite under microWakeWord/ into
# the image), sent to a device whose core.devices[].wake_model / vad_model is empty: the wake
# one by its name (wake_model_files), the VAD one by its path.
RELEASE_WAKE_MODEL = "zakhar/zakhar_v31"
RELEASE_VAD_MODEL = "microWakeWord/vad/vad.tflite"  # stock: esphome/micro-wake-word-models models/v2/vad.tflite
# Name prefix of the wake-word models uploaded through the panel (uploaded_wake_models_dir).
UPLOADED_WAKE_PREFIX = "uploaded/"
# The server keeps at most buffer_ms minus the play reserve (play_prime_ms) minus this much
# audio ahead of the device's playback.
PACING_MARGIN_MS = 500
# The wait for `played` gives up after the stream's duration + buffer_ms + this much.
PLAYED_WAIT_MARGIN_S = 5.0

# Maximum duration (seconds) of a manual "record X seconds" capture; the panel API
# validates against it.
CAPTURE_MAX_SECONDS = 300
# A capture still running this long past its start + seconds is finished with what came.
CAPTURE_WAIT_MARGIN = 8.0

# Panel control ids.
WAKE_CUTOFF = "wake_probability_cutoff"
VAD_CUTOFF = "vad_cutoff"
PLAY_PRIME = "play_prime_ms"
AUTO_VOLUME_THRESHOLD = "auto_volume_threshold"
AUTO_VOLUME_GAIN = "auto_volume_gain"
SPEAKER_VOLUME = "speaker_volume"
# The controls kept in the device config: control id -> (field, its minimum, its maximum, the
# decimals a value keeps).
_CONFIG_FIELDS = {
    WAKE_CUTOFF: ("wake_cutoff", 0, 100, 0),
    VAD_CUTOFF: ("vad_cutoff", 0, 100, 0),
    PLAY_PRIME: ("play_prime_ms", 0, 2000, 0),
    AUTO_VOLUME_THRESHOLD: ("auto_volume_threshold", -120, 0, 0),
    AUTO_VOLUME_GAIN: ("auto_volume_gain", 0, 10, 1),
}
# The device config's fields `config` carries. The auto volume threshold and gain are not among
# them: the server turns them into `volume_boost` (_boost).
_DEVICE_FIELDS = ("wake_cutoff", "vad_cutoff", "play_prime_ms", "auto_volume")
# The `ambient` values a connection keeps: a minute of them, the device measuring every 15 s.
AMBIENT_KEEP = 4
# A device's boot time (the hello's whole-second uptime counted back from its arrival) moving
# later than this between two of its connections is a reboot, not the jitter of a reconnect.
REBOOT_MARGIN_S = 10
# A `stats` whose largest free block of internal RAM is under this many bytes is logged as a
# warning: the speaker's link needs a 6 KB internal block for its WebSocket task at every connect.
LOW_HEAP_BLOCK = 8192

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


def _config_fields(cfg) -> dict[str, int | bool]:
    """The device config's fields that `config` carries."""
    return {field: getattr(cfg, field) for field in _DEVICE_FIELDS}


def _boost(cfg, ambient) -> int:
    """The volume boost, percent points -100..100: (the median of the `ambient` values - the
    threshold) x the gain, rounded: noise above the threshold raises the volume, below it lowers
    it; 0 before any value and with auto volume off."""
    if not cfg.auto_volume or not ambient:
        return 0
    boost = round((statistics.median(ambient) - cfg.auto_volume_threshold) * cfg.auto_volume_gain)
    return max(-100, min(100, boost))


def _model_paths(cfg) -> dict[str, tuple[str, str]]:
    """The model sent to a device per `model` kind, as (name, path): the configured one, else
    the release's. A wake value that names a model in wake_model_files opens that model's file;
    any other value is the path of the file and its own name."""
    wake = cfg.wake_model or RELEASE_WAKE_MODEL
    vad = cfg.vad_model or RELEASE_VAD_MODEL
    return {"wake": (wake, wake_model_files().get(wake, wake)), "vad": (vad, vad)}


def wake_word_of(cfg) -> str | None:
    """The wake word of the model a device runs: the `wake_word` of the model's manifest,
    the .json with the model file's stem next to it (the ESPHome micro-wake-word format).
    None without a device config or a readable manifest naming the word. Read on every
    call: the configured model changes at runtime."""
    if cfg is None:
        return None
    try:
        path = Path(_model_paths(cfg)["wake"][1]).with_suffix(".json")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    word = manifest.get("wake_word") if isinstance(manifest, dict) else None
    return word if isinstance(word, str) else None


def uploaded_wake_models_dir() -> Path:
    """Where the panel saves uploaded wake-word models: under DATA_DIR, the docker volume, so
    they survive image updates. DATA_DIR is read at call time so tests can monkeypatch it."""
    return Path(config_store.DATA_DIR) / "wake_models"


def wake_model_files() -> dict[str, str]:
    """The wake-word models a device's wake_model can name, {name: path}: every .tflite under
    microWakeWord/ except the VAD model's directory, plus every .tflite directly in
    uploaded_wake_models_dir(); the path is relative to the working directory (where
    send_model opens it). microWakeWord/esphome/<stem>.tflite is esphome/<stem>;
    microWakeWord/v<N>/…/zakhar.tflite is zakhar/zakhar_v<N> and any other <stem>.tflite
    under v<N> is zakhar/zakhar_v<N>_<stem>; an uploaded <stem>.tflite is uploaded/<stem>."""
    files = {}
    for p in Path("microWakeWord").rglob("*.tflite"):
        if p.is_relative_to("microWakeWord/vad"):
            continue
        top = p.parts[1]
        if top == "esphome":
            name = f"esphome/{p.stem}"
        else:
            name = f"zakhar/zakhar_{top}" + ("" if p.stem == "zakhar" else f"_{p.stem}")
        files[name] = p.as_posix()
    # A missing directory (nothing uploaded yet) globs to nothing.
    for p in uploaded_wake_models_dir().glob("*.tflite"):
        files[f"{UPLOADED_WAKE_PREFIX}{p.stem}"] = p.as_posix()
    return files


def list_wake_models() -> list[str]:
    """The names of wake_model_files in natural version order (v8 before v10)."""
    return sorted(wake_model_files(),
                  key=lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)])


class CaptureBusyError(Exception):
    """A capture is already in progress on this device (the panel API answers 409)."""


class NoCaptureError(Exception):
    """No capture is running on this device to stop (the panel API answers 409)."""


class DeviceOfflineError(RuntimeError):
    """The device is not connected (the panel API answers 409, a dataset take waits)."""


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
    streaming soxr resampler and, with `compress`, run through a ReplyCompressor."""

    def __init__(self, mime: str, out_rate: int, compress: bool):
        self._wav = mime == "audio/wav"
        self._rate = pcm_rate(mime)
        if not self._wav and self._rate is None:
            raise ValueError(f"unsupported audio mime {mime!r}")
        self._out_rate = out_rate
        self._pending = b""  # WAV: the clip so far; PCM: an odd byte split across chunks
        self._resampler = None
        self._compressor = ReplyCompressor(out_rate) if compress else None

    def feed(self, chunk: bytes) -> bytes:
        data = self._pending + chunk
        if self._wav:
            self._pending = data
            return b""
        n = len(data) - len(data) % 2
        self._pending = data[n:]
        return self._out(np.frombuffer(data[:n], dtype="<i2"), last=False)

    def flush(self) -> bytes:
        if not self._wav:
            return self._out(np.zeros(0, dtype="<i2"), last=True)
        with wave.open(io.BytesIO(self._pending), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"expected 16-bit WAV, got {w.getsampwidth() * 8}-bit")
            self._rate, channels = w.getframerate(), w.getnchannels()
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        mono = frames.reshape(-1, channels).mean(axis=1).astype("<i2")
        return self._out(mono, last=True)

    def _out(self, samples, last: bool) -> bytes:
        samples = self._resample(samples, last)
        if self._compressor is not None:
            samples = self._compressor.process(samples)
            if last:
                samples = np.concatenate([samples, self._compressor.flush()])
        return samples.astype("<i2").tobytes()

    def _resample(self, samples, last: bool):
        if self._rate == self._out_rate:
            return samples
        if self._resampler is None:
            self._resampler = soxr.ResampleStream(self._rate, self._out_rate, 1, dtype="int16")
        return self._resampler.resample_chunk(samples, last=last)


class _Stream:
    """One play() stream on the wire, from play_start until the device reports it."""

    def __init__(self, session):
        self.id = next(_STREAM_IDS)
        self.session = session
        self.sent_s = 0.0  # seconds of audio sent
        # Resolves to "played", "dropped" or "closed" (the connection went away).
        self.ended = asyncio.get_running_loop().create_future()
        # From `played`: how many times the stream ran dry before its play_end and the ms of silence
        # played meanwhile. None when the device does not report them.
        self.dry = None
        self.gap_ms = None

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
    can purge the audio its `play_stop` flushes on the device."""

    def __init__(self, ws: web.WebSocketResponse, hello: dict):
        self.ws = ws
        self.name = hello["id"]
        self.fw = hello.get("fw")
        self.connected_at = time.time()
        # When the device booted, on the server clock: its hello `uptime` counted back from now.
        self.booted_at = self.connected_at - hello["uptime"]
        # The wake model the device reported loaded (`model_loaded`); None until it does.
        self.model = None
        self.speaker = bool(hello.get("caps", {}).get("speaker"))
        # The device reopens its microphone on a follow-up `listening`.
        self.follow_up = bool(hello.get("caps", {}).get("follow_up"))
        self.mic_channels = hello.get("mic", {}).get("channels", 1)
        play = hello.get("play", {})
        self.play_rate = play.get("rate")
        self.buffer_ms = play.get("buffer_ms")
        self.volume = hello.get("volume")
        self.ambient: deque = deque(maxlen=AMBIENT_KEEP)  # the last `ambient` values, dBFS
        self.boost = 0  # the `volume_boost` last sent; the device starts at 0
        self.wake_peak = None
        # The session the server stamps its messages with, and whether its run is still
        # going (from wake until the terminal state): plays in that window belong to
        # the session. While a wake's run is held before the server takes it (the
        # pipeline's proposal) the device stays on this number and lets the held wake's
        # through too; its `listening` or `state idle` (_end_wake) rules on the wake.
        self.session = None
        self.session_open = False
        self.config_sent = None  # _config_fields() last sent in `config`
        self.log_on = None  # `on` last sent in `log_stream`
        self.models: dict[str, str] = {}  # kind -> the name last sent in `model`
        self.closed = False
        self.held_capture = hello.get("capture")  # the id of the capture the device holds
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
        self.config_sent = _config_fields(cfg)
        self.send({"type": "config", **self.config_sent})

    def send_boost(self, value: int) -> None:
        self.boost = value
        self.send({"type": "volume_boost", "value": value})

    def send_log_stream(self, on: bool) -> None:
        self.log_on = on
        self.send({"type": "log_stream", "on": on})

    def send_model(self, kind: str, name: str, path: str) -> bool:
        """Send the .tflite file at `path` as the device's `kind` model named `name`: `model`,
        then its bytes in 0x03 frames. Queued as no play stream, so a wake does not purge them.
        A file that cannot be read is logged and nothing goes out (False)."""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            logger.error(f"{self.name}: {kind} model not sent: {e}")
            return False
        self.models[kind] = name
        self.send({"type": "model", "kind": kind, "name": name, "size": len(data)})
        for i in range(0, len(data), PLAY_FRAME_BYTES):
            self._enqueue(bytes([MODEL]) + data[i:i + PLAY_FRAME_BYTES])
        return True

    def start_session(self, session) -> None:
        """A wake the server takes: the playback is dropped (drop_playback), with
        `play_stop` before anything else of the new session."""
        self.session = session
        self.session_open = True
        self.drop_playback()

    @property
    def playing(self) -> bool:
        """Whether a play stream is live: from its play_start until the device reports it."""
        return bool(self._streams)

    def drop_playback(self) -> None:
        """Drop the playback (a wake the server takes): every stream in
        flight and every queued audio frame is dropped here, and `play_stop` drops on
        the device what it holds and a stream whose play_start is still on the wire."""
        self._out = deque(item for item in self._out if item[0] is None)
        for stream in self._streams.values():
            stream.end("dropped")
        self._streams.clear()
        self._play_until = 0.0
        self.send({"type": "play_stop"})

    def on_played(self, stream_id, dropped: bool, dry=None, gap_ms=None) -> None:
        stream = self._streams.pop(stream_id, None)
        if stream is not None:
            stream.dry, stream.gap_ms = dry, gap_ms
            stream.end("dropped" if dropped else "played")

    def _stream_msg(self, kind: str, stream: _Stream) -> dict:
        msg = {"type": kind, "id": stream.id}
        if stream.session is not None:
            msg["session"] = stream.session
        return msg

    async def play(self, mime: str, audio, compress: bool) -> dict:
        """Send one clip (bytes or an async chunk iterator) as a play stream, paced to
        the device's buffer, and return once the device reports it. Streams go out back
        to back in call order. The iterator is always closed. With `compress` the audio
        goes through a ReplyCompressor.

        Returns {"outcome", "dry", "gap_ms"}: outcome "played" or "dropped"; dry how
        many times a played stream ran dry before its play_end and gap_ms the ms of
        silence played meanwhile, both None for a dropped stream."""
        chunks = _one_chunk(audio) if isinstance(audio, (bytes, bytearray)) else audio
        try:
            out = _PcmOut(mime, self.play_rate, compress)
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
        outcome = stream.ended.result()
        if outcome != "played":
            return {"outcome": outcome, "dry": None, "gap_ms": None}
        if stream.dry > 0:
            logger.warning(f"{self.name}: stream {stream.id} ran dry {stream.dry} times, "
                           f"{stream.gap_ms} ms of silence")
        return {"outcome": outcome, "dry": stream.dry, "gap_ms": stream.gap_ms}

    async def _send_pcm(self, stream: _Stream, pcm: bytes) -> None:
        """Queue PCM frames, keeping at most buffer_ms - play_prime_ms - PACING_MARGIN_MS
        of audio ahead of the device, whose playback is taken to run at real time from the
        first frame sent. The device starts only once it holds play_prime_ms, so that much
        more stays in its buffer (the last play_prime_ms sent in `config`)."""
        loop = asyncio.get_running_loop()
        ahead = (self.buffer_ms - self.config_sent["play_prime_ms"] - PACING_MARGIN_MS) / 1000
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

    def close(self, code: int | None = None) -> None:
        """Stop sending, end every pending play, and (with a code) close the socket in
        the background. A running capture outlives the connection (Device keeps it)."""
        if self.closed:
            return
        self.closed = True
        self._writer.cancel()
        for stream in self._streams.values():
            stream.end("closed")
        self._streams.clear()
        if code is not None:
            _spawn(self.ws.close(code=code))


def _channels(payload: bytes, channels: int) -> tuple[bytes, bytes | None]:
    """Channel 0 and channel 1 (None for a one-channel mic) of interleaved mic audio."""
    if channels == 2:
        samples = np.frombuffer(payload, dtype="<i2")
        return samples[0::2].tobytes(), samples[1::2].tobytes()
    return payload, None


class Device:
    """One device name: its Pipeline and its running capture (both kept across reconnects)
    and its live Connection."""

    def __init__(self, name: str, runtime):
        self.name = name
        self.rt = runtime
        self.conn: Connection | None = None
        self.pipeline = Pipeline(name, runtime)
        self.pipeline.send_event = self._send_event
        self.pipeline.play = self.play
        self.pipeline.take_session = self._take_session
        self.pipeline.end_wake = self._end_wake
        self.pipeline.wake_word = lambda: wake_word_of(self.cfg)
        self.capture: captures.Capture | None = None
        self._capture_watchdog: asyncio.TimerHandle | None = None
        # The device log could not be written last time: warned once until a write succeeds.
        self._log_write_failed = False
        # The boot time of the device's last connection in this process, None before its first:
        # a later one tells a reboot from a reconnect.
        self.booted_at: float | None = None

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
        self.pipeline.has_follow_up = conn.speaker and conn.follow_up
        return old

    async def detach(self, conn: Connection) -> None:
        conn.close()
        if self.conn is conn:
            self.conn = None
            logger.info(f"{self.name}: disconnected")
            error = conn.ws.exception()
            self.publish_event("disconnected", {
                "code": conn.ws.close_code, "error": str(error) if error is not None else None,
                "connected_s": round(time.time() - conn.connected_at, 1),
            })
            await self.pipeline.on_disconnect()

    def publish_event(self, kind: str, payload: dict) -> None:
        """Publish a device event to the telemetry (src/telemetry.py)."""
        telemetry.publish(self.rt, f"devices/{self.name}/event/{kind}", payload)

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

    async def play(self, mime: str, audio) -> dict:
        """The pipeline's play hook; returns what Connection.play does. The device's
        compress_replies is read per stream, so a change applies from the next one."""
        conn = self.conn
        if conn is None or not conn.speaker:
            if not isinstance(audio, (bytes, bytearray)):
                await _aclose(audio)
            raise RuntimeError(f"{self.name} cannot play: offline or no speaker")
        return await conn.play(mime, audio, self.cfg.compress_replies)

    def _take_session(self, session) -> None:
        """The pipeline's take_session hook: the run of the `wake` that opened `session`
        is taken — the session starts, `play_stop` before anything else of it."""
        if self.conn is not None:
            self.conn.start_session(session)

    def _end_wake(self, session) -> None:
        """The pipeline's end_wake hook: the run of the `wake` that proposed `session`
        ends without being taken (its check rejected it, or it lost the arbitration).
        `state idle` of that number refuses it: the device drops the proposal, whose audio
        stops unless the current session listens for a follow-up, and stays on the
        current session, whose answer plays on."""
        if self.conn is not None:
            self.conn.send({"type": "state", "session": session, "value": "idle"})

    async def on_text(self, conn: Connection, msg: dict) -> None:
        if conn is not self.conn:
            return  # a replaced connection still draining
        kind = msg.get("type")
        if kind == "wake":
            # Over a running answer or an open session the wake may be the speaker's
            # own voice, which the check is told; the answer goes on until the wake is
            # taken (take_session). One over a playing stream only stops it.
            over_answer = conn.session_open or conn.playing
            self.publish_event("wake", {"session": msg["session"], "word": msg.get("word"),
                                        "over_answer": over_answer})
            self.pipeline.on_wake(msg["session"], msg.get("word"), over_answer=over_answer,
                                  over_playback=conn.playing)
        elif kind == "button":
            # The device's button stops what plays (a ring, an answer) as `play_stop` does.
            if conn.playing:
                conn.drop_playback()
        elif kind == "played":
            if msg.get("dropped"):
                self.publish_event("played", {"dropped": True})
                conn.on_played(msg["id"], True)
            else:
                self.publish_event("played", {"dry": msg["dry"], "gap_ms": msg["gap_ms"]})
                conn.on_played(msg["id"], False, msg["dry"], msg["gap_ms"])
        elif kind == "volume":
            self.publish_event("volume", {"value": msg["value"]})
            conn.volume = msg["value"]
        elif kind == "stats":
            telemetry.publish(self.rt, f"devices/{self.name}/stats",
                              {k: v for k, v in msg.items() if k != "type"})
            block = msg.get("heap_block")
            if isinstance(block, int) and block < LOW_HEAP_BLOCK:
                logger.warning(f"{self.name}: largest free internal RAM block {block} B, under "
                               f"{LOW_HEAP_BLOCK} B: the next connect may not get its 6 KB")
        elif kind == "ambient":
            value = msg.get("value")
            # A microphone reading all zeros measures -inf, which arrives as null: not a level.
            level = isinstance(value, (int, float)) and math.isfinite(value)
            self.publish_event("ambient", {"value": value, "zeros": not level})
            if level:
                conn.ambient.append(value)
                self.update_boost()
        elif kind == "wake_peak":
            conn.wake_peak = msg["value"]
        elif kind == "log":
            # A log that cannot be written (a full or read-only volume) drops its lines, not the
            # link: the device would resend its queued lines on every reconnect.
            try:
                device_logs.append(self.name, msg["text"], self.rt.core.device_log_days)
                self._log_write_failed = False
            except OSError as e:
                if not self._log_write_failed:
                    logger.warning(f"{self.name}: device log not written: {e}")
                self._log_write_failed = True
        elif kind == "audio_end":
            if self.capture is not None and msg.get("id") == self.capture.id:
                busy = msg.get("reason") == "busy"
                self.finish_capture(f"{self.name} is busy: a voice session is running" if busy else None)
        elif kind == "model_loaded":
            ok = bool(msg.get("ok"))
            if ok:
                logger.info(f"{self.name}: {msg.get('kind')} model {msg.get('name')} loaded")
            else:
                logger.error(f"{self.name}: {msg.get('kind')} model {msg.get('name')} did not load")
            if msg.get("kind") == "wake":
                # A failed load: the panel shows no model (the device keeps the previous one
                # only when it could not allocate the buffer).
                conn.model = msg.get("name") if ok else None

    async def on_binary(self, conn: Connection, data: bytes) -> None:
        if conn is not self.conn:
            return
        if data[0] == CAPTURE_AUDIO:
            if self.capture is not None:
                # A capture records the channel the pipeline would use.
                ch0, ch1 = _channels(data[5:], conn.mic_channels)
                use_ch1 = self.rt.core.vad.mic_channel == 1 and ch1
                self.capture.write(int.from_bytes(data[1:5], "little"), ch1 if use_ch1 else ch0)
        elif data[0] == MIC:
            await self.pipeline.on_audio(*_channels(data[1:], conn.mic_channels))

    def update_boost(self) -> None:
        """Send `volume_boost` to a connected device whose boost (_boost of its config and its
        `ambient` values) is not the one it has."""
        conn = self.conn
        if conn is None:
            return
        boost = _boost(self.cfg, conn.ambient)
        if boost != conn.boost:
            conn.send_boost(boost)

    def versions(self) -> list[dict]:
        conn = self.conn
        return [
            {"id": "config_version", "name": "Firmware", "value": conn.fw},
            {"id": "model_version", "name": "Model", "value": conn.model},
        ]

    def controls(self) -> list[dict]:
        """Panel controls: the cutoffs, the playback reserve and the auto volume's threshold and
        gain (server config) and, on a connected device with a speaker, its volume."""
        cfg = self.cfg
        out = [
            {"id": WAKE_CUTOFF, "name": "Wake word cutoff", "value": cfg.wake_cutoff,
             "min": 0, "max": 100, "step": 1, "unit": "%"},
            {"id": VAD_CUTOFF, "name": "On-device VAD cutoff", "value": cfg.vad_cutoff,
             "min": 0, "max": 100, "step": 1, "unit": "%"},
            {"id": PLAY_PRIME, "name": "Playback reserve", "value": cfg.play_prime_ms,
             "min": 0, "max": 2000, "step": 50, "unit": "ms"},
            {"id": AUTO_VOLUME_THRESHOLD, "name": "Auto volume threshold",
             "value": cfg.auto_volume_threshold, "min": -120, "max": 0, "step": 1, "unit": "dBFS"},
            {"id": AUTO_VOLUME_GAIN, "name": "Auto volume gain",
             "value": cfg.auto_volume_gain, "min": 0, "max": 10, "step": 0.1, "unit": "%/dB"},
        ]
        conn = self.conn
        if conn is not None and conn.speaker:
            out.append({"id": SPEAKER_VOLUME, "name": "Volume", "value": conn.volume,
                        "min": 0, "max": 100, "step": 1, "unit": "%"})
        return out

    def readings(self) -> list[dict]:
        """Panel readings of the auto volume: the room's noise (the median of the connection's
        `ambient` values, None before the first) and the boost sent (None offline)."""
        conn = self.conn
        ambient = list(conn.ambient) if conn is not None else []
        return [
            {"id": "ambient_noise", "name": "Ambient noise",
             "value": statistics.median(ambient) if ambient else None, "unit": "dBFS"},
            {"id": "volume_boost", "name": "Volume boost",
             "value": conn.boost if conn is not None else None, "unit": "%"},
        ]

    def controls_snapshot(self) -> dict:
        """The panel's controls view: {device, online, controls, readings, versions}."""
        return {
            "device": self.name,
            "online": self.online,
            "controls": self.controls(),
            "readings": self.readings(),
            "versions": self.versions() if self.online else [],
        }

    def wake_prob(self) -> dict:
        conn = self.conn
        return {
            "device": self.name,
            "online": conn is not None,
            "available": conn is not None,
            "value": conn.wake_peak if conn is not None else None,
        }

    def set_wake_prob_stream(self, enabled: bool) -> dict:
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        conn.send({"type": "wake_peak_stream", "on": bool(enabled)})
        if not enabled:
            conn.wake_peak = None
        return {"device": self.name, "online": True, "available": True, "value": conn.wake_peak}

    def set_detect(self, enabled: bool) -> dict:
        """Debug switch for the device's wake-word detection. Not tracked here: the device
        turns detection back on at every new connection."""
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        conn.send({"type": "detect", "on": bool(enabled)})
        logger.info(f"{self.name}: wake-word detection {'on' if enabled else 'off'}")
        return {"device": self.name, "detect": bool(enabled)}

    def reload_model(self) -> dict:
        """The direct reload command: re-read the model files and send them; the device
        swaps them without a restart. Returns {device, models: {kind: name sent}}.
        Raises RuntimeError (offline)."""
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        logger.info(f"{self.name}: reloading the models")
        sent = {kind: name for kind, (name, path) in _model_paths(self.cfg).items()
                if conn.send_model(kind, name, path)}
        return {"device": self.name, "models": sent}

    def set_control(self, control_id: str, value: float) -> None:
        """A cutoff, the playback reserve or an auto volume setting is written to the config (the reconfigure that follows sends `config` or
        `volume_boost` to a connected device); the volume goes to the device.
        Raises LookupError (unknown control / no speaker) or RuntimeError (offline)."""
        config = _CONFIG_FIELDS.get(control_id)
        if config is not None:
            field, low, high, digits = config
            devices = [d.model_dump(mode="json") for d in self.rt.core.devices]
            for d in devices:
                if d["name"] == self.name:
                    d[field] = max(low, min(high, round(value, digits) if digits else round(value)))
            self.rt.svc.apply({"core": {"devices": devices}})
            return
        pct = max(0, min(100, round(value)))
        if control_id != SPEAKER_VOLUME:
            raise LookupError(f"{self.name} has no control {control_id!r}")
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        if not conn.speaker:
            raise LookupError(f"{self.name} has no speaker")
        conn.send({"type": "set_volume", "value": pct})
        conn.volume = pct

    def get_volume(self) -> int:
        """The speaker volume in percent: the last one the device reported or the server
        set. Raises RuntimeError (offline, or the device has not reported its volume) or
        LookupError (no speaker)."""
        conn = self.conn
        if conn is None:
            raise RuntimeError(f"{self.name} is offline")
        if not conn.speaker:
            raise LookupError(f"{self.name} has no speaker")
        if conn.volume is None:
            raise RuntimeError(f"{self.name} has not reported its volume")
        return conn.volume

    def set_volume(self, level: int | None = None, change: int | None = None) -> int:
        """Set the speaker volume to `level` percent, or move it by `change` percent from
        the current volume (get_volume); exactly one of the two (else
        ValueError). Goes through set_control: clamped to 0..100, raises LookupError (no
        speaker) or RuntimeError (offline). Returns the new volume in percent."""
        if (level is None) == (change is None):
            raise ValueError("exactly one of level and change is required")
        if change is not None:
            level = self.get_volume() + change
        self.set_control(SPEAKER_VOLUME, level)
        return self.conn.volume

    async def say(self, text: str) -> None:
        """Speak `text` (the panel's "Say on device"). Raises DeviceOfflineError, or what
        announce raises when the speech fails."""
        if not self.online:
            raise DeviceOfflineError(f"{self.name} is offline")
        await self.announce(text, "say")

    async def announce(self, text: str, reason: str, request: str | None = None) -> None:
        """Proactively say `text`: the text to the device as an announcement and, on
        a device with a speaker, the synthesized speech.

        Every announcement is a run in the run log: `reason` names who made the device
        speak ("say", "reminder", "assistant"), `request` is the request text it answers
        (if any), `text` is the reply, and a failure carries the error stage ("device"
        when offline, "TTS" when the speech failed). An announcement that reached the
        device also joins its conversation context as a note that it was spoken, so
        a follow-up question can refer to it. The text is spoken as it is, without the
        stress stage. Raises on failure, after the run is recorded."""
        t0 = time.perf_counter()
        record = new_record(self.name, reason, result="ok", stt_text=request, llm_text=text)
        try:
            conn = self.conn
            if conn is None:
                record.update(result="error", error_stage="device", error_text="offline")
                raise RuntimeError(f"{self.name} is offline")
            conn.send({"type": "reply", "text": text})
            await self.pipeline.speak_reply(text, record, accent=False)
            ctx = self.rt.core.context
            context.append_spoken(self.pipeline._context_path, text,
                                  max_turns=ctx.max_turns, ttl_seconds=ctx.ttl_seconds)
        finally:
            record["t_total"] = int((time.perf_counter() - t0) * 1000)
            await record_run(self.rt, record)

    def start_capture(self, seconds: int) -> captures.Capture:
        """Create a capture on disk and send `capture` for it. Raises DeviceOfflineError or
        CaptureBusyError."""
        conn = self.conn
        if conn is None:
            raise DeviceOfflineError(f"{self.name} is offline")
        if self.capture is not None:
            raise CaptureBusyError(f"{self.name} capture already in progress")
        seconds = max(1, min(int(seconds), CAPTURE_MAX_SECONDS))
        cap = captures.Capture.create(self.name, seconds)
        self._run_capture(cap)
        conn.send({"type": "capture", "id": cap.id, "seconds": seconds})
        logger.info(f"{self.name}: ⏺️ capture {cap.name} {seconds}s")
        return cap

    async def record_capture(self, seconds: int) -> captures.Capture:
        """Record a capture and return it once it is finished. Raises like start_capture."""
        cap = self.start_capture(seconds)
        await cap.finished.wait()
        return cap

    def stop_capture(self) -> captures.Capture:
        """End the running capture early and return it: it is `done` with the audio that came
        and that length. Online, `capture_stop` makes the device send what it recorded and its
        `audio_end`, and the watchdog finishes it CAPTURE_WAIT_MARGIN from now if they do not
        come; offline, it is finished now. Raises NoCaptureError."""
        cap = self.capture
        if cap is None:
            raise NoCaptureError(f"{self.name} has no capture running")
        cap.stopped = True
        conn = self.conn
        logger.info(f"{self.name}: ⏹️ capture {cap.name} stop" + (" sent" if conn else " while offline"))
        if conn is None:
            self.finish_capture()
            return cap
        self._send_stop(conn, cap)
        return cap

    def _send_stop(self, conn: Connection, cap: captures.Capture) -> None:
        """Send `capture_stop` for `cap`; the watchdog finishes it CAPTURE_WAIT_MARGIN from now
        if the rest of its audio and its `audio_end` do not come."""
        conn.send({"type": "capture_stop", "id": cap.id})
        self._capture_watchdog.cancel()
        self._capture_watchdog = asyncio.get_running_loop().call_later(CAPTURE_WAIT_MARGIN, self.finish_capture)

    def _run_capture(self, cap: captures.Capture) -> None:
        """Make `cap` the running capture; the watchdog finishes it with what came by
        CAPTURE_WAIT_MARGIN past its end."""
        self.capture = cap
        delay = cap.started + cap.seconds + CAPTURE_WAIT_MARGIN - time.time()
        self._capture_watchdog = asyncio.get_running_loop().call_later(delay, self.finish_capture)

    def finish_capture(self, error: str | None = None) -> None:
        cap, self.capture = self.capture, None
        if cap is None:
            return
        self._capture_watchdog.cancel()
        cap.finish(error)
        logger.info(f"{self.name}: ⏹️ capture {cap.name} {cap.state}"
                    + (f": {cap.error}" if cap.error else "")
                    + (f", {len(cap.gaps)} gap(s)" if cap.gaps else ""))

    def resume_capture(self, conn: Connection) -> None:
        """After a hello: take up the capture the device holds, or record the time left of
        one it lost by restarting (docs/device-protocol.md, "Capture"). The capture the
        device holds may be one on disk only (the server restarted); every other unfinished
        capture of the device on disk is finished as is. A held capture that is not taken up
        gets `capture_stop`."""
        now = time.time()
        running = self.capture.stem if self.capture is not None else None
        for cap in captures.of_device(self.name):
            if not cap.running or cap.stem == running:
                continue
            if (self.capture is None and cap.id == conn.held_capture
                    and now < cap.started + cap.seconds + CAPTURE_WAIT_MARGIN):
                self._run_capture(cap)
                continue
            cap.finish()
            logger.info(f"{self.name}: capture {cap.name} found unfinished; finished as is ({cap.state})")
        cap = self.capture
        held = conn.held_capture
        if held is not None and (cap is None or cap.id != held):
            # Otherwise the device records it to its end with wake-word detection paused.
            conn.send({"type": "capture_stop", "id": held})
            logger.info(f"{self.name}: ⏹️ capture {held} held by the device is not taken up; stop sent")
        if cap is None:
            return
        if conn.held_capture == cap.id:
            start = max(0, cap.end - cap.base)
            conn.send({"type": "capture_resume", "id": cap.id, "from": start})
            logger.info(f"{self.name}: capture {cap.name} resumed from frame {start}")
            if cap.stopped:
                # A stop sent before the link was lost may never have reached the device.
                self._send_stop(conn, cap)
            return
        if cap.stopped:
            # The device came back without it (it restarted): a stopped capture records no more.
            self.finish_capture()
            return
        left = cap.started + cap.seconds - now
        if left < 1:
            self.finish_capture()
            return
        cap.segment(now)
        conn.send({"type": "capture", "id": cap.id, "seconds": round(left, 3)})
        logger.info(f"{self.name}: came back without capture {cap.name}; recording the {left:.1f}s left")


class DeviceManager:
    """The device server: one aiohttp app (/device, /health) and one Device per name."""

    def __init__(self, runtime):
        self.rt = runtime
        self.host = runtime.core.audio.host
        self.port = runtime.core.audio.port
        self._devices: dict[str, Device] = {}
        self._runner: web.AppRunner | None = None
        # One phrase heard by several devices keeps one session (docs/device-protocol.md).
        self.arbiter = WakeArbiter(
            online_count=lambda: sum(1 for d in self.clients if d.online),
            online_pipelines=lambda: [d.pipeline for d in self.clients if d.online],
            send_reply=self._send_reply,
            any_playing=lambda: any(d.conn.playing for d in self.clients if d.online))

    def _device(self, name: str) -> Device:
        device = self._devices.get(name)
        if device is None:
            device = self._devices[name] = Device(name, self.rt)
            device.pipeline.arbiter = self.arbiter
        return device

    def _send_reply(self, pipeline, text: str) -> None:
        """The arbiter's hook: show `text` on the device of `pipeline` as an
        announcement — a `reply` without `session`. An offline device is skipped."""
        device = self._devices.get(pipeline.name)
        if device is not None and device.conn is not None:
            device.conn.send({"type": "reply", "text": text})

    @property
    def clients(self) -> list[Device]:
        """The enabled configured devices."""
        return [self._device(cfg.name) for cfg in self.rt.core.devices if cfg.enabled]

    def find(self, device_name: str) -> Device:
        """The enabled configured device of that name; LookupError for any other name."""
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
                # Seconds since the device booted; None while offline.
                "uptime_seconds": int(time.time() - device.conn.booted_at) if online else None,
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
        await target.announce(text, "reminder")

    def capture_status(self, device_name: str) -> dict:
        """The device's running capture, else its newest one on disk, else idle."""
        device = self._devices.get(device_name)
        cap = (device.capture if device is not None else None) or captures.newest(device_name)
        return cap.snapshot() if cap is not None else captures.idle(device_name)

    def capture_stop(self, device_name: str) -> captures.Capture:
        """End the device's running capture early (Device.stop_capture) and return it. Raises
        LookupError for an unknown device, NoCaptureError when none runs."""
        return self.find(device_name).stop_capture()

    def capture_delete(self, device_name: str, name: str) -> bool:
        """Delete a capture from disk; False when there is no such capture. Raises
        CaptureBusyError while it runs."""
        device = self._devices.get(device_name)
        if device is not None and device.capture is not None and device.capture.name == name:
            raise CaptureBusyError(f"{device_name} capture {name} is running")
        cap = captures.find(device_name, name)
        if cap is None:
            return False
        cap.remove()
        return True

    async def play_chime(self, sound_path: str, device_name: str | None = None) -> dict:
        """Play the given chime on the named device, or on every device, for an
        operator preview. Returns {"played": [names], "offline": [names]}; a device
        that fails to play is reported offline. Raises LookupError for an unknown name."""
        mime, audio = await asyncio.to_thread(build_ack_clip, sound_path)
        targets = [self.find(device_name)] if device_name is not None else self.clients
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
        replace an older connection of the same id, then send `config`, `volume_boost`, what
        the device's capture needs, `log_stream` and the models."""
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
        # First, before the connection is attached: a play that starts on it (during the await
        # below) paces by the reserve in config_sent.
        conn.send_config(cfg)
        # A new connection has no `ambient` yet: the boost is 0.
        conn.send_boost(0)
        old = device.attach(conn)
        if old is not None:
            old.close(CLOSE_REPLACED)
            await device.pipeline.on_disconnect()
        rebooted = (device.booted_at is not None
                    and conn.booted_at - device.booted_at > REBOOT_MARGIN_S)
        device.booted_at = conn.booted_at
        device.publish_event("connected", {
            "fw": conn.fw, "uptime": hello["uptime"], "reset": hello.get("reset"),
            "speaker": conn.speaker, "replaced": old is not None, "rebooted": rebooted,
        })
        # Before the models: the device holds a capture's unsent audio only in its ~7 s tap.
        # A capture file that cannot be read must not keep the device from connecting.
        try:
            device.resume_capture(conn)
        except Exception as e:
            logger.error(f"{name}: captures not taken up: {e}")
        # Off too: the device keeps collecting its log across a lost link until told.
        conn.send_log_stream(cfg.log)
        for kind, (model, path) in _model_paths(cfg).items():
            conn.send_model(kind, model, path)
        logger.info(
            f"{name}: connected ({conn.fw}, speaker={conn.speaker}"
            f"{', replaced the older connection' if old else ''})"
        )
        return device, conn

    async def reconfigure(self) -> None:
        """Apply the device config to live connections: close a device that left the
        config or got disabled, send `config` to one whose `config` values changed,
        `volume_boost` to one whose boost changed with its auto volume settings, `log_stream`
        to one whose log switch changed and a model whose name changed."""
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
                continue
            if conn.config_sent != _config_fields(cfg):
                conn.send_config(cfg)
            if not cfg.auto_volume:
                # The device stops measuring: its old values would show and come back stale.
                conn.ambient.clear()
            device.update_boost()
            if conn.log_on != cfg.log:
                conn.send_log_stream(cfg.log)
            for kind, (model, path) in _model_paths(cfg).items():
                if conn.models.get(kind) != model:
                    conn.send_model(kind, model, path)

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
