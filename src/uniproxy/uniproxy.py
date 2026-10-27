# Adapted from https://github.com/amirilivice/quasard @ a142b19, alice_server/uniproxy.py.
"""The Yandex Station's stock quasar as a device of the zakhar server.

quasar connects to `uniProxyUrl` of its quasar.cfg (GET /uni.ws) and speaks the uniproxy
protocol: JSON events (firmware 1.56) or AAPI protobuf frames (3.x, aliceapi.py). The station
side here is quasard's: SynchronizeState, VoiceInput with its Ogg/Opus microphone stream, the
empty partial ASR results, streamcontrol, TTS.Speak streams of Ogg/Opus. What quasard answered
with its own ASR, LLM and TTS comes from the zakhar server instead: a Link per station device id
is a client of docs/device-protocol.md on the server's /device, kept connected while a station
connection of that id is open.

  station                               Link                          server
  Vins.VoiceInput                  ──►  wake {session, «Алиса»}  ──►
  Ogg/Opus microphone              ──►  0x01 frames: held until the station's SpotterEnd (or
                                        PREROLL_WAIT_S) as the pre-roll, then live
  Spotter.Validation               ◄──  state listening          ◄──
                                        state transcribing: the microphone stops going to the server
  final ASR.Result, mic closed     ◄──  transcript (its text)    ◄──
  VinsResponse, TTS.Speak, Ogg     ◄──  reply, then the answer's play_start / 0x02 / play_end
  empty VinsResponse               ◄──  state idle or error without an answer, a refused wake

The wake word stays quasar's own: the station spots «Алиса», and the server's wake-word models
are refused (`model_loaded ok: false`). Streams of a session before its `reply` (the end-of-phrase
chime, a filler) are not sent to the station; any other stream (an announcement) goes as a
TTS.Speak of no request. The station reports nothing of its playback, so `played` goes once a
stream's duration has passed since its first audio came.

The server end-points a phrase from the microphone audio, so when the station stops its own stream
first (a streamcontrol close or Vins.CancelRequest), the Link feeds the server silence until its VAD
ends the session; a cancel then goes back to the station as no answer at all.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from aiohttp import WSCloseCode, WSMsgType, web
from loguru import logger

from . import aliceapi, protocol as P
from .audio import ASR_RATE, OggOpusStreamEncoder, OpusStreamDecoder, frame_stream, parse_stream_frame, split_ogg_pages

# streamcontrol.action of the station's SpotterEnd (the firmware's Close is 0, LazyTTS 3).
STREAM_SPOTTER_END = 2
# The station closing its own microphone stream, and the reasons that mean "no answer".
STREAM_CLOSE = 0
REASON_CANCEL = 7
REASON_NO_SPEECH = 8

# Binary frame kinds of the device protocol (first byte).
MIC = 0x01
PLAY = 0x02
MODEL = 0x03

WAKE_WORD = "Алиса"
# Samples per 0x01 frame: 32 ms at 16 kHz.
MIC_FRAME = 512
# The longest the microphone audio is held as the pre-roll when no SpotterEnd comes.
PREROLL_WAIT_S = 2.0
# How often an empty partial ASR.Result keeps the station listening (quasard's default).
PARTIAL_INTERVAL_S = 0.3
# The longest the server is fed silence after the station stops its microphone, so the server's
# VAD end-points a request the station abandoned instead of staying in `listening` forever.
DRAIN_MAX_S = 3.0
# The playback rate the server resamples to; TTS.Speak streams are 48 kHz Ogg/Opus.
PLAY_RATE = 48000
# The station queues a whole answer itself: with a buffer this large the server's pacing lets it
# send the speech as fast as it is synthesized.
BUFFER_MS = 60000
# The hello's volume: TSynchronizeStateRequest (aliceapi.desc) carries no sound level.
VOLUME = 50
# Liveness and reconnects as the device protocol has them for a device.
SILENCE_S = 30
RECONNECT_MAX_S = 30


@dataclass
class VoiceRequest:
    """One VoiceInput of a station and the session its wake proposed."""

    station: StationSession
    ref: str  # messageId of the VoiceInput
    stream_id: int  # the station's microphone stream
    request_id: str
    sequence_number: int
    spotter_validation: bool
    session: int = 0
    decoder: OpusStreamDecoder = field(default_factory=OpusStreamDecoder)
    pcm: bytearray = field(default_factory=bytearray)  # s16le audio not sent yet
    samples: int = 0  # decoded so far
    preroll_sent: bool = False
    mic_open: bool = True  # its audio goes to the server
    taken: bool = False  # the server took the wake (`listening`)
    cancelled: bool = False  # the station abandoned it: no answer goes back
    draining: bool = False  # silence is being fed so the server end-points it
    asr_sent: bool = False  # its final ASR.Result went out
    reply: str | None = None
    responded: bool = False  # its VinsResponse went out
    directives: list[dict] = field(default_factory=list)  # for its VinsResponse
    partial_no: int = 0

    @property
    def duration_ms(self) -> int:
        return self.samples * 1000 // ASR_RATE


@dataclass
class Play:
    """One play stream of the server, from its play_start until it is reported."""

    id: int
    station: StationSession | None = None  # None: not sent to the station
    sid: int = 0  # its stream on the station connection
    encoder: OggOpusStreamEncoder | None = None
    samples: int = 0
    first_at: float | None = None  # loop time of its first audio
    timer: asyncio.TimerHandle | None = None


class Bridge:
    """The /uni.ws endpoint: every station connection runs a StationSession, and every station
    device id has its Link to the zakhar server while one of its connections is open."""

    def __init__(self, server_url: Callable[[], str]):
        self.server_url = server_url  # the server's /device URL, read at each (re)connect
        self.links: dict[str, Link] = {}

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        device_id = request.headers.get("X-UPRX-DEVICE-ID", "unknown")
        link = self.links.get(device_id)
        if link is None:
            link = self.links[device_id] = Link(self, device_id)
        station = StationSession(ws, request.headers, link)
        link.stations.append(station)
        try:
            await station.run()
        finally:
            link.stations.remove(station)
            if not link.stations:
                del self.links[device_id]
                await link.stop()
        return ws

    async def close(self) -> None:
        """Close every station connection; the link of each device goes with its last one."""
        for link in list(self.links.values()):
            for station in list(link.stations):
                await station.ws.close(code=WSCloseCode.GOING_AWAY)


class Link:
    """The zakhar device of one station device id. The station connection that sent a request
    gets its answer; any other stream goes to the latest connection."""

    def __init__(self, bridge: Bridge, device_id: str):
        self.bridge = bridge
        self.device_id = device_id
        self.stations: list[StationSession] = []  # open connections, the latest last
        self.request: VoiceRequest | None = None  # the latest VoiceInput while its run lasts
        self._started = time.monotonic()  # the device's "boot", for the hello's uptime
        self._session = 0
        self._out: asyncio.Queue | None = None  # what goes to the server, while connected
        self._plays: dict[int, Play] = {}  # not reported yet
        self._open: Play | None = None  # the one between its play_start and play_end
        self._model: dict | None = None  # the `model` whose bytes are coming
        self._model_left = 0
        self._drains: set[asyncio.Task] = set()  # silence feeds for abandoned requests
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(_log_task_error)

    @property
    def up(self) -> bool:
        return self._out is not None

    async def stop(self) -> None:
        self._task.cancel()
        for t in list(self._drains):
            t.cancel()
        await asyncio.gather(self._task, *self._drains, return_exceptions=True)

    # ------------------------------------------------------------ server connection

    async def _run(self) -> None:
        delay = 1
        async with aiohttp.ClientSession() as http:
            while True:
                url = self.bridge.server_url()
                try:
                    async with http.ws_connect(url, timeout=aiohttp.ClientWSTimeout(ws_receive=SILENCE_S)) as ws:
                        self._out = asyncio.Queue()
                        writer = asyncio.create_task(self._write(ws, self._out))
                        self._send(self._hello())
                        logger.info(f"station {self.device_id}: linked to {url}")
                        delay = 1
                        try:
                            async for msg in ws:
                                if msg.type == WSMsgType.TEXT:
                                    await self._on_message(json.loads(msg.data))
                                elif msg.type == WSMsgType.BINARY:
                                    await self._on_binary(msg.data)
                        finally:
                            self._out = None
                            writer.cancel()
                            await self._lost()
                    logger.warning(f"station {self.device_id}: link closed (code {ws.close_code})")
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                    logger.warning(f"station {self.device_id}: link to {url} failed: {e!r}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_S)

    @staticmethod
    async def _write(ws, out: asyncio.Queue) -> None:
        try:
            while True:
                data = await out.get()
                if isinstance(data, str):
                    await ws.send_str(data)
                else:
                    await ws.send_bytes(data)
        except ConnectionError:
            pass  # the receive loop sees the connection go

    def _send(self, msg: dict | bytes) -> None:
        """Queue a message or a binary frame for the server, in order; nothing while not connected."""
        if self._out is not None:
            self._out.put_nowait(msg if isinstance(msg, bytes) else json.dumps(msg, ensure_ascii=False))

    def _hello(self) -> dict:
        return {
            "type": "hello", "id": self.device_id, "fw": "quasar bridge",
            "uptime": int(time.monotonic() - self._started),
            "caps": {"speaker": True, "screen": False, "follow_up": False},
            "mic": {"channels": 1},
            "play": {"rate": PLAY_RATE, "buffer_ms": BUFFER_MS},
            "volume": VOLUME,
        }

    async def _lost(self) -> None:
        """The connection is gone: the stream the station is being sent closes, and the request
        ends as a failed run does."""
        await self._drop_plays()
        self._model = None
        if self.request is not None:
            await self._end(self.request)

    # ------------------------------------------------------------ from the station

    def wake(self, req: VoiceRequest) -> None:
        """A VoiceInput: `wake` proposing the next session. The request before it is left as the
        station left it: its microphone stops, nothing more of it goes to the station."""
        if self.request is not None:
            self.request.mic_open = False
        self._session += 1
        req.session = self._session
        self.request = req
        self._send({"type": "wake", "session": req.session, "word": WAKE_WORD})
        asyncio.get_running_loop().call_later(PREROLL_WAIT_S, self.send_preroll, req)
        logger.info(f"station {self.device_id}: wake, session {req.session}")

    def mic(self, req: VoiceRequest, pcm: np.ndarray) -> None:
        """Decoded microphone audio of `req` (float32, 16 kHz): held until the pre-roll goes,
        live after it."""
        req.samples += pcm.size
        req.pcm += (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        if req.preroll_sent:
            self._send_mic(req)

    def send_preroll(self, req: VoiceRequest) -> None:
        """Send the audio held so far as the pre-roll burst (at the station's SpotterEnd, or
        PREROLL_WAIT_S after the VoiceInput); live audio follows it."""
        if req.preroll_sent or not req.mic_open:
            return
        req.preroll_sent = True
        self._send_mic(req)

    def _send_mic(self, req: VoiceRequest) -> None:
        size = MIC_FRAME * 2
        end = len(req.pcm) - len(req.pcm) % size
        for i in range(0, end, size):
            self._send(bytes([MIC]) + req.pcm[i:i + size])
        del req.pcm[:end]

    def station_stopped(self, req: VoiceRequest | None, cancelled: bool) -> None:
        """The station closed its microphone for `req` before the server end-pointed it (a
        streamcontrol close or Vins.CancelRequest). The server ends a phrase from the audio, so
        the Link keeps feeding it silence until its VAD does; `cancelled` sends no answer back."""
        if req is None or req is not self.request or req.draining:
            return
        req.cancelled = req.cancelled or cancelled
        req.draining = True
        self.send_preroll(req)  # flush whatever speech was held, then drain silence
        t = asyncio.create_task(self._drain_loop(req))
        self._drains.add(t)
        t.add_done_callback(self._drains.discard)
        t.add_done_callback(_log_task_error)

    async def _drain_loop(self, req: VoiceRequest) -> None:
        silence = bytes([MIC]) + bytes(MIC_FRAME * 2)
        for _ in range(int(DRAIN_MAX_S * ASR_RATE / MIC_FRAME)):
            if req is not self.request or not req.mic_open or not self.up:
                return
            self._send(silence)
            await asyncio.sleep(MIC_FRAME / ASR_RATE)

    # ------------------------------------------------------------ from the server

    def _of(self, session) -> VoiceRequest | None:
        req = self.request
        return req if req is not None and req.session == session else None

    async def _on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "state":
            await self._on_state(msg)
        elif kind == "reply":
            req = self._of(msg.get("session"))
            if req is not None:
                req.reply = msg.get("text", "")
        elif kind == "transcript":
            req = self._of(msg.get("session"))
            if req is not None:
                await self._close_mic(req)
                await self._finish_asr(req, msg.get("text", ""))
        elif kind == "play_start":
            await self._on_play_start(msg)
        elif kind == "play_end":
            await self._on_play_end(msg["id"])
        elif kind == "play_stop":
            for stream_id in await self._drop_plays():
                self._send({"type": "played", "id": stream_id, "dropped": True})
        elif kind == "set_volume":
            await self._on_set_volume(msg["value"])
        elif kind == "model":
            self._model, self._model_left = msg, msg["size"]
        elif kind == "capture":
            self._send({"type": "audio_end", "reason": "busy", "id": msg["id"]})
        # config, volume_boost, log_stream, wake_peak_stream, detect: nothing to do.

    async def _on_binary(self, data: bytes) -> None:
        if data[0] == PLAY:
            await self._on_audio(data[1:])
        elif data[0] == MODEL and self._model is not None:
            self._model_left -= len(data) - 1
            if self._model_left <= 0:
                self._send({"type": "model_loaded", "kind": self._model["kind"],
                            "name": self._model["name"], "ok": False})
                self._model = None

    async def _on_state(self, msg: dict) -> None:
        req = self._of(msg.get("session"))
        if req is None:
            return
        value = msg.get("value")
        if value == "listening":
            if not req.taken:
                req.taken = True
                if req.spotter_validation:
                    await req.station.send_json(P.spotter_validation(req.ref, self.device_id, True))
        elif value in ("idle", "error"):
            await self._end(req)
        else:
            await self._close_mic(req)

    async def _close_mic(self, req: VoiceRequest) -> None:
        """The microphone of `req` stops (a `state` other than `listening`). A refused wake fails
        its spotter validation and the station's stream closes; a taken one waits for its
        transcript (_finish_asr): quasar takes an empty final ASR.Result for a phrase it did not
        hear and plays no answer after it."""
        if not req.mic_open:
            return
        req.mic_open = False
        if req.taken or req.cancelled:
            return
        if req.spotter_validation:
            await req.station.send_json(P.spotter_validation(req.ref, self.device_id, False))
        await req.station.send_json(P.stream_control(req.stream_id))

    async def _finish_asr(self, req: VoiceRequest, text: str) -> None:
        """A taken wake's final ASR.Result with `text`, and the station's stream closes; once."""
        if not req.taken or req.asr_sent or req.cancelled:
            return
        req.asr_sent = True
        await req.station.send_json(P.asr_result(req.ref, text, req.partial_no, True, req.duration_ms))
        await req.station.send_json(P.stream_control(req.stream_id))

    async def _end(self, req: VoiceRequest) -> None:
        """The run of `req` is over: its microphone stops and, if no answer went, an empty
        VinsResponse does."""
        await self._close_mic(req)
        await self._finish_asr(req, "")
        if self.request is req:
            self.request = None
        if not req.responded and not req.cancelled:
            req.responded = True
            await req.station.respond(req, "", req.directives)

    async def _on_play_start(self, msg: dict) -> None:
        play = self._open = self._plays[msg["id"]] = Play(msg["id"])
        req = self._of(msg.get("session"))
        if req is not None and req.reply is None:
            return  # the chime or a filler before the answer: not sent to the station
        if req is not None and req.cancelled:
            return  # a cancelled request: reported played (station stays None), not sent out
        if req is not None and not req.responded:
            station, ref = req.station, req.ref
            await self._finish_asr(req, "")
            req.responded = True
            await station.respond(req, req.reply, req.directives)
        else:
            station, ref = (req.station if req is not None else self.stations[-1]), ""
        play.station, play.sid, play.encoder = station, station.next_stream_id(), OggOpusStreamEncoder()
        await station.send_json(P.tts_speak(ref, play.sid))

    async def _on_audio(self, payload: bytes) -> None:
        play = self._open
        if play is None:
            return
        if play.first_at is None:
            play.first_at = asyncio.get_running_loop().time()
        pcm = np.frombuffer(payload, dtype="<i2")
        play.samples += pcm.size
        if play.station is not None:
            await play.station.send_audio(play.sid, play.encoder.add(pcm, PLAY_RATE))

    async def _on_play_end(self, stream_id: int) -> None:
        play = self._plays.get(stream_id)
        if play is None:
            return
        if self._open is play:
            self._open = None
        if play.station is not None:
            await play.station.send_audio(play.sid, play.encoder.close())
            await play.station.send_json(P.stream_control(play.sid))
        loop = asyncio.get_running_loop()
        left = 0.0 if play.first_at is None else play.first_at + play.samples / PLAY_RATE - loop.time()
        play.timer = loop.call_later(max(left, 0.0), self._played, play)

    def _played(self, play: Play) -> None:
        if self._plays.pop(play.id, None) is not None:
            self._send({"type": "played", "id": play.id, "dry": 0, "gap_ms": 0})

    async def _drop_plays(self) -> list[int]:
        """Close the stream the station is being sent and forget every unreported one; return
        their ids."""
        play, self._open = self._open, None
        if play is not None and play.station is not None:
            play.encoder.close()
            await play.station.send_json(P.stream_control(play.sid))
        for p in self._plays.values():
            if p.timer is not None:
                p.timer.cancel()
        ids = list(self._plays)
        self._plays.clear()
        return ids

    async def _on_set_volume(self, value: int) -> None:
        action = P.client_action("sound_set_level", {"new_level": round(value / 10), "new_percent_level": value})
        req = self.request
        if req is not None and not req.responded:
            # With the request's VinsResponse, as Alice's own volume commands come.
            req.directives.append(action)
        else:
            await self.stations[-1].send_json(
                P.directive("System", "Push", {"directives": [action]}, ack=int(time.time() * 1000)))


class StationSession:
    """One /uni.ws connection of a station: quasard's station side, its voice requests handed
    to the device's Link. Nothing the station sends is logged: SynchronizeState carries its
    Yandex OAuth token."""

    def __init__(self, ws: web.WebSocketResponse, headers, link: Link):
        self.ws = ws
        self.link = link
        self.device_id = link.device_id
        self.session_id = headers.get("X-UPRX-SSID") or str(uuid.uuid4())
        self.proto = False  # the station sent an AAPI frame: answer in protobuf (firmware 3.x)
        self._send_lock = asyncio.Lock()
        self._next_out_stream = 2  # server streams are even
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ io

    async def send_json(self, msg: dict) -> None:
        """Send a server message, built in the JSON form of firmware 1.56; in protobuf for 3.x."""
        if self.proto:
            pm = aliceapi.server_to_proto(msg)
            if pm is not None:
                await self.send_bytes(aliceapi.encode_server(pm))
            return
        async with self._send_lock:
            await self._write(self.ws.send_str, json.dumps(msg, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self._write(self.ws.send_bytes, data)

    async def _write(self, send, data) -> None:
        # A connection that went away drops what the link still sends it.
        if self.ws.closed:
            return
        try:
            await send(data)
        except ConnectionError:
            pass

    async def send_audio(self, sid: int, data: bytes) -> None:
        for chunk in split_ogg_pages(data):
            await self.send_bytes(frame_stream(sid, chunk))

    def next_stream_id(self) -> int:
        sid = self._next_out_stream
        self._next_out_stream += 2
        return sid

    async def respond(self, req: VoiceRequest, text: str, directives: list[dict]) -> None:
        await self.send_json(P.vins_response(req.ref, req.request_id, self.session_id, req.sequence_number,
                                             text, speech=text, directives=directives))

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(_log_task_error)
        return t

    # ------------------------------------------------------------ main loop

    async def run(self) -> None:
        logger.info(f"station {self.device_id}: connected")
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._on_binary(msg.data)
        finally:
            for t in list(self._tasks):
                t.cancel()
            logger.info(f"station {self.device_id}: disconnected")

    async def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"station {self.device_id}: a text frame that is not JSON")
            return
        await self._on_message(msg)

    async def _on_message(self, msg: dict) -> None:
        if "streamcontrol" in msg:
            await self._on_stream_control(msg["streamcontrol"])
            return
        event = msg.get("event")
        if not event:
            return
        header = event.get("header", {})
        payload = event.get("payload", {})
        name = f"{header.get('namespace')}.{header.get('name')}"
        ref = header.get("messageId", "")
        if name == "System.SynchronizeState":
            await self.send_json(P.sync_state_response(ref, self.session_id))
            await self.send_json(P.push_notify(ref))
        elif name == "System.EchoRequest":
            await self.send_json(P.directive("System", "EchoResponse", payload, ref=ref))
        elif name == "Log.Spotter":
            # The station sends the recording of the activation phrase and waits for the ack.
            await self.send_json(P.directive("Log", "Ack", {}, ref=ref))
        elif header.get("name") == "RequestStat":
            # Without the ack the station repeats RequestStat.
            await self.send_json(P.directive("Log", "RequestStatAck", {}, ref=ref))
        elif name in ("Vins.VoiceInput", "ASR.Recognize"):
            await self._on_voice_input(header, payload)
        elif name == "Vins.CancelRequest":
            req = self.link.request
            if req is not None and req.station is self:
                self.link.station_stopped(req, cancelled=True)
        elif header.get("namespace") == "Vins":
            # TextInput (the station sends some of its own after SynchronizeState), VinsRequest,
            # CustomInput, the player's server_action callbacks: none goes to the server.
            await self._respond_empty(ref, payload)

    async def _on_binary(self, data: bytes) -> None:
        if len(data) < 4:
            return
        if data.startswith(aliceapi.MAGIC):
            # Firmware 3.x: events come in protobuf; the session switches and they go on in the JSON form.
            if not self.proto:
                logger.info(f"station {self.device_id}: speaks protobuf (AAPI)")
                self.proto = True
            try:
                msg = aliceapi.client_to_json(aliceapi.decode_client(data))
            except Exception as e:
                logger.warning(f"station {self.device_id}: AAPI frame not parsed ({type(e).__name__})")
                return
            if msg is not None:
                await self._on_message(msg)
            return
        stream_id, chunk = parse_stream_frame(data)
        req = self._request(stream_id)
        if req is None or not req.mic_open:
            return
        try:
            pcm = req.decoder.feed(chunk)
        except Exception as e:
            logger.warning(f"station {self.device_id}: microphone audio not decoded ({type(e).__name__})")
            return
        self.link.mic(req, pcm)

    def _request(self, stream_id) -> VoiceRequest | None:
        """The link's request when it is this connection's microphone stream `stream_id`."""
        req = self.link.request
        return req if req is not None and req.station is self and req.stream_id == stream_id else None

    # ------------------------------------------------------------ voice

    async def _on_voice_input(self, header: dict, payload: dict) -> None:
        hdr = payload.get("header") or {}
        req = VoiceRequest(
            station=self,
            ref=header.get("messageId", ""),
            stream_id=int(header.get("streamId", 1)),
            request_id=hdr.get("request_id") or str(uuid.uuid4()),
            sequence_number=int(hdr.get("sequence_number") or 0),
            spotter_validation=bool(payload.get("enable_spotter_validation")),
        )
        if not self.link.up:
            logger.warning(f"station {self.device_id}: no link to the server, the request gets an empty answer")
            await self.respond(req, "", [])
            return
        await self.send_json(P.input_start_ack(req.ref))
        self.link.wake(req)
        self._spawn(self._partials_loop(req))

    async def _partials_loop(self, req: VoiceRequest) -> None:
        """Empty partial ASR.Results while the microphone streams, as uniproxy sends them."""
        while True:
            await asyncio.sleep(PARTIAL_INTERVAL_S)
            if not req.mic_open:
                return
            await self.send_json(P.asr_result(req.ref, "", req.partial_no, False, req.duration_ms))
            req.partial_no += 1

    async def _on_stream_control(self, sc: dict) -> None:
        req = self._request(sc.get("streamId"))
        if req is None:
            return
        if sc.get("action") == STREAM_SPOTTER_END:
            self.link.send_preroll(req)
        elif sc.get("action") == STREAM_CLOSE and req.mic_open:
            # A close after the server stopped listening is no cancel: quasar closes its stream
            # itself once it gets the final ASR.Result, and waits for the answer.
            self.link.station_stopped(req, sc.get("reason") in (REASON_CANCEL, REASON_NO_SPEECH))

    async def _respond_empty(self, ref: str, payload: dict) -> None:
        hdr = payload.get("header") or {}
        await self.send_json(P.vins_response(ref, hdr.get("request_id") or str(uuid.uuid4()), self.session_id,
                                             int(hdr.get("sequence_number") or 0), "", speech=""))


def _log_task_error(t: asyncio.Task) -> None:
    if not t.cancelled() and t.exception() is not None:
        logger.opt(exception=t.exception()).error("station bridge task failed")
