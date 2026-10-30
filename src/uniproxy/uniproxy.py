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
  TTS.Speak, Ogg (chime, filler)   ◄──  play_start / 0x02 / play_end before the reply
  empty ASR.Result (partials)      ◄──  after the transcript: the answer takes longer than
                                        quasar's 15-s request timeout, so the final result
                                        waits for it and the partials hold the station
  final ASR.Result, mic closed     ◄──  with the reply, at the answer's play_start
  VinsResponse, TTS.Speak, Ogg     ◄──  reply, then the answer's play_start / 0x02 / play_end
  empty VinsResponse               ◄──  state idle or error without an answer, a refused wake

An AAPI station (firmware 3.x) times a request out 12 s after the phrase unless an AliceResponse
comes, and the partials do not hold it. There the transcript gets at once an AliceResponse
skeleton that opens a response stream; the answer completes it as a DeferredAliceResponse (an
empty one on idle or error without an answer).

The wake word stays quasar's own: the station spots «Алиса», and the server's wake-word models
are refused (`model_loaded ok: false`). Every stream of a session (the end-of-phrase chime, a
filler, the answer) goes out as a TTS.Speak of its request; the VinsResponse waits for the
reply's text. An AAPI station plays only the first TTS stream of a request, so there they all
go into one TTS.Speak of it, closed after the answer. A stream of no session (an announcement)
goes as a TTS.Speak of no request. The station reports nothing of its playback, so `played`
goes once a stream's duration has passed since its first audio came.

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
# The station closing its own microphone stream: a cancel (no answer goes back) and no speech
# heard (an empty window unless the server heard a phrase, see Link._end).
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
# The longest a follow-up cycle waits for the server's follow-up `listening` (the stage may be
# off, or the answer was dropped): past it the cycle's stream closes and its audio is dropped.
FOLLOWUP_LISTEN_WAIT_S = 4.0
# How often an empty partial ASR.Result keeps the station listening (quasard's default).
PARTIAL_INTERVAL_S = 0.3
# The playback rate the server resamples to; TTS.Speak streams are 48 kHz Ogg/Opus.
PLAY_RATE = 48000
# The silence step of a shared AAPI TTS stream between its plays (Link._keep_shared).
KEEP_S = 0.2
# AAPI: the directive that marks where in a response's chain the request's TTS plays
# (protocol.vins_response puts the same one in front of a voiced answer).
TTS_PLACEHOLDER = {"name": "tts_play_placeholder", "payload": {"channel": "Dialog"},
                   "sub_name": "tts_play_placeholder", "type": "client_action", "is_led_silent": True}
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
    transcript: str = ""  # its final text, the ASR.Result held until the answer (see _hold_asr)
    reply: str | None = None
    responded: bool = False  # its VinsResponse went out
    stream_started: bool = False  # AAPI: its early stream-start AliceResponse skeleton went out
    tts_sid: int = 0  # AAPI after the skeleton: the one TTS.Speak stream its plays share (0: none)
    tts_encoder: OggOpusStreamEncoder | None = None
    directives: list[dict] = field(default_factory=list)  # for its VinsResponse
    partial_no: int = 0
    followup: bool = False  # the post-answer listening window of an already answered session
    no_speech: bool = False  # the station closed its microphone hearing no speech (reason 8)

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
    request: VoiceRequest | None = None  # AAPI: the request whose one TTS stream it goes into


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
        self.device_id = device_id  # the station's own id, as it names itself to uniproxy
        # Its name as a zakhar device: a station id is "LB" and 30 hex digits that differ in the tail.
        self.name = "YA-" + device_id[-8:]
        self.stations: list[StationSession] = []  # open connections, the latest last
        self.request: VoiceRequest | None = None  # the latest VoiceInput while its run lasts
        self._started = time.monotonic()  # the device's "boot", for the hello's uptime
        self._session = 0
        # The session whose follow-up `listening` already came: a window the server armed before
        # the station's VoiceInput for it arrived (the two travel different links and race).
        self._window = 0
        self._out: asyncio.Queue | None = None  # what goes to the server, while connected
        self._plays: dict[int, Play] = {}  # not reported yet
        self._open: Play | None = None  # the one between its play_start and play_end
        self._model: dict | None = None  # the `model` whose bytes are coming
        self._model_left = 0
        self._drains: set[asyncio.Task] = set()  # silence feeds for abandoned requests
        # final-ASR holds and shared-TTS-stream silence (_keep_shared) of requests waiting for an answer
        self._holds: set[asyncio.Task] = set()
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(_log_task_error)

    @property
    def up(self) -> bool:
        return self._out is not None

    async def stop(self) -> None:
        self._task.cancel()
        for t in list(self._drains) + list(self._holds):
            t.cancel()
        await asyncio.gather(self._task, *self._drains, *self._holds, return_exceptions=True)

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
                        logger.info(f"station {self.name}: linked to {url}")
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
                    logger.warning(f"station {self.name}: link closed (code {ws.close_code})")
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                    logger.warning(f"station {self.name}: link to {url} failed: {e!r}")
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
            "type": "hello", "id": self.name, "fw": "quasar bridge",
            "uptime": int(time.monotonic() - self._started),
            "caps": {"speaker": True, "screen": False, "follow_up": True},
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
        prev = self.request
        if prev is not None:
            prev.mic_open = False
            if prev.followup and not prev.responded:
                # A wake word spotted during the follow-up window: the cycle's stream ends with
                # it, the new VoiceInput wakes as any wake does. The session's armed follow-up
                # window dies with the cycle.
                self._window = 0
                self._spawn_send(prev.station.send_json(P.stream_control(prev.stream_id)))
            if prev.tts_sid:
                self._spawn_send(self._close_shared(prev))  # its shared TTS stream goes with it
        self._session += 1
        req.session = self._session
        self.request = req
        self._send({"type": "wake", "session": req.session, "word": WAKE_WORD})
        asyncio.get_running_loop().call_later(PREROLL_WAIT_S, self.send_preroll, req)
        logger.info(f"station {self.name}: wake, session {req.session}")

    def attach_followup(self, req: VoiceRequest) -> None:
        """The station's post-answer listening window (the answer's shouldListen): a VoiceInput
        while the session's answer is out. It attaches to the answered session as its follow-up
        cycle: the same session goes on, no new wake and no pre-roll — the microphone audio is
        held until the server's follow-up `listening` releases it."""
        prev = self.request
        if prev is None:
            # The session ended while the VoiceInput was being acked: no open session, so the
            # VoiceInput goes the way any VoiceInput without one does.
            self.wake(req)
            return
        req.session = prev.session
        req.followup = True
        self.request = req
        if self._window == req.session:
            # The window's `listening` already came, before this VoiceInput: the session was
            # taken and the held audio goes out at once.
            req.taken = True
            self._window = 0
            self.send_preroll(req)
        asyncio.get_running_loop().call_later(FOLLOWUP_LISTEN_WAIT_S, self._followup_timeout, req)
        logger.info(f"station {self.name}: follow-up window, session {req.session}")

    def _followup_timeout(self, req: VoiceRequest) -> None:
        """The server's follow-up `listening` never came for the cycle: the audio it held is
        dropped and it ends as an empty window does (_end) — a final empty ASR.Result, its
        stream's close, no answer (firmware 1.56; an AAPI station gets an empty answer instead,
        _end_silently): the station just stops listening."""
        if req is not self.request or req.preroll_sent or req.responded or req.cancelled:
            return
        req.mic_open = False
        req.pcm.clear()
        req.responded = True
        if self._window == req.session:
            self._window = 0  # the closed cycle was the window's last chance
        self.request = None
        if req.station.proto:
            self._spawn_send(self._end_silently(req))  # see _end
            return
        self._spawn_send(self._send_final(req, ""))

    async def _end_silently(self, req: VoiceRequest) -> None:
        """An AAPI listening with no phrase ends with an empty answer and its stream's close, and
        no final ASR.Result, on which quasar would play its no-speech sound."""
        req.responded = req.asr_sent = True
        await req.station.respond(req, "", [])
        await req.station.send_json(P.stream_control(req.stream_id))

    def _spawn_send(self, coro) -> None:
        t = asyncio.create_task(coro)
        t.add_done_callback(_log_task_error)

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
        """Silence until the server stops listening to `req`: its VAD ends a phrase with no
        speech only after the pre-roll and its no-speech wait, both its own settings, and at
        the latest at its longest phrase. A shorter feed leaves the run listening until the
        station's next wake replaces it."""
        silence = bytes([MIC]) + bytes(MIC_FRAME * 2)
        while True:
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
                await self._hold_asr(req, msg.get("text", ""))
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
                if req.followup:
                    self.send_preroll(req)  # the audio the follow-up cycle held goes out
            elif req.responded:
                # A `listening` of an already taken, answered session: its follow-up window is
                # armed — possibly before the station's VoiceInput for it arrives.
                self._window = req.session
        elif value in ("idle", "error"):
            await self._end(req)
        else:
            await self._close_mic(req)

    async def _close_mic(self, req: VoiceRequest) -> None:
        """The microphone of `req` stops (a `state` other than `listening`). A refused wake fails
        its spotter validation and the station's stream closes; a taken one waits for its
        transcript (_finish_asr): quasar takes an empty final ASR.Result for a phrase it did not
        hear and plays no answer after it. _close_mic never closes a follow-up cycle's stream: an
        empty window ends with a final ASR.Result first (_end, _followup_timeout; an AAPI station
        gets an empty answer instead, _end_silently); a wake word
        that supersedes the cycle closes it in Link.wake."""
        if not req.mic_open:
            return
        req.mic_open = False
        if req.taken or req.cancelled or req.followup:
            return
        if req.spotter_validation:
            await req.station.send_json(P.spotter_validation(req.ref, self.device_id, False))
        await req.station.send_json(P.stream_control(req.stream_id))

    async def _finish_asr(self, req: VoiceRequest, text: str) -> None:
        """A taken wake's final ASR.Result with `text`, and the station's stream closes; once."""
        if not req.taken or req.cancelled:
            return
        await self._send_final(req, text)

    async def _send_final(self, req: VoiceRequest, text: str) -> None:
        """The final ASR.Result of `req` with `text`, then its stream's close; once."""
        if req.asr_sent:
            return
        req.asr_sent = True
        await req.station.send_json(P.asr_result(req.ref, text, req.partial_no, True, req.duration_ms))
        await req.station.send_json(P.stream_control(req.stream_id))

    async def _hold_asr(self, req: VoiceRequest, text: str) -> None:
        """Take the transcript but hold its final ASR.Result until the answer plays: quasar
        times a request out 15 s after the final result and plays its error sound over a
        slower answer, so the result goes out together with the VinsResponse, and empty
        partial ASR.Results keep the station listening meanwhile."""
        if not req.taken or req.asr_sent or req.cancelled or req.responded:
            return
        req.transcript = text
        if req.station.proto:
            # AAPI firmware 3.x: an early stream-start AliceResponse quenches the station's
            # waitForResult timer at once; the empty partials below hold only firmware 1.56.
            await req.station.respond_stream_start(req)
            req.stream_started = True
            return
        t = asyncio.create_task(self._hold_loop(req))
        self._holds.add(t)
        t.add_done_callback(self._holds.discard)
        t.add_done_callback(_log_task_error)

    async def _hold_loop(self, req: VoiceRequest) -> None:
        """Empty partial ASR.Results while the answer is being made (see _hold_asr)."""
        while (req is self.request and not req.asr_sent and not req.cancelled
               and not req.responded and self.up):
            await asyncio.sleep(PARTIAL_INTERVAL_S)
            if req is not self.request or req.asr_sent or req.cancelled or not self.up:
                return
            await req.station.send_json(P.asr_result(req.ref, "", req.partial_no, False, req.duration_ms))
            req.partial_no += 1

    async def _end(self, req: VoiceRequest) -> None:
        """The run of `req` is over: its microphone stops and, if no answer went, an empty
        VinsResponse does; a follow-up window, or a taken wake the station heard no speech in,
        with no phrase gets only its final empty ASR.Result and no answer (an AAPI station an empty
        answer instead, _end_silently)."""
        await self._close_mic(req)
        if (req.followup or (req.no_speech and req.taken)) and not req.transcript and not req.responded \
                and req.station.proto:
            # AAPI: quasar plays its no-speech sound on a final empty ASR.Result; an empty answer
            # ends its listening without one, as the device protocol's silent window ends.
            await self._end_silently(req)
        await self._finish_asr(req, req.transcript)
        if (req.followup or (req.no_speech and req.taken)) and not req.transcript and not req.responded:
            # The window ended with no phrase (the station's own close of its stream, the
            # server's idle, the link gone), or a taken wake the station itself heard no speech in
            # (a post-answer listening that came after the run ended wakes anew): on firmware
            # 1.56 it ends the way real Yandex ends an empty window —
            # a final empty ASR.Result, then the stream's close, and no answer. On a bare close
            # quasar's post-answer state machine, still waiting for the final result, plays its
            # error sound.
            req.responded = True
            await self._send_final(req, "")
        if self.request is req:
            self.request = None
        if not req.responded and not req.cancelled:
            req.responded = True
            if req.station.proto and req.stream_started:
                # AAPI: a request that got the early skeleton but no spoken answer (a silent
                # command, an empty/error reply) must close that stream with a final
                # DeferredAliceResponse, not a second full AliceResponse (mirrors _on_play_start).
                await req.station.respond_deferred(req, "", req.directives)
            else:
                await req.station.respond(req, "", req.directives)
        await self._close_shared(req)

    async def _keep_shared(self, req: VoiceRequest, sid: int) -> None:
        """Silence into the request's shared TTS stream, at its own pace, while no play goes into
        it: the firmware's TTS player fails a stream that gets no audio for 10 s
        (onTtsPlayerError), and a slow answer comes later than that after the chime."""
        while req.tts_sid == sid:
            if self._open is None or self._open.request is not req:
                await req.station.send_audio(sid, req.tts_encoder.silence(KEEP_S))
            await asyncio.sleep(KEEP_S)

    async def _close_shared(self, req: VoiceRequest) -> None:
        """The request's shared TTS stream (_on_play_start) ends here, unless a play still goes
        into it: that play's play_end closes it then."""
        if not req.tts_sid or (self._open is not None and self._open.request is req):
            return
        sid, encoder = req.tts_sid, req.tts_encoder
        req.tts_sid, req.tts_encoder = 0, None  # before the awaits: _keep_shared stops
        await req.station.send_audio(sid, encoder.close())
        await req.station.send_json(P.stream_control(sid))

    async def _on_play_start(self, msg: dict) -> None:
        play = self._open = self._plays[msg["id"]] = Play(msg["id"])
        req = self._of(msg.get("session"))
        if req is not None and req.cancelled:
            return  # a cancelled request: reported played (station stays None), not sent out
        shared = req is not None and req.station.proto and not req.responded
        if req is None or req.responded:
            station, ref = (req.station if req is not None else self.stations[-1]), ""
        elif req.reply is None:
            # The chime or a filler while the answer is not ready: a plain TTS.Speak of the
            # request; the VinsResponse waits for the reply's text.
            station, ref = req.station, req.ref
        else:
            station, ref = req.station, req.ref
            await self._finish_asr(req, req.transcript)
            req.responded = True
            if station.proto:
                # AAPI firmware 3.x: the finished answer completes the stream-start skeleton as a
                # DeferredAliceResponse, not a second AliceResponse.
                await station.respond_deferred(req, req.reply, req.directives)
            else:
                await station.respond(req, req.reply, req.directives)
        if shared:
            # AAPI: the firmware plays a single TTS stream of a request, into the skeleton's
            # tts_play_placeholder, and ignores later ones, as the cloud streams one answer. So the
            # chime (which may come before the skeleton), a filler and the answer go into one
            # TTS.Speak, closed after the answer.
            if not req.tts_sid:
                req.tts_sid, req.tts_encoder = station.next_stream_id(), OggOpusStreamEncoder()
                await station.send_json(P.tts_speak(ref, req.tts_sid))
                t = asyncio.create_task(self._keep_shared(req, req.tts_sid))
                self._holds.add(t)
                t.add_done_callback(self._holds.discard)
                t.add_done_callback(_log_task_error)
            play.station, play.sid, play.encoder, play.request = station, req.tts_sid, req.tts_encoder, req
            return
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
        req = play.request
        if play.station is not None and (req is None or req.responded or req.cancelled
                                         or req is not self.request):
            if req is not None:
                req.tts_sid, req.tts_encoder = 0, None  # before the awaits: _keep_shared stops
            await play.station.send_audio(play.sid, play.encoder.close())
            await play.station.send_json(P.stream_control(play.sid))
        # A play of a live request's shared stream before its answer leaves the stream open: the
        # silence of _keep_shared goes on in it and pushes out the page the Ogg muxer holds.
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
            if play.request is not None:
                play.request.tts_sid, play.request.tts_encoder = 0, None  # _keep_shared stops
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
        if req is not None and not req.responded and not req.stream_started:
            # With the request's VinsResponse, as Alice's own volume commands come. After an AAPI
            # skeleton it goes as a Push: the firmware runs no directive of a DeferredAliceResponse.
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
        self.name = link.name
        self.session_id = headers.get("X-UPRX-SSID") or str(uuid.uuid4())
        self.proto = False  # the station sent an AAPI frame: answer in protobuf (firmware 3.x)
        self._send_lock = asyncio.Lock()
        # One send_audio's pages go out together: the silence of a shared TTS stream
        # (Link._keep_shared) and a play's audio come from two tasks into one Ogg stream.
        self._audio_lock = asyncio.Lock()
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
        async with self._audio_lock:
            for chunk in split_ogg_pages(data):
                await self.send_bytes(frame_stream(sid, chunk))

    def next_stream_id(self) -> int:
        sid = self._next_out_stream
        self._next_out_stream += 2
        return sid

    async def respond(self, req: VoiceRequest, text: str, directives: list[dict]) -> None:
        # shouldListen keeps the station listening after the answer: the server's follow-up
        # window (docs/device-protocol.md, "Follow-up") then runs in the same session. An empty
        # answer (a refused wake, an idle/error end) leaves it off.
        await self.send_json(P.vins_response(req.ref, req.request_id, self.session_id, req.sequence_number,
                                             text, speech=text, directives=directives,
                                             should_listen=bool(text)))

    async def respond_stream_start(self, req: VoiceRequest) -> None:
        """AAPI firmware 3.x: an early AliceResponse carrying only the response-stream directives
        and no answer, sent the moment the transcript comes. The firmware times a request out 12 s
        from the end of the phrase waiting for an AliceResponse (its waitForResult); the empty
        partial ASR.Results that hold firmware 1.56 never reach that timer. This matches the real
        cloud's response_stream_start carcass; the finished answer follows as a
        DeferredAliceResponse (respond_deferred)."""
        directives = [
            P.client_action("response_stream_start"),
            dict(TTS_PLACEHOLDER),
            P.client_action("response_stream_placeholder"),
        ]
        msg = P.vins_response(req.ref, req.request_id, self.session_id, req.sequence_number,
                              "", speech="", directives=directives, should_listen=True)
        payload = msg["directive"]["payload"]
        # As the cloud's skeleton: a voice answer is announced though its text is not here yet.
        # With has_voice_response off the firmware drops tts_play_placeholder from the chain,
        # runs response_stream_placeholder at once and ends the turn before the answer comes.
        payload["voice_response"]["has_voice_response"] = True
        payload["response"]["is_streaming"] = True
        await self.send_json(msg)

    async def respond_deferred(self, req: VoiceRequest, text: str, directives: list[dict]) -> None:
        """AAPI firmware 3.x: the finished answer as a single final DeferredAliceResponse chunk
        (IsLast), completing the early stream-start skeleton; the TTS audio follows it as it does
        for firmware 1.56. should_listen is carried in the base response, as respond() carries it
        in the VinsResponse, so the follow-up window still arms."""
        payload = {
            "response_partial_num": 0,
            "is_last": True,
            "request_id": req.request_id,
            "base_response": {"text": text, "should_listen": bool(text), "directives": list(directives or [])},
        }
        await self.send_json(P.directive("Vins", "DeferredAliceResponse", payload, ref=req.ref))

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(_log_task_error)
        return t

    # ------------------------------------------------------------ main loop

    async def run(self) -> None:
        logger.info(f"station {self.name}: connected")
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._on_binary(msg.data)
        finally:
            for t in list(self._tasks):
                t.cancel()
            logger.info(f"station {self.name}: disconnected")

    async def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"station {self.name}: a text frame that is not JSON")
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
                logger.info(f"station {self.name}: speaks protobuf (AAPI)")
                self.proto = True
            try:
                msg = aliceapi.client_to_json(aliceapi.decode_client(data))
            except Exception as e:
                logger.warning(f"station {self.name}: AAPI frame not parsed ({type(e).__name__})")
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
            logger.warning(f"station {self.name}: microphone audio not decoded ({type(e).__name__})")
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
            logger.warning(f"station {self.name}: no link to the server, the request gets an empty answer")
            await self.respond(req, "", [])
            return
        await self.send_json(P.input_start_ack(req.ref))
        prev = self.link.request
        if prev is not None and prev.responded:
            # The station's post-answer listening window (shouldListen of the answer): it
            # attaches to the answered session as its follow-up cycle instead of waking a new
            # one. A wake-word spot ends this stream and wakes through `Link.wake` instead —
            # the spot's SpotterEnd cannot be waited for here, it comes after this VoiceInput.
            self.link.attach_followup(req)
        else:
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
            # A no-speech close (the station's own 5 s of silence) is no cancel: quasar still
            # waits 12 s for the request's end and plays its error sound without one.
            req.no_speech = sc.get("reason") == REASON_NO_SPEECH
            self.link.station_stopped(req, sc.get("reason") == REASON_CANCEL)

    async def _respond_empty(self, ref: str, payload: dict) -> None:
        hdr = payload.get("header") or {}
        await self.send_json(P.vins_response(ref, hdr.get("request_id") or str(uuid.uuid4()), self.session_id,
                                             int(hdr.get("sequence_number") or 0), "", speech=""))


def _log_task_error(t: asyncio.Task) -> None:
    if not t.cancelled() and t.exception() is not None:
        logger.opt(exception=t.exception()).error("station bridge task failed")
