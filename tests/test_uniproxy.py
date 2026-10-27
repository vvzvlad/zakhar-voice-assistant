"""The Yandex Station bridge end to end: a fake station speaks the uniproxy JSON protocol to the
Bridge's /uni.ws over a real WebSocket, and the bridge's link speaks docs/device-protocol.md to a
fake zakhar server's /device."""

import asyncio
import json
import struct
import uuid

import aiohttp
import numpy as np
from aiohttp import web

from src.uniproxy.audio import OggOpusStreamEncoder, OpusStreamDecoder, split_ogg_pages
from src.uniproxy.uniproxy import MIC, PLAY, Bridge

TOKEN = "station-oauth-token"


def tone(seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2")


def ogg_pages(data: bytes) -> list[bytes]:
    return split_ogg_pages(data, max_chunk=1)  # one page per chunk, as the station sends them


async def serve(route: str, handler) -> tuple[web.AppRunner, int]:
    app = web.Application()
    app.add_routes([web.get(route, handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, runner.addresses[0][1]


async def grown(buf: bytearray, beyond: int) -> None:
    """Wait until `buf` holds more than `beyond` bytes."""
    async def wait():
        while len(buf) <= beyond:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 5)


class FakeServer:
    """The zakhar server's /device: the link's messages in order, its 0x01 audio in `mic`."""

    def __init__(self):
        self.ws = None
        self.texts: list[str] = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        self.mic = bytearray()
        self.mic_frames: set[int] = set()  # the sizes of the 0x01 frames

    async def handle(self, request):
        self.ws = web.WebSocketResponse()
        await self.ws.prepare(request)
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.texts.append(msg.data)
                await self._inbox.put(json.loads(msg.data))
            elif msg.data[0] == MIC:
                self.mic_frames.add(len(msg.data))
                self.mic += msg.data[1:]
        return self.ws

    async def recv(self) -> dict:
        return await asyncio.wait_for(self._inbox.get(), 5)

    async def send(self, **msg) -> None:
        await self.ws.send_str(json.dumps(msg, ensure_ascii=False))


class FakeStation:
    """The station end of /uni.ws. Binary stream frames are collected per stream id; the empty
    partial ASR results are skipped."""

    def __init__(self, ws):
        self.ws = ws
        self.audio: dict[int, bytearray] = {}

    async def event(self, namespace: str, name: str, payload: dict, **header) -> str:
        mid = str(uuid.uuid4())
        await self.ws.send_str(json.dumps({"event": {
            "header": {"namespace": namespace, "name": name, "messageId": mid, **header},
            "payload": payload}}))
        return mid

    async def mic(self, stream_id: int, data: bytes) -> None:
        for page in ogg_pages(data):
            await self.ws.send_bytes(struct.pack(">I", stream_id) + page)

    async def recv(self) -> dict:
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), 5)
            if msg.type == aiohttp.WSMsgType.BINARY:
                sid = struct.unpack(">I", msg.data[:4])[0]
                self.audio.setdefault(sid, bytearray()).extend(msg.data[4:])
                continue
            data = json.loads(msg.data)
            d = data.get("directive", {})
            if d.get("header", {}).get("name") == "Result" and not d["payload"]["endOfUtt"]:
                continue
            return data

    async def recv_with_partials(self) -> dict:
        """recv() that returns the empty partial ASR results too, instead of skipping them."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), 5)
            if msg.type == aiohttp.WSMsgType.BINARY:
                sid = struct.unpack(">I", msg.data[:4])[0]
                self.audio.setdefault(sid, bytearray()).extend(msg.data[4:])
                continue
            return json.loads(msg.data)

    async def directive(self, name: str) -> dict:
        d = (await self.recv())["directive"]
        assert d["header"]["name"] == name, d["header"]
        return d


async def test_station_session_through_the_bridge():
    server = FakeServer()
    server_runner, server_port = await serve("/device", server.handle)
    bridge = Bridge(lambda: f"ws://127.0.0.1:{server_port}/device")
    bridge_runner, bridge_port = await serve("/uni.ws", bridge.handle)
    http = aiohttp.ClientSession()
    try:
        ws = await http.ws_connect(f"ws://127.0.0.1:{bridge_port}/uni.ws",
                                   headers={"X-UPRX-DEVICE-ID": "LB0000000000000527720000d3336afa",
                                            "X-UPRX-SSID": "ssid1"})
        station = FakeStation(ws)
        hello = await server.recv()
        assert hello["type"] == "hello" and hello["id"] == "YA-d3336afa"
        assert hello["caps"] == {"speaker": True, "screen": False, "follow_up": False}
        assert hello["mic"] == {"channels": 1} and hello["play"]["rate"] == 48000
        assert hello["volume"] == 50

        await station.event("System", "SynchronizeState", {"auth_token": TOKEN})
        await station.directive("SynchronizeStateResponse")
        await station.directive("Push")
        # A TextInput the station sends on its own gets an empty answer and stays off the server.
        await station.event("Vins", "TextInput", {"request": {"event": {"type": "text_input", "text": "x"}}})
        empty = await station.directive("VinsResponse")
        assert empty["payload"]["voice_response"]["has_voice_response"] is False

        # The wake: the microphone is held until SpotterEnd, goes as the pre-roll burst, then live.
        ref = await station.event("Vins", "VoiceInput", {"header": {"request_id": "r1"},
                                                         "enable_spotter_validation": True}, streamId=1)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
        mic = OggOpusStreamEncoder()
        await station.mic(1, mic.add(tone(1.0, 16000), 16000))
        await asyncio.sleep(0.5)
        assert len(server.mic) == 0
        await ws.send_str(json.dumps({"streamcontrol": {"action": 2, "reason": 0, "streamId": 1}}))
        await grown(server.mic, 0)
        await asyncio.sleep(0.2)
        preroll = len(server.mic)
        assert preroll >= 2 * 16000 * 0.5
        await station.mic(1, mic.add(tone(0.5, 16000), 16000))
        await grown(server.mic, preroll)
        assert server.mic_frames == {1 + 2 * 512}

        await server.send(type="play_stop")
        await server.send(type="state", session=1, value="listening")
        assert (await station.directive("Validation"))["payload"]["valid"] == 1
        # The chime goes out before the transcript, as in production: the server plays it once
        # the wake word is found, before STT. It goes as a plain TTS.Speak of the request; the
        # VinsResponse waits for the reply's text.
        await server.send(type="state", session=1, value="transcribing")
        await server.send(type="play_start", id=7, session=1)
        await server.ws.send_bytes(bytes([PLAY]) + tone(0.1, 48000).tobytes())
        await server.send(type="play_end", id=7, session=1)
        chime = await station.directive("Speak")
        assert chime["header"]["refMessageId"] == ref and chime["header"]["streamId"] == 2
        assert (await station.recv())["streamcontrol"]["streamId"] == 2
        assert station.audio[2].startswith(b"OggS")
        assert await server.recv() == {"type": "played", "id": 7, "dry": 0, "gap_ms": 0}
        await server.send(type="transcript", session=1, text="алиса включи свет")
        # The final ASR.Result is held until the answer: quasar gives a request 15 s from the
        # final result and plays its error sound over a slower answer, so the first thing the
        # station gets after the transcript is an empty partial holding it, not the final.
        held = await station.recv_with_partials()
        assert held["directive"]["payload"]["endOfUtt"] is False
        # quasar closes its own microphone stream once it has got its own final result from the
        # station side: that is no cancel, the answer still goes to the station.
        await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 7, "streamId": 1}}))

        # The answer: VinsResponse and TTS.Speak at its play_start, its audio as Ogg/Opus, closed at
        # play_end, and `played` once its duration has passed.
        await server.send(type="reply", session=1, text="Свет включён")
        await server.send(type="state", session=1, value="speaking")
        await server.send(type="play_start", id=8, session=1)
        sent_at = asyncio.get_running_loop().time()
        answer = tone(0.3, 48000).tobytes()
        for i in range(0, len(answer), 4096):
            await server.ws.send_bytes(bytes([PLAY]) + answer[i:i + 4096])
        await server.send(type="play_end", id=8, session=1)
        final = await station.directive("Result")
        assert final["payload"]["endOfUtt"] is True and final["header"]["refMessageId"] == ref
        assert final["payload"]["recognition"][0]["normalized"] == "алиса включи свет"
        assert (await station.recv())["streamcontrol"]["streamId"] == 1
        vins = await station.directive("VinsResponse")
        assert vins["payload"]["header"]["ref_message_id"] == ref
        assert vins["payload"]["voice_response"]["output_speech"]["text"] == "Свет включён"
        speak = await station.directive("Speak")
        assert speak["header"]["refMessageId"] == ref and speak["header"]["streamId"] == 4
        assert (await station.recv())["streamcontrol"]["streamId"] == 4
        assert station.audio[4].startswith(b"OggS")
        assert OpusStreamDecoder().feed(bytes(station.audio[4])).size > 0
        assert await server.recv() == {"type": "played", "id": 8, "dry": 0, "gap_ms": 0}
        assert asyncio.get_running_loop().time() - sent_at >= 0.29
        await server.send(type="state", session=1, value="idle")

        # A refused wake: validation fails, the stream closes, an empty answer. The idle of session
        # 1 sent nothing: its answer went already.
        ref2 = await station.event("Vins", "VoiceInput", {"header": {"request_id": "r2"},
                                                          "enable_spotter_validation": True}, streamId=3)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 2, "word": "Алиса"}
        await server.send(type="state", session=2, value="idle")
        assert (await station.directive("Validation"))["payload"]["valid"] == 0
        assert (await station.recv())["streamcontrol"]["streamId"] == 3
        empty = await station.directive("VinsResponse")
        assert empty["payload"]["header"]["ref_message_id"] == ref2
        assert empty["payload"]["voice_response"]["has_voice_response"] is False

        # Session 3: the station cancels mid-phrase. Its microphone closes before the server
        # end-points, so the bridge feeds the server silence until its VAD ends the session, and
        # the cancel gets no answer back.
        ref3 = await station.event("Vins", "VoiceInput", {"header": {"request_id": "r3"},
                                                          "enable_spotter_validation": True}, streamId=5)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 3, "word": "Алиса"}
        mic3 = OggOpusStreamEncoder()
        await station.mic(5, mic3.add(tone(0.3, 16000), 16000))
        await ws.send_str(json.dumps({"streamcontrol": {"action": 2, "reason": 0, "streamId": 5}}))
        await server.send(type="play_stop")
        await server.send(type="state", session=3, value="listening")
        assert (await station.directive("Validation"))["payload"]["valid"] == 1
        await grown(server.mic, 0)
        before = len(server.mic)
        await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 7, "streamId": 5}}))
        await grown(server.mic, before)  # silence keeps arriving until the server end-points
        await server.send(type="state", session=3, value="transcribing")
        await server.send(type="state", session=3, value="idle")
        assert ref3 not in server.texts  # the ref is the station's; this just keeps it referenced

        assert not any(TOKEN in text for text in server.texts)
        await ws.close()
    finally:
        await http.close()
        await bridge_runner.cleanup()
        await server_runner.cleanup()
