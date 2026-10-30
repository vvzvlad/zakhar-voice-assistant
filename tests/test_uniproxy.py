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

from src.uniproxy import aliceapi
from src.uniproxy.audio import OggOpusStreamEncoder, OpusStreamDecoder, parse_stream_frame, split_ogg_pages
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
        assert hello["caps"] == {"speaker": True, "screen": False, "follow_up": True}
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


async def test_silence_goes_on_until_the_server_ends_an_abandoned_request():
    # The station closes its microphone with no speech heard (reason 8). The server ends such
    # a phrase only after the pre-roll and its no-speech wait, longer than any fixed feed: the
    # silence goes on until the server stops listening, then stops.
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await station.event("Vins", "VoiceInput", {"header": {"request_id": "r1"}}, streamId=1)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
        await server.send(type="state", session=1, value="listening")
        await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 8, "streamId": 1}}))
        # Past 3.1 s of silence, and on until the server's state ends it.
        await grown(server.mic, 2 * 16000 * 31 // 10)
        await server.send(type="state", session=1, value="idle")
        await asyncio.sleep(0.2)
        fed = len(server.mic)
        await asyncio.sleep(0.2)
        assert len(server.mic) == fed
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_a_wake_the_station_heard_no_speech_in_ends_as_an_empty_window():
    # The station closes its microphone with no speech heard (reason 8), as its post-answer
    # listening that wakes anew after the run ended does. It still waits for the request's end
    # and plays its error sound 12 s later without one: the request ends as an empty window.
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await station.event("Vins", "VoiceInput", {"header": {"request_id": "r1"}}, streamId=1)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
        await server.send(type="state", session=1, value="listening")
        await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 8, "streamId": 1}}))
        await grown(server.mic, 2 * 16000 // 10)  # the close got through: the silence feed began
        await server.send(type="state", session=1, value="idle")
        await empty_window_end(station, 1)
    finally:
        await hang_up(http, bridge_runner, server_runner)


# ------------------------------------------------------------------ follow-up

async def connect():
    """A bridge against a fake server, with a station connected to it (its hello already read)."""
    server = FakeServer()
    server_runner, server_port = await serve("/device", server.handle)
    bridge = Bridge(lambda: f"ws://127.0.0.1:{server_port}/device")
    bridge_runner, bridge_port = await serve("/uni.ws", bridge.handle)
    http = aiohttp.ClientSession()
    ws = await http.ws_connect(f"ws://127.0.0.1:{bridge_port}/uni.ws",
                               headers={"X-UPRX-DEVICE-ID": "LB0000000000000527720000d3336afa",
                                        "X-UPRX-SSID": "ssid1"})
    station = FakeStation(ws)
    hello = await server.recv()
    assert hello["type"] == "hello" and hello["caps"]["follow_up"] is True
    return server, station, ws, http, bridge_runner, server_runner


async def hang_up(http, bridge_runner, server_runner) -> None:
    await http.close()
    await bridge_runner.cleanup()
    await server_runner.cleanup()


def wakes(server) -> int:
    return len([t for t in server.texts if '"type": "wake"' in t])


async def wake_and_answer(server, station, ws, session: int, phrase: str, answer: str) -> int:
    """One full wake cycle on `session` (its microphone stream is the `session`-th odd stream):
    the wake word spotted, the phrase transcribed, the answer played out and reported. Returns
    the stream id the answer played on."""
    stream_id = session * 2 - 1
    await station.event("Vins", "VoiceInput", {"header": {"request_id": f"r{session}"},
                                               "enable_spotter_validation": True},
                        streamId=stream_id)
    await station.directive("InputStartAck")
    assert await server.recv() == {"type": "wake", "session": session, "word": "Алиса"}
    mic = OggOpusStreamEncoder()
    await station.mic(stream_id, mic.add(tone(0.3, 16000), 16000))
    before = len(server.mic)
    await ws.send_str(json.dumps({"streamcontrol": {"action": 2, "reason": 0, "streamId": stream_id}}))
    await grown(server.mic, before)  # the pre-roll goes out at the SpotterEnd
    await server.send(type="state", session=session, value="listening")
    await station.directive("Validation")
    await server.send(type="state", session=session, value="transcribing")
    await server.send(type="transcript", session=session, text=phrase)
    await station.recv_with_partials()  # the empty partial holds the station while the answer is made
    await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 7, "streamId": stream_id}}))
    await server.send(type="reply", session=session, text=answer)
    await server.send(type="play_start", id=session, session=session)
    await server.ws.send_bytes(bytes([PLAY]) + tone(0.2, 48000).tobytes())
    await server.send(type="play_end", id=session, session=session)
    final = await station.directive("Result")
    assert final["payload"]["endOfUtt"] is True
    assert final["payload"]["recognition"][0]["normalized"] == phrase
    await station.recv()  # the microphone stream's close
    vins = await station.directive("VinsResponse")
    assert vins["payload"]["voice_response"]["output_speech"]["text"] == answer
    speak = await station.directive("Speak")
    sid = speak["header"]["streamId"]
    await station.recv()  # the answer stream's close
    assert await server.recv() == {"type": "played", "id": session, "dry": 0, "gap_ms": 0}
    return sid


async def followup_input(station, stream_id: int, request_id: str) -> None:
    """The station's post-answer listening: a VoiceInput with no wake-word spot behind it."""
    await station.event("Vins", "VoiceInput", {"header": {"request_id": request_id}},
                        streamId=stream_id)
    await station.directive("InputStartAck")


async def answer(server, station, session: int, play_id: int, phrase: str, reply: str) -> tuple[int, dict]:
    """The answer half of a cycle, for a phrase the server has already taken: the transcript, the
    VinsResponse and the TTS stream at its play_start, `played` once its duration has passed.
    Returns the stream id the answer played on and the VinsResponse's voice_response."""
    await server.send(type="state", session=session, value="transcribing")
    await server.send(type="transcript", session=session, text=phrase)
    await station.recv_with_partials()  # the empty partial holds the station while the answer is made
    await server.send(type="reply", session=session, text=reply)
    await server.send(type="play_start", id=play_id, session=session)
    await server.ws.send_bytes(bytes([PLAY]) + tone(0.2, 48000).tobytes())
    await server.send(type="play_end", id=play_id, session=session)
    final = await station.directive("Result")
    assert final["payload"]["endOfUtt"] is True
    assert final["payload"]["recognition"][0]["normalized"] == phrase
    await station.recv()  # the cycle's microphone stream close
    vins = await station.directive("VinsResponse")
    assert vins["payload"]["voice_response"]["output_speech"]["text"] == reply
    speak = await station.directive("Speak")
    sid = speak["header"]["streamId"]
    await station.recv()  # the answer stream's close
    assert await server.recv() == {"type": "played", "id": play_id, "dry": 0, "gap_ms": 0}
    return sid, vins["payload"]["voice_response"]


async def test_the_followup_window_runs_in_the_answered_session():
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        sid1 = await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        assert wakes(server) == 1

        # The server arms the follow-up window once the answer is reported, and the station
        # opens its post-answer listening: a VoiceInput with no wake-word spot behind it.
        await server.send(type="state", session=1, value="listening")
        await followup_input(station, 3, "r-fup1")
        assert wakes(server) == 1  # no second wake: the window attaches to session 1

        # Its microphone audio goes into the same session, with no pre-roll and no SpotterEnd.
        mic2 = OggOpusStreamEncoder()
        before = len(server.mic)
        await station.mic(3, mic2.add(tone(0.4, 16000), 16000))
        await grown(server.mic, before)
        sid2, vr = await answer(server, station, 1, 2, "а теперь выключи", "Свет выключён")
        assert vr["should_listen"] is True and vr["output_speech"]["text"] == "Свет выключён"
        assert sid2 != sid1  # the second answer plays on its own stream
        assert OpusStreamDecoder().feed(bytes(station.audio[sid2])).size > 0

        # The next window's VoiceInput may also come before the server's `listening`: its audio
        # is held until the listening releases it.
        mic3 = OggOpusStreamEncoder()
        await followup_input(station, 5, "r-fup2")
        assert wakes(server) == 1
        before = len(server.mic)
        await station.mic(5, mic3.add(tone(0.4, 16000), 16000))
        await asyncio.sleep(0.3)
        assert len(server.mic) == before  # held: the window's `listening` has not come yet
        await server.send(type="state", session=1, value="listening")
        await grown(server.mic, before)  # released by the listening
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_a_wake_word_in_the_followup_window_wakes_a_new_session():
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        await server.send(type="state", session=1, value="listening")
        await followup_input(station, 3, "r-fup")
        assert wakes(server) == 1
        # The wake word is spotted in the window: its SpotterEnd, then a VoiceInput that wakes.
        await ws.send_str(json.dumps({"streamcontrol": {"action": 2, "reason": 0, "streamId": 3}}))
        await station.event("Vins", "VoiceInput", {"header": {"request_id": "r2"}}, streamId=5)
        await station.directive("InputStartAck")
        assert await server.recv() == {"type": "wake", "session": 2, "word": "Алиса"}
        # The follow-up cycle's stream ends with the wake.
        sc = await station.recv()
        assert sc["streamcontrol"]["streamId"] == 3 and sc["streamcontrol"]["action"] == 0
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_a_followup_window_without_the_servers_listening_closes(monkeypatch):
    monkeypatch.setattr("src.uniproxy.uniproxy.FOLLOWUP_LISTEN_WAIT_S", 0.2)
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        # The follow-up stage stays off: no `state listening` follows the played.
        await followup_input(station, 3, "r-fup")
        assert wakes(server) == 1
        before = len(server.mic)
        mic2 = OggOpusStreamEncoder()
        await station.mic(3, mic2.add(tone(0.4, 16000), 16000))
        await asyncio.sleep(0.1)
        assert len(server.mic) == before  # held while the cycle waits for the listening
        # The wait runs out: the held audio is dropped and the cycle ends as an empty window does.
        await empty_window_end(station, 3)
        assert len(server.mic) == before  # the held audio never went to the server
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def empty_window_end(station, stream_id: int) -> None:
    """A follow-up window that ends with no phrase ends as real Yandex ends one: a final empty
    ASR.Result first (on a bare close quasar's post-answer state machine, still waiting for
    it, plays its error sound), then the stream's close, and no answer."""
    final = await station.recv()
    assert final["directive"]["header"]["name"] == "Result", final
    assert final["directive"]["payload"]["endOfUtt"] is True
    assert final["directive"]["payload"]["recognition"][0]["normalized"] == ""
    sc = await station.recv()
    assert sc["streamcontrol"]["streamId"] == stream_id and sc["streamcontrol"]["action"] == 0
    try:
        msg = await asyncio.wait_for(station.recv(), 0.5)
    except asyncio.TimeoutError:
        return
    raise AssertionError(f"the empty window got a message after its close: {msg}")


async def test_the_followup_window_ended_idle_closes_its_stream():
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        await server.send(type="state", session=1, value="listening")
        await followup_input(station, 3, "r-fup")
        # No speech in the window: the server ends the session with idle.
        await server.send(type="state", session=1, value="idle")
        await empty_window_end(station, 3)
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_the_followup_window_closed_by_the_station_ends_with_a_final_result():
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        await server.send(type="state", session=1, value="listening")
        await followup_input(station, 3, "r-fup")
        # The station's own end of the window: no speech heard (reason 8), then the server's idle.
        await ws.send_str(json.dumps({"streamcontrol": {"action": 0, "reason": 8, "streamId": 3}}))
        await server.send(type="state", session=1, value="idle")
        await empty_window_end(station, 3)
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_the_followup_window_ended_before_its_listening_gets_no_bare_close():
    server, station, ws, http, bridge_runner, server_runner = await connect()
    try:
        await wake_and_answer(server, station, ws, 1, "алиса включи свет", "Свет включён")
        # The window's VoiceInput comes first and the session ends before its `listening`.
        await followup_input(station, 3, "r-fup")
        await server.send(type="state", session=1, value="idle")
        await empty_window_end(station, 3)
    finally:
        await hang_up(http, bridge_runner, server_runner)


# ------------------------------------------------------------------ AAPI (firmware 3.x)


async def wait_for_proto(ws, kind: str, deadline_s: float = 3.0) -> dict:
    """The next server AAPI event of `kind` (e.g. "AliceResponse"), skipping other events, the
    stream-control frames and the audio stream frames. Fails if none comes within the window."""
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline_s
    while True:
        remaining = end - loop.time()
        if remaining <= 0:
            raise AssertionError(f"no server {kind} within {deadline_s}s")
        try:
            msg = await asyncio.wait_for(ws.receive(), remaining)
        except asyncio.TimeoutError:
            raise AssertionError(f"no server {kind} within {deadline_s}s")
        if msg.type == aiohttp.WSMsgType.BINARY and msg.data.startswith(aliceapi.MAGIC):
            event = aliceapi.decode_server(msg.data).get("Event")
            if event and kind in event:
                return event


async def proto_frames(ws, seconds: float = 1.5) -> list[dict]:
    """Every server AAPI message (events and stream controls) within `seconds`."""
    out = []
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds
    while (remaining := end - loop.time()) > 0:
        try:
            msg = await asyncio.wait_for(ws.receive(), remaining)
        except asyncio.TimeoutError:
            break
        if msg.type == aiohttp.WSMsgType.BINARY and msg.data.startswith(aliceapi.MAGIC):
            out.append(aliceapi.decode_server(msg.data))
    return out


async def aapi_wake(server, ws, text: str) -> None:
    """An AAPI VoiceInput the server takes and transcribes as `text`."""
    await ws.send_bytes(aliceapi.encode_client({"Event": {
        "Header": {"MessageId": "ref1", "StreamId": 1},
        "VoiceInput": {"Header": {"RequestId": "r1", "SequenceNumber": 0},
                       "EnableSpotterValidation": True,
                       "Request": {"Event": {"Type": "voice_input"}}}}}))
    assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
    await server.send(type="state", session=1, value="listening")
    await server.send(type="transcript", session=1, text=text)


async def play(server, play_id: int) -> None:
    await server.send(type="play_start", id=play_id, session=1)
    await server.ws.send_bytes(bytes([PLAY]) + tone(0.1, 48000).tobytes())
    await server.send(type="play_end", id=play_id, session=1)


async def test_an_aapi_station_gets_an_early_skeleton_response_before_the_reply():
    """Firmware 3.x (AAPI): the station times its waitForResult out 12 s from the end of the
    phrase, waiting for an AliceResponse; the empty partial ASR results do not reach that timer.
    So on a slow answer the bridge sends an early stream-start AliceResponse the moment the
    transcript comes, before the reply is ready, and delivers the finished answer as a
    DeferredAliceResponse."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_wake(server, ws, "расскажи зачем керамику обжигают")

        # Before any reply exists: the early stream-start AliceResponse, with the directives that
        # quench waitForResult and no answer voice yet.
        event = await wait_for_proto(ws, "AliceResponse")
        names = [d["Name"] for d in event["AliceResponse"]["Response"]["Directives"]]
        assert "response_stream_start" in names
        assert "tts_play_placeholder" in names
        assert "response_stream_placeholder" in names
        # A voice answer is announced and the response marked streaming, as in the cloud's
        # skeleton: otherwise the firmware drops tts_play_placeholder and ends the turn at once.
        assert event["AliceResponse"]["VoiceResponse"]["HasVoiceResponse"] is True
        assert event["AliceResponse"]["Response"]["IsStreaming"] is True

        # Only now does the reply arrive and play: the finished answer is a DeferredAliceResponse,
        # IsLast set, carrying the text; the TTS audio follows as for firmware 1.56.
        await server.send(type="reply", session=1, text="Чтобы спечь черепок и закрепить форму.")
        await play(server, 8)
        deferred = (await wait_for_proto(ws, "DeferredAliceResponse"))["DeferredAliceResponse"]
        assert deferred["IsLast"] is True
        assert deferred["BaseResponse"]["Text"] == "Чтобы спечь черепок и закрепить форму."
        assert deferred["BaseResponse"]["ShouldListen"] is True
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_an_aapi_station_gets_the_filler_and_the_answer_in_one_tts_stream():
    """Firmware 3.x (AAPI): the station plays only a request's first TTS stream, into the
    skeleton's tts_play_placeholder, and drops the later ones. So a filler and the answer go into
    one TTS.Speak, closed only after the answer."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_wake(server, ws, "какая погода на выходных")
        await wait_for_proto(ws, "AliceResponse")
        await play(server, 7)  # a filler while the answer is not ready
        await server.send(type="reply", session=1, text="В субботу пасмурно.")
        await play(server, 8)

        frames = await proto_frames(ws)
        speaks = [m["Event"]["Header"]["StreamId"] for m in frames if "TtsSpeak" in m.get("Event", {})]
        closes = [m["StreamControl"]["StreamId"] for m in frames if "StreamControl" in m]
        assert len(speaks) == 1
        assert closes.count(speaks[0]) == 1
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_an_aapi_tts_stream_gets_silence_while_the_answer_is_not_ready():
    """Firmware 3.x (AAPI): the TTS player fails a stream that gets no audio for 10 s, and a slow
    answer comes later than that after the chime. So between the plays the shared stream gets
    silence at its own pace."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_wake(server, ws, "какая погода на выходных")
        await wait_for_proto(ws, "AliceResponse")
        await play(server, 7)  # the chime; the answer is not ready yet
        sid, audio = None, 0
        loop = asyncio.get_running_loop()
        end = loop.time() + 1.5
        while (remaining := end - loop.time()) > 0:
            try:
                msg = await asyncio.wait_for(ws.receive(), remaining)
            except asyncio.TimeoutError:
                break
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            if msg.data.startswith(aliceapi.MAGIC):
                event = aliceapi.decode_server(msg.data).get("Event", {})
                if "TtsSpeak" in event:
                    sid = int(event["Header"]["StreamId"])  # a uint64: a string in the dict
            elif sid is not None and parse_stream_frame(msg.data)[0] == sid:
                audio += 1
        # 0.1 s of chime is one Ogg page at most; the rest is the silence that keeps the stream.
        assert audio >= 5
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_an_aapi_listening_with_no_speech_ends_with_an_empty_answer_not_a_final_result():
    """Firmware 3.x (AAPI): on a final empty ASR.Result quasar plays its no-speech sound, and a
    listening with no phrase ends without a sound. So such a request (here the post-answer
    listening that woke anew, which the station closes itself with no speech heard) ends with an
    empty AliceResponse and no final empty ASR.Result."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await ws.send_bytes(aliceapi.encode_client({"Event": {
            "Header": {"MessageId": "ref1", "StreamId": 1},
            "VoiceInput": {"Header": {"RequestId": "r1", "SequenceNumber": 0},
                           "Request": {"Event": {"Type": "voice_input"}}}}}))
        assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
        await server.send(type="state", session=1, value="listening")
        await ws.send_bytes(aliceapi.encode_client(
            {"StreamControl": {"StreamId": 1, "Close": {"Reason": "NO_SPEECH"}}}))
        await grown(server.mic, 2 * 16000 // 10)  # the close got through: the silence feed began
        await server.send(type="state", session=1, value="idle")

        frames = await proto_frames(ws)
        finals = [m for m in frames if m.get("Event", {}).get("AsrResult", {}).get("EndOfUtt")]
        answers = [m for m in frames if "AliceResponse" in m.get("Event", {})]
        assert not finals
        assert len(answers) == 1
        assert not answers[0]["Event"]["AliceResponse"].get("VoiceResponse", {}).get("ShouldListen")
        assert any(m.get("StreamControl", {}).get("StreamId") in (1, "1") for m in frames)
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def aapi_answered(server, ws) -> None:
    """An AAPI request answered and played; the server got its `played`."""
    await aapi_wake(server, ws, "алиса включи свет")
    await wait_for_proto(ws, "AliceResponse")
    await server.send(type="reply", session=1, text="Свет включён.")
    await play(server, 8)
    await wait_for_proto(ws, "DeferredAliceResponse")
    assert (await server.recv())["type"] == "played"


async def aapi_followup_input(ws, stream_id: int = 3) -> None:
    """The station's post-answer listening: an AAPI VoiceInput on its next stream."""
    await ws.send_bytes(aliceapi.encode_client({"Event": {
        "Header": {"MessageId": "ref-fup", "StreamId": stream_id},
        "VoiceInput": {"Header": {"RequestId": "r-fup", "SequenceNumber": 1},
                       "Request": {"Event": {"Type": "voice_input"}}}}}))


def assert_silent_end(frames: list[dict], stream_id: int) -> None:
    """An AAPI listening with no phrase ends with one empty answer and its stream's close, and
    no final ASR.Result, on which quasar would play its no-speech sound."""
    assert not [m for m in frames if m.get("Event", {}).get("AsrResult", {}).get("EndOfUtt")]
    answers = [m["Event"]["AliceResponse"] for m in frames if "AliceResponse" in m.get("Event", {})]
    assert len(answers) == 1
    assert not answers[0].get("VoiceResponse", {}).get("ShouldListen")
    assert [int(m["StreamControl"]["StreamId"]) for m in frames if "StreamControl" in m].count(stream_id) == 1


async def test_an_aapi_followup_window_with_no_speech_ends_without_a_sound():
    """Firmware 3.x (AAPI): the post-answer window the server opened ends on silence (its idle)
    without a sound, as the device protocol has it."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_answered(server, ws)
        await server.send(type="state", session=1, value="listening")
        await aapi_followup_input(ws)
        await asyncio.sleep(0.2)
        await server.send(type="state", session=1, value="idle")
        assert_silent_end(await proto_frames(ws), 3)
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_an_aapi_followup_window_the_server_never_opens_ends_without_a_sound(monkeypatch):
    """Firmware 3.x (AAPI): a post-answer window whose follow-up `listening` never comes (the
    server's follow-up stage off) ends at the bridge's wait without a sound."""
    monkeypatch.setattr("src.uniproxy.uniproxy.FOLLOWUP_LISTEN_WAIT_S", 0.2)
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_answered(server, ws)
        await aapi_followup_input(ws)
        assert_silent_end(await proto_frames(ws, 1.0), 3)
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_a_wake_the_station_heard_no_speech_in_still_gets_the_answer_the_server_heard():
    """The station closes its microphone hearing no speech (reason 8), but the server heard a
    phrase: that close is no cancel, the phrase gets its answer."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await ws.send_bytes(aliceapi.encode_client({"Event": {
            "Header": {"MessageId": "ref1", "StreamId": 1},
            "VoiceInput": {"Header": {"RequestId": "r1", "SequenceNumber": 0},
                           "Request": {"Event": {"Type": "voice_input"}}}}}))
        assert await server.recv() == {"type": "wake", "session": 1, "word": "Алиса"}
        await server.send(type="state", session=1, value="listening")
        await ws.send_bytes(aliceapi.encode_client(
            {"StreamControl": {"StreamId": 1, "Close": {"Reason": "NO_SPEECH"}}}))
        await grown(server.mic, 2 * 16000 // 10)  # the close got through: the silence feed began
        await server.send(type="state", session=1, value="transcribing")
        await server.send(type="transcript", session=1, text="алиса привет")
        await wait_for_proto(ws, "AliceResponse")
        await server.send(type="reply", session=1, text="Привет.")
        await play(server, 8)
        deferred = (await wait_for_proto(ws, "DeferredAliceResponse"))["DeferredAliceResponse"]
        assert deferred["BaseResponse"]["Text"] == "Привет."
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)


async def test_an_aapi_request_without_an_answer_closes_its_skeleton_and_tts_stream():
    """Firmware 3.x (AAPI): a request whose run ends with no answer after its skeleton and chime
    (a silent command, an error) gets a final empty DeferredAliceResponse, and its shared TTS
    stream closes once."""
    server, _, ws, http, bridge_runner, server_runner = await connect()
    try:
        await aapi_wake(server, ws, "выключи свет")
        await wait_for_proto(ws, "AliceResponse")
        await play(server, 7)  # the end-of-phrase chime
        await server.send(type="state", session=1, value="idle")

        frames = await proto_frames(ws)
        deferred = [m["Event"]["DeferredAliceResponse"] for m in frames
                    if "DeferredAliceResponse" in m.get("Event", {})]
        assert len(deferred) == 1
        assert deferred[0]["IsLast"] is True
        assert not deferred[0]["BaseResponse"].get("Text")
        speaks = [m["Event"]["Header"]["StreamId"] for m in frames if "TtsSpeak" in m.get("Event", {})]
        closes = [m["StreamControl"]["StreamId"] for m in frames if "StreamControl" in m]
        assert len(speaks) == 1
        assert closes.count(speaks[0]) == 1
        await ws.close()
    finally:
        await hang_up(http, bridge_runner, server_runner)
