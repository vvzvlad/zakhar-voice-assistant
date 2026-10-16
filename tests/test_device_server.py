"""The device server end to end: a fake device speaks docs/device-protocol.md over a
real WebSocket to a DeviceManager bound on 127.0.0.1, which drives a real Pipeline
(fake STT/LLM/TTS backends) and a real ConfigService."""

import asyncio
import io
import itertools
import json
import wave

import aiohttp
import numpy as np
import pytest
import soxr

from src import device_server
from src.config_service import ConfigService
from src.device_server import MIC, PLAY, CaptureBusyError, CaptureEmptyError, DeviceManager
from src.plugins.base import Deps
from src.plugins.vad.webrtc import WebRtcVadBackend
from src.prompt_store import PromptStore
from src.runtime import Runtime
from src.tts import TtsBackend, pcm_mime
from tests.test_pipeline import FakeToolHub, FakeVad, patch_llm

# 3 speech frames then 6 silence frames (20 ms VAD frames) -> an end-point.
PHRASE_VAD = [True, True, True, False, False, False, False, False, False]


def stereo(ch0: int, ch1: int, ms: int) -> bytes:
    """Interleaved 2-channel 16 kHz s16le mic audio with constant channel values."""
    n = 16 * ms
    return np.column_stack([np.full(n, ch0), np.full(n, ch1)]).astype("<i2").tobytes()


def samples(pcm: bytes) -> set[int]:
    return set(np.frombuffer(pcm, dtype="<i2").tolist())


class RecordingStt:
    def __init__(self):
        self.pcm = []

    async def transcribe(self, pcm):
        self.pcm.append(pcm)
        return "включи свет"


class PcmTts(TtsBackend):
    """Synthesizes raw PCM at 16 kHz: the queued clips first, then a 0.1 s clip."""

    def __init__(self):
        self.clips = []
        self.calls = 0

    async def synthesize(self, text, lang="ru"):
        self.calls += 1
        clip = self.clips.pop(0) if self.clips else b"\x01\x00" * 1600
        return pcm_mime(16000), clip


class FakeDevice:
    """The device end of the protocol. Play frames are collected per stream id."""

    def __init__(self, ws):
        self.ws = ws
        self.plays: dict[int, bytes] = {}
        self.frame_times: list[tuple[float, int]] = []  # (arrival, payload bytes)
        self._stream = None

    async def recv(self, timeout=5.0) -> dict:
        """The next text message; play frames before it go into self.plays."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            if msg.type == aiohttp.WSMsgType.BINARY:
                assert msg.data[0] == PLAY
                self.plays[self._stream] += msg.data[1:]
                self.frame_times.append((asyncio.get_running_loop().time(), len(msg.data) - 1))
                continue
            assert msg.type == aiohttp.WSMsgType.TEXT, (msg.type, msg.data)
            data = json.loads(msg.data)
            if data["type"] == "play_start":
                self._stream = data["id"]
                self.plays[self._stream] = b""
            return data

    async def send(self, **msg):
        await self.ws.send_str(json.dumps(msg))

    async def mic(self, payload: bytes):
        await self.ws.send_bytes(bytes([MIC]) + payload)

    async def phrase(self, ch0=1000, ch1=-1000):
        """Speak a phrase the fake VAD end-points: three 60 ms mic frames."""
        for _ in range(3):
            await self.mic(stereo(ch0, ch1, 60))

    async def wake(self, session):
        await self.send(type="wake", session=session, word="Захар")
        assert await self.recv() == {"type": "play_stop"}
        assert await self.recv() == state(session, "listening")

    async def finish(self) -> list[dict]:
        """Read the session to its end, answering `played` for every stream."""
        got = []
        while not got or got[-1]["type"] != "state" or got[-1]["value"] not in ("idle", "error"):
            got.append(await self.recv())
            if got[-1]["type"] == "play_end":
                await self.send(type="played", id=got[-1]["id"])
        return got

    async def closed_with(self) -> int:
        msg = await asyncio.wait_for(self.ws.receive(), 5.0)
        assert msg.type == aiohttp.WSMsgType.CLOSE, (msg.type, msg.data)
        return msg.data


class Server:
    def __init__(self, tmp_path):
        doc = {
            "version": 1,
            "stt": {"instances": {}},
            "llm": {"instances": {}},
            "tts": {"variants": [{"provider": "yandex"}], "instances": {}},
            "core": {
                "audio": {"host": "127.0.0.1", "port": 0},
                "ack": {"enabled": False},
                "vad": {"min_speech_ms": 40, "silence_ms": 100, "max_utterance_ms": 400,
                        "no_speech_timeout_ms": 200, "trim_start_ms": 0},
                "devices": [{"name": "kitchen"}],
                "chains": [{"name": "main", "stt": "groq", "handler": "openrouter"}],
            },
        }
        self.svc = ConfigService(doc, Deps(http_cloud=None, http_local=None),
                                 path=str(tmp_path / "config.json"))
        prompt = tmp_path / "prompt.md"
        prompt.write_text("PROMPT", encoding="utf-8")
        self.stt = RecordingStt()
        self.tts = PcmTts()
        self.rt = Runtime(
            self.svc,
            vad_backend=WebRtcVadBackend(vad_factory=lambda: FakeVad(PHRASE_VAD)),
            stt_backends={"groq": self.stt},
            llm_backends={"openrouter": object()},
            tts_backends={"yandex": self.tts},
            hub=FakeToolHub(),
            prompt_store=PromptStore(str(tmp_path / "prompts.db"), seed_path=str(prompt)),
        )
        self.manager = self.rt.manager = DeviceManager(self.rt)
        self.http = aiohttp.ClientSession()

    async def connect(self, id="kitchen", *, speaker=True, rate=16000, buffer_ms=4000) -> FakeDevice:
        ws = await self.http.ws_connect(f"http://127.0.0.1:{self.manager.port}/device")
        hello = {"type": "hello", "id": id, "fw": "voicepe 0.1.0 (a1b2c3d)",
                 "model": "zakhar_v28", "caps": {"speaker": speaker, "screen": False},
                 "mic": {"channels": 2}}
        if speaker:
            hello.update(play={"rate": rate, "buffer_ms": buffer_ms}, volume=60)
        await ws.send_str(json.dumps(hello))
        return FakeDevice(ws)

    def device(self, name="kitchen"):
        return self.manager._device(name)


@pytest.fixture
async def srv(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    patch_llm(monkeypatch, reply="готово")
    server = Server(tmp_path)
    await server.manager.start()
    yield server
    await server.http.close()
    await server.manager.stop()


async def until(pred, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


def state(session, value):
    return {"type": "state", "session": session, "value": value}


def session_msgs(session, *rest):
    """The wire messages of one voice session from `transcribing` on."""
    return [
        {"type": "state", "session": session, "value": "transcribing"},
        {"type": "transcript", "session": session, "text": "включи свет"},
        {"type": "state", "session": session, "value": "thinking"},
        {"type": "reply", "session": session, "text": "готово"},
        {"type": "state", "session": session, "value": "speaking"},
        *rest,
    ]


async def test_hello_gets_config_and_shows_online(srv):
    dev = await srv.connect()
    assert await dev.recv() == {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30}
    status = srv.manager.statuses()[0]
    assert (status["name"], status["online"]) == ("kitchen", True)
    assert [v["value"] for v in status["versions"]] == ["voicepe 0.1.0 (a1b2c3d)", "zakhar_v28"]


async def test_wake_session_end_to_end(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    await dev.wake(17)
    await dev.phrase(ch0=1000, ch1=-1000)

    got = [await dev.recv() for _ in range(7)]
    stream = got[5]["id"]
    assert got == session_msgs(
        17,
        {"type": "play_start", "id": stream, "session": 17},
        {"type": "play_end", "id": stream, "session": 17},
    )
    # STT got channel 0 of the interleaved mic audio.
    assert samples(srv.stt.pcm[0]) == {1000}
    # The device rate equals the TTS rate: the clip arrives unchanged.
    assert dev.plays[stream] == b"\x01\x00" * 1600
    # The run waits for `played` before the session ends.
    with pytest.raises(asyncio.TimeoutError):
        await dev.recv(timeout=0.2)
    await dev.send(type="played", id=stream)
    assert await dev.recv() == {"type": "state", "session": 17, "value": "idle"}


async def test_new_wake_drops_the_older_session(srv):
    # Session 1 is mid-playback (paced, never reported played) when wake 2 arrives:
    # nothing of session 1 may reach the device after that.
    srv.tts.clips.append(b"\x02\x00" * 32000)  # 2 s
    dev = await srv.connect(buffer_ms=700)
    await dev.recv()  # config
    await dev.wake(1)
    await dev.phrase()
    got = [await dev.recv() for _ in range(6)]
    assert got[-1]["type"] == "play_start"
    old = got[-1]["id"]

    await dev.send(type="wake", session=2, word="Захар")
    assert await dev.recv() == {"type": "play_stop"}
    await dev.send(type="played", id=old, dropped=True)
    assert await dev.recv() == state(2, "listening")
    sent_before_wake = len(dev.plays[old])
    assert 0 < sent_before_wake < 64000

    await dev.phrase()
    got = [await dev.recv() for _ in range(7)]
    new = got[5]["id"]
    await dev.send(type="played", id=new)
    got.append(await dev.recv())
    assert got == session_msgs(
        2,
        {"type": "play_start", "id": new, "session": 2},
        {"type": "play_end", "id": new, "session": 2},
        {"type": "state", "session": 2, "value": "idle"},
    )
    assert len(dev.plays[old]) == sent_before_wake
    assert dev.plays[new] == b"\x01\x00" * 1600


async def test_wake_drops_an_announcement_already_on_the_wire(srv):
    # The announcement's play_start left before the server saw the wake: play_stop
    # drops it on the device, and neither the announcement nor the session waits on it.
    dev = await srv.connect()
    await dev.recv()  # config
    say = asyncio.create_task(srv.manager.announce("kitchen", "молоко"))
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    start = await dev.recv()
    assert start["type"] == "play_start" and "session" not in start
    assert (await dev.recv())["type"] == "play_end"

    await dev.send(type="wake", session=1, word="Захар")
    assert await dev.recv() == {"type": "play_stop"}
    await dev.send(type="played", id=start["id"], dropped=True)
    assert await dev.recv() == state(1, "listening")
    await asyncio.wait_for(say, 2)

    await dev.phrase()
    got = await dev.finish()
    stream = got[5]["id"]
    assert got == session_msgs(
        1,
        {"type": "play_start", "id": stream, "session": 1},
        {"type": "play_end", "id": stream, "session": 1},
        state(1, "idle"),
    )


async def test_run_ends_when_played_never_comes(srv, monkeypatch):
    monkeypatch.setattr(device_server, "PLAYED_WAIT_MARGIN_S", 0.2)
    dev = await srv.connect(buffer_ms=700)
    await dev.recv()  # config
    await dev.wake(1)
    await dev.phrase()
    got = [await dev.recv() for _ in range(7)]
    assert got[-1]["type"] == "play_end"
    # 0.1 s of audio + 0.7 s of buffer + the 0.2 s margin, then it counts as dropped.
    assert await dev.recv(timeout=3) == state(1, "idle")


async def test_no_speech_sends_no_transcript(srv):
    srv.rt.vad_backend = WebRtcVadBackend(vad_factory=lambda: FakeVad([False]))
    dev = await srv.connect()
    await dev.recv()  # config
    await dev.wake(1)
    for _ in range(4):
        await dev.mic(stereo(0, 0, 60))
    assert await dev.recv() == state(1, "idle")


async def test_playback_is_paced_to_the_device_buffer(srv):
    # buffer_ms 700 -> at most 200 ms of audio ahead of real-time playback.
    dev = await srv.connect(buffer_ms=700)
    await dev.recv()  # config
    pcm = b"\x03\x00" * 12800  # 0.8 s at 16 kHz
    play = asyncio.create_task(srv.device().play(pcm_mime(16000), pcm))
    start = await dev.recv()
    assert start["type"] == "play_start" and "session" not in start
    assert (await dev.recv())["type"] == "play_end"
    assert dev.plays[start["id"]] == pcm

    t0 = dev.frame_times[0][0]
    received = 0
    for t, n in dev.frame_times:
        assert n <= 4096
        received += n
        assert received / 32000 - (t - t0) <= 0.2 + 0.05
    assert dev.frame_times[-1][0] - t0 >= 0.35
    assert not play.done()  # still waiting for `played`
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(play, 2)


async def test_audio_is_converted_to_the_device_rate(srv):
    dev = await srv.connect(rate=48000)
    await dev.recv()  # config
    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(stereo(100, 300, 500))
    for mime, clip in (("audio/wav", wav.getvalue()), (pcm_mime(16000), b"\xc8\x00" * 8000)):
        play = asyncio.create_task(srv.device().play(mime, clip))
        start = await dev.recv()
        await dev.recv()  # play_end
        await dev.send(type="played", id=start["id"])
        await play
        out = np.frombuffer(dev.plays[start["id"]], dtype="<i2")
        # 0.5 s mono at 48 kHz; the stereo WAV is downmixed to 200.
        assert abs(len(out) - 24000) <= 64
        assert np.median(out) == 200


async def test_pcm_split_mid_sample_resamples_like_one_shot(srv):
    dev = await srv.connect(rate=48000)
    await dev.recv()  # config
    x = (8000 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype("<i2")
    raw = x.tobytes()

    async def chunks():
        i = 0
        for size in itertools.cycle((1, 333, 4097, 7, 10001)):
            if i >= len(raw):
                return
            yield raw[i:i + size]
            i += size

    play = asyncio.create_task(srv.device().play(pcm_mime(44100), chunks()))
    start = await dev.recv()
    await dev.recv()  # play_end
    await dev.send(type="played", id=start["id"])
    await play
    out = np.frombuffer(dev.plays[start["id"]], dtype="<i2").astype(int)
    ref = soxr.resample(x, 44100, 48000).astype(int)
    assert len(out) == len(ref)
    assert np.abs(out - ref).max() <= 4


async def test_capture_returns_the_selected_channel_as_wav(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    for mic_channel, expected in ((0, 700), (1, -700)):
        srv.svc.apply({"core": {"vad": {"mic_channel": mic_channel}}})
        job = asyncio.create_task(srv.manager.capture("kitchen", 2))
        assert await dev.recv() == {"type": "capture", "seconds": 2}
        await dev.mic(stereo(700, -700, 100))
        await dev.send(type="audio_end", reason="capture")
        with wave.open(io.BytesIO(await asyncio.wait_for(job, 2))) as w:
            assert (w.getnchannels(), w.getframerate()) == (1, 16000)
            assert samples(w.readframes(w.getnframes())) == {expected}


async def test_capture_busy_empty_and_concurrent(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    # During a session the device refuses with `busy`; its mic frames stay with the session.
    await dev.wake(1)
    job = asyncio.create_task(srv.manager.capture("kitchen", 2))
    assert await dev.recv() == {"type": "capture", "seconds": 2}
    with pytest.raises(CaptureBusyError):
        await srv.manager.capture("kitchen", 2)
    await dev.phrase(ch0=1000)
    await dev.send(type="audio_end", reason="busy")
    with pytest.raises(RuntimeError, match="busy"):
        await asyncio.wait_for(job, 2)
    await dev.finish()
    assert samples(srv.stt.pcm[0]) == {1000}

    job = asyncio.create_task(srv.manager.capture("kitchen", 2))
    await dev.recv()  # capture
    await dev.send(type="audio_end", reason="capture")
    with pytest.raises(CaptureEmptyError):
        await asyncio.wait_for(job, 2)


async def test_controls(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    controls = srv.manager.device_controls("kitchen")["controls"]
    assert [(c["id"], c["value"]) for c in controls] == [
        ("wake_probability_cutoff", 80), ("vad_cutoff", 30), ("speaker_volume", 60),
    ]

    # A cutoff is written to the config; the reconfigure that follows sends `config`.
    srv.manager.set_device_control("kitchen", "wake_probability_cutoff", 55)
    assert srv.svc.core.devices[0].wake_cutoff == 55
    await srv.manager.reconfigure()
    assert await dev.recv() == {"type": "config", "wake_cutoff": 55, "vad_cutoff": 30}

    srv.manager.set_device_control("kitchen", "speaker_volume", 40)
    assert await dev.recv() == {"type": "set_volume", "value": 40}
    await dev.send(type="volume", value=70)
    await until(lambda: srv.manager.device_controls("kitchen")["controls"][2]["value"] == 70)

    srv.manager.set_wake_prob_stream("kitchen", True)
    assert await dev.recv() == {"type": "wake_peak_stream", "on": True}
    await dev.send(type="wake_peak", value=42)
    await until(lambda: srv.manager.wake_prob("kitchen")["value"] == 42)


async def test_set_detect_sends_switch_and_refuses_offline(srv):
    with pytest.raises(RuntimeError, match="offline"):
        srv.manager.set_detect("kitchen", False)
    dev = await srv.connect()
    await dev.recv()  # config
    assert srv.manager.set_detect("kitchen", False) == {"device": "kitchen", "detect": False}
    assert await dev.recv() == {"type": "detect", "on": False}
    srv.manager.set_detect("kitchen", True)
    assert await dev.recv() == {"type": "detect", "on": True}


async def test_unknown_disabled_and_replaced_ids(srv):
    # An unknown id is added to the config as an enabled device.
    hall = await srv.connect("hall")
    assert (await hall.recv())["type"] == "config"
    assert [(d.name, d.enabled) for d in srv.svc.core.devices] == [("kitchen", True), ("hall", True)]

    # A second connection of the same id replaces the first (4001); the pipeline stays.
    first = await srv.connect()
    await first.recv()  # config
    pipeline = srv.device().pipeline
    disconnects = []
    on_disconnect = pipeline.on_disconnect

    async def spy():
        disconnects.append(1)
        await on_disconnect()

    pipeline.on_disconnect = spy
    second = await srv.connect()
    assert (await second.recv())["type"] == "config"
    assert await first.closed_with() == 4001
    await asyncio.sleep(0.1)
    assert srv.device().conn is not None and srv.device().pipeline is pipeline
    assert disconnects == [1]

    # Disabling a connected device closes it (4000); a disabled id is refused (4000).
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "enabled": False}, {"name": "hall"}]}})
    await srv.manager.reconfigure()
    assert await second.closed_with() == 4000
    refused = await srv.connect()
    assert await refused.closed_with() == 4000

    # A device removed from the config is closed too (it is added again if it reconnects).
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "enabled": False}]}})
    await srv.manager.reconfigure()
    assert await hall.closed_with() == aiohttp.WSCloseCode.OK


async def test_text_only_device_gets_no_tts(srv):
    dev = await srv.connect("panel", speaker=False)
    await dev.recv()  # config
    await dev.wake(5)
    await dev.phrase()
    got = [await dev.recv() for _ in range(6)]
    assert got == session_msgs(5, {"type": "state", "session": 5, "value": "idle"})

    await srv.manager.announce("panel", "молоко")
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    assert srv.tts.calls == 0
    assert dev.plays == {}
