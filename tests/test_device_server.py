"""The device server end to end: a fake device speaks docs/device-protocol.md over a
real WebSocket to a DeviceManager bound on 127.0.0.1, which drives a real Pipeline
(fake STT/LLM/TTS backends) and a real ConfigService."""

import asyncio
import io
import itertools
import json
import wave
from pathlib import Path

import aiohttp
import numpy as np
import pytest
import soxr

from src import captures, device_server
from src.config_service import ConfigService
from src.device_server import (
    CAPTURE_AUDIO, MIC, MODEL, PLAY, CaptureBusyError, DeviceManager, DeviceOfflineError,
)
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


def is_model(msg) -> bool:
    """A `model` message or one of its 0x03 frames."""
    if msg.type == aiohttp.WSMsgType.BINARY:
        return msg.data[0] == MODEL
    return msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)["type"] == "model"


class FakeDevice:
    """The device end of the protocol. Play frames are collected per stream id."""

    def __init__(self, ws):
        self.ws = ws
        self.plays: dict[int, bytes] = {}
        self.frame_times: list[tuple[float, int]] = []  # (arrival, payload bytes)
        self._stream = None

    async def recv(self, timeout=5.0) -> dict:
        """The next text message other than `model`; play frames before it go into
        self.plays, models are skipped (model() reads them)."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            if is_model(msg):
                continue
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

    async def model(self, timeout=5.0) -> tuple[dict, bytes]:
        """The next frame must start a `model`: it and the 0x03 payload that follows."""
        msg = await asyncio.wait_for(self.ws.receive(), timeout)
        assert msg.type == aiohttp.WSMsgType.TEXT, (msg.type, msg.data)
        model = json.loads(msg.data)
        assert model["type"] == "model", model
        data = b""
        while len(data) < model["size"]:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            assert msg.type == aiohttp.WSMsgType.BINARY and msg.data[0] == MODEL, (msg.type, msg.data)
            assert len(msg.data) - 1 <= 4096
            data += msg.data[1:]
        return model, data

    async def send(self, **msg):
        await self.ws.send_str(json.dumps(msg))

    async def mic(self, payload: bytes):
        await self.ws.send_bytes(bytes([MIC]) + payload)

    async def capture_audio(self, offset: int, payload: bytes):
        await self.ws.send_bytes(bytes([CAPTURE_AUDIO]) + offset.to_bytes(4, "little") + payload)

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
        while is_model(msg):
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

    async def connect(self, id="kitchen", *, speaker=True, rate=16000, buffer_ms=4000,
                      capture=None) -> FakeDevice:
        ws = await self.http.ws_connect(f"http://127.0.0.1:{self.manager.port}/device")
        hello = {"type": "hello", "id": id, "fw": "voicepe 0.1.0 (a1b2c3d)",
                 "caps": {"speaker": speaker, "screen": False}, "mic": {"channels": 2}}
        if speaker:
            hello.update(play={"rate": rate, "buffer_ms": buffer_ms}, volume=60)
        if capture is not None:
            hello["capture"] = capture
        await ws.send_str(json.dumps(hello))
        return FakeDevice(ws)

    def device(self, name="kitchen"):
        return self.manager._device(name)


@pytest.fixture
async def srv(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path))
    patch_llm(monkeypatch, reply="готово")
    # The release's models; the wake one spans three 0x03 frames.
    for const, name, size in (("RELEASE_WAKE_MODEL", "zakhar.tflite", 10000),
                              ("RELEASE_VAD_MODEL", "vad.tflite", 3000)):
        (tmp_path / name).write_bytes(bytes(i % 251 for i in range(size)))
        monkeypatch.setattr(device_server, const, str(tmp_path / name))
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


async def test_hello_gets_config_and_both_models(srv):
    dev = await srv.connect()
    assert await dev.recv() == {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30}
    # wake_model / vad_model are empty: the release's files go out.
    for kind, path in (("wake", device_server.RELEASE_WAKE_MODEL), ("vad", device_server.RELEASE_VAD_MODEL)):
        msg, data = await dev.model()
        assert msg == {"type": "model", "kind": kind, "name": path, "size": len(data)}
        assert data == Path(path).read_bytes()
    status = srv.manager.statuses()[0]
    assert (status["name"], status["online"]) == ("kitchen", True)
    assert [v["value"] for v in status["versions"]] == ["voicepe 0.1.0 (a1b2c3d)", None]
    # The panel's Model is the wake model the device reports loaded.
    await dev.send(type="model_loaded", kind="wake", name=device_server.RELEASE_WAKE_MODEL, ok=True)
    await until(lambda: srv.manager.statuses()[0]["versions"][1]["value"] == device_server.RELEASE_WAKE_MODEL)
    # A failed load leaves the device without a wake model, so the panel shows none.
    await dev.send(type="model_loaded", kind="wake", name="models/other.tflite", ok=False)
    await until(lambda: srv.manager.statuses()[0]["versions"][1]["value"] is None)


async def test_models_follow_the_config_and_the_reload_command(srv, tmp_path):
    with pytest.raises(LookupError):
        srv.manager.reload_model("nope")
    with pytest.raises(RuntimeError, match="offline"):
        srv.manager.reload_model("kitchen")
    dev = await srv.connect()
    await dev.recv()  # config
    await dev.model()
    await dev.model()

    # A changed path is sent; the kind whose path stayed is not.
    other = tmp_path / "other.tflite"
    other.write_bytes(b"\x07" * 100)
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "vad_model": str(other)}]}})
    await srv.manager.reconfigure()
    assert await dev.model() == ({"type": "model", "kind": "vad", "name": str(other), "size": 100}, b"\x07" * 100)
    with pytest.raises(asyncio.TimeoutError):
        await dev.model(timeout=0.2)

    # The reload command re-reads and sends both; a file that cannot be read sends nothing.
    missing = str(tmp_path / "missing.tflite")
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "wake_model": missing, "vad_model": str(other)}]}})
    assert srv.manager.reload_model("kitchen") == {"device": "kitchen", "models": {"vad": str(other)}}
    assert (await dev.model())[0] == {"type": "model", "kind": "vad", "name": str(other), "size": 100}


async def test_a_wake_model_name_sends_its_file(srv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "microWakeWord/v28/model/zakhar.tflite"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x05" * 100)
    dev = await srv.connect()
    await dev.recv()  # config
    await dev.model()
    await dev.model()

    # `model` carries the name, the bytes come from the file it names.
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "wake_model": "zakhar/zakhar_v28"}]}})
    await srv.manager.reconfigure()
    assert await dev.model() == ({"type": "model", "kind": "wake", "name": "zakhar/zakhar_v28", "size": 100},
                                 b"\x05" * 100)
    # The name already sent is not sent again.
    await srv.manager.reconfigure()
    with pytest.raises(asyncio.TimeoutError):
        await dev.model(timeout=0.2)
    assert srv.manager.reload_model("kitchen")["models"]["wake"] == "zakhar/zakhar_v28"


def test_wake_models_are_named_and_listed_in_version_order_without_vad(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in ("v10/candidates/angular_s10_hardneg.tflite", "v10/candidates/angular_s08_hardneg.tflite",
                 "v31/model/zakhar.tflite", "v8/model/zakhar.tflite", "v19/zakhar.tflite", "v1/model/zakhar.tflite",
                 "esphome/okay_nabu.tflite", "vad/vad.tflite", "v8/model/notes.txt"):
        (tmp_path / "microWakeWord" / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "microWakeWord" / path).write_bytes(b"")
    assert device_server.list_wake_models() == [
        "esphome/okay_nabu",
        "zakhar/zakhar_v1",
        "zakhar/zakhar_v8",
        "zakhar/zakhar_v10_angular_s08_hardneg",
        "zakhar/zakhar_v10_angular_s10_hardneg",
        "zakhar/zakhar_v19",
        "zakhar/zakhar_v31",
    ]


def test_an_uploaded_wake_model_is_named_by_its_stem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path / "data"))
    assert device_server.wake_model_files() == {}  # no upload directory yet
    path = tmp_path / "data/wake_models/zakhar_v34.tflite"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    assert device_server.wake_model_files() == {"uploaded/zakhar_v34": path.as_posix()}


def test_the_release_wake_model_is_a_model_of_the_repo():
    assert device_server.RELEASE_WAKE_MODEL in device_server.wake_model_files()


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


async def test_announce_reports_the_tts_attempts(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    kitchen = next(c for c in srv.manager.clients if c.name == "kitchen")
    attempts = []
    say = asyncio.create_task(kitchen.announce("молоко", attempts))
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    start = await dev.recv()
    assert (await dev.recv())["type"] == "play_end"
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(say, 2)
    assert [(a["kind"], a["outcome"]) for a in attempts] == [("tts", "spoke")]


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


def capture_status(srv):
    return srv.manager.capture_status("kitchen")


def capture_samples(name) -> np.ndarray:
    """The samples of kitchen's capture `name` as its file holds them."""
    return np.frombuffer(captures.wav_bytes(captures.find("kitchen", name))[44:], dtype="<i2")


async def lose_link(srv, dev):
    await dev.ws.close()
    await until(lambda: not srv.device().online)


async def test_capture_is_written_as_its_audio_arrives(srv):
    srv.svc.apply({"core": {"vad": {"mic_channel": 1}}})
    dev = await srv.connect()
    await dev.recv()  # config
    name = srv.manager.capture_start("kitchen", 1)["name"]
    msg = await dev.recv()
    assert msg == {"type": "capture", "id": srv.device().capture.id, "seconds": 1}
    # "recording" only from the first frame on.
    assert capture_status(srv)["state"] == "starting"
    # The channel the pipeline uses is on disk before the capture ends.
    await dev.capture_audio(0, stereo(700, -700, 500))
    await until(lambda: capture_status(srv)["duration"] == 0.5)
    assert capture_status(srv)["state"] == "recording"
    assert set(capture_samples(name)) == {-700}
    await dev.capture_audio(8000, stereo(700, -700, 500))
    await dev.send(type="audio_end", reason="capture", id=msg["id"])
    await until(lambda: capture_status(srv)["state"] == "done")
    assert capture_status(srv) == {"state": "done", "device": "kitchen", "name": name, "seconds": 1,
                                   "remaining": 0, "duration": 1.0, "gaps": [], "error": None}
    with wave.open(captures.find("kitchen", name).wav) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16000, 16000)


async def test_capture_refusals(srv):
    with pytest.raises(LookupError):
        srv.manager.capture_start("nope", 1)
    with pytest.raises(DeviceOfflineError):
        srv.manager.capture_start("kitchen", 1)
    assert capture_status(srv)["state"] == "idle"
    dev = await srv.connect()
    await dev.recv()  # config
    name = srv.manager.capture_start("kitchen", 1)["name"]
    cid = (await dev.recv())["id"]
    with pytest.raises(CaptureBusyError):
        srv.manager.capture_start("kitchen", 1)
    with pytest.raises(CaptureBusyError):
        srv.manager.capture_delete("kitchen", name)
    # A device in a voice session refuses with `busy`.
    await dev.send(type="audio_end", reason="busy", id=cid)
    await until(lambda: capture_status(srv)["state"] == "error")
    assert "busy" in capture_status(srv)["error"]
    assert srv.manager.capture_delete("kitchen", name) is True
    assert capture_status(srv)["state"] == "idle"


async def test_capture_goes_on_after_a_lost_link(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    name = srv.manager.capture_start("kitchen", 1)["name"]
    cid = (await dev.recv())["id"]
    await dev.capture_audio(0, stereo(100, 0, 300))
    await until(lambda: capture_status(srv)["duration"] == 0.3)
    await lose_link(srv, dev)
    assert capture_status(srv)["state"] == "recording"
    # The device comes back holding the capture: it goes on from the first frame missing.
    dev = await srv.connect(capture=cid)
    await dev.recv()  # config
    assert await dev.recv() == {"type": "capture_resume", "id": cid, "from": 4800}
    await dev.capture_audio(4800, stereo(200, 0, 700))
    await dev.send(type="audio_end", reason="capture", id=cid)
    await until(lambda: capture_status(srv)["state"] == "done")
    assert (capture_status(srv)["duration"], capture_status(srv)["gaps"]) == (1.0, [])
    pcm = capture_samples(name)
    assert set(pcm[:4800]) == {100} and set(pcm[4800:]) == {200}


async def test_a_device_restarted_mid_capture_records_the_time_left(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    name = srv.manager.capture_start("kitchen", 3)["name"]
    first = (await dev.recv())["id"]
    await dev.capture_audio(0, stereo(100, 0, 200))
    await until(lambda: capture_status(srv)["duration"] == 0.2)
    await lose_link(srv, dev)
    srv.device().capture.started -= 1.5  # the restart took 1.5 s
    dev = await srv.connect()  # no capture in hello
    await dev.recv()  # config
    msg = await dev.recv()
    assert msg["type"] == "capture" and msg["id"] != first and 1 < msg["seconds"] <= 1.5
    await dev.capture_audio(0, stereo(200, 0, 1000))
    await dev.send(type="audio_end", reason="capture", id=msg["id"])
    await until(lambda: capture_status(srv)["state"] == "done")
    # The new segment sits where the wall clock puts it, after a gap; the rest is a tail gap.
    (at, hole), tail = captures.find("kitchen", name).gaps
    base = at + hole
    assert at == 3200 and 24000 <= base < 32000
    assert tail == [base + 16000, 48000 - base - 16000]
    pcm = capture_samples(name)
    assert set(pcm[:at]) == {100} and set(pcm[at:base]) == {0} and set(pcm[base:]) == {200}


async def test_a_capture_the_device_never_finishes_keeps_what_came(srv, monkeypatch):
    monkeypatch.setattr(device_server, "CAPTURE_WAIT_MARGIN", 0.2)
    dev = await srv.connect()
    await dev.recv()  # config
    srv.manager.capture_start("kitchen", 1)
    await dev.recv()  # capture
    await dev.capture_audio(0, stereo(100, 0, 400))
    await until(lambda: capture_status(srv)["duration"] == 0.4)
    await lose_link(srv, dev)
    # The device never comes back: the audio that came stays, the rest is a tail gap.
    await until(lambda: capture_status(srv)["state"] == "done", timeout=3.0)
    assert capture_status(srv)["gaps"] == [{"at": 0.4, "seconds": 0.6}]
    # No audio at all ends as an error.
    dev = await srv.connect()
    await dev.recv()  # config
    srv.manager.capture_start("kitchen", 1)
    await dev.recv()  # capture
    await until(lambda: capture_status(srv)["state"] == "error", timeout=3.0)
    assert capture_status(srv)["error"] == "no audio"


async def test_a_device_back_without_the_capture_after_its_end_gets_no_new_one(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    srv.manager.capture_start("kitchen", 1)
    await dev.recv()  # capture
    await dev.capture_audio(0, stereo(100, 0, 1000))
    await lose_link(srv, dev)  # its audio_end went down with the link
    srv.device().capture.started -= 1  # and its time is over
    dev = await srv.connect()  # no capture in hello
    await dev.recv()  # config
    await until(lambda: capture_status(srv)["state"] == "done")
    assert (capture_status(srv)["duration"], capture_status(srv)["gaps"]) == (1.0, [])
    with pytest.raises(asyncio.TimeoutError):
        await dev.recv(timeout=0.3)  # no new `capture`


async def test_a_restarted_server_resumes_the_capture_from_disk(srv, tmp_path):
    dev = await srv.connect()
    await dev.recv()  # config
    name = srv.manager.capture_start("kitchen", 1)["name"]
    cid = (await dev.recv())["id"]
    await dev.capture_audio(0, stereo(100, 0, 400))
    await until(lambda: capture_status(srv)["duration"] == 0.4)
    orphan = captures.Capture.create("kitchen", 5)  # an unfinished capture the device does not hold
    # Another server process on the same data directory; the device reconnects to it.
    other = Server(tmp_path)
    await other.manager.start()
    try:
        dev = await other.connect(capture=cid)
        await dev.recv()  # config
        assert await dev.recv() == {"type": "capture_resume", "id": cid, "from": 6400}
        assert captures.find("kitchen", orphan.name).state == "error"  # finished as is: no audio
        await dev.capture_audio(6400, stereo(200, 0, 600))
        await dev.send(type="audio_end", reason="capture", id=cid)
        await until(lambda: other.device().capture is None)
        cap = captures.find("kitchen", name)
        assert (cap.state, cap.end, cap.gaps) == ("done", 16000, [])
        pcm = capture_samples(name)
        assert set(pcm[:6400]) == {100} and set(pcm[6400:]) == {200}
    finally:
        await other.http.close()
        await other.manager.stop()


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
