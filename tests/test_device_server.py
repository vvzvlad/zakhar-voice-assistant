"""The device server end to end: a fake device speaks docs/device-protocol.md over a
real WebSocket to a DeviceManager bound on 127.0.0.1, which drives a real Pipeline
(fake STT/LLM/TTS backends) and a real ConfigService."""

import asyncio
import io
import itertools
import json
import re
import wave
from datetime import date
from pathlib import Path

import aiohttp
import numpy as np
import pytest
import soxr

from src import captures, context, device_server
from src.config_service import ConfigService
from src.device_server import (
    CAPTURE_AUDIO, MIC, MODEL, PLAY, CaptureBusyError, DeviceManager, DeviceOfflineError,
)
from src.plugins.base import Deps
from src.plugins.vad.webrtc import WebRtcVadBackend
from src.prompt_store import PromptStore
from src.runs_store import RunsStore
from src.runtime import Runtime
from src.tts import TtsBackend, pcm_mime
from src.wakeword import WakewordVerdict, WakewordVerifier
from tests.test_pipeline import FakeToolHub, FakeVad, patch_llm
from tests.test_pipeline_followup import FakeGate

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


def log_switch(msg) -> bool | None:
    """The `on` of a `log_stream` message, None for any other frame."""
    if msg.type != aiohttp.WSMsgType.TEXT:
        return None
    data = json.loads(msg.data)
    return data["on"] if data["type"] == "log_stream" else None


def boost_value(msg) -> int | None:
    """The `value` of a `volume_boost` message, None for any other frame."""
    if msg.type != aiohttp.WSMsgType.TEXT:
        return None
    data = json.loads(msg.data)
    return data["value"] if data["type"] == "volume_boost" else None


class FakeDevice:
    """The device end of the protocol. Play frames are collected per stream id."""

    def __init__(self, ws):
        self.ws = ws
        self.plays: dict[int, bytes] = {}
        self.frame_times: list[tuple[float, int]] = []  # (arrival, payload bytes)
        self._stream = None

    async def recv(self, timeout=5.0) -> dict:
        """The next text message other than `model`, `log_stream` and `volume_boost`; play
        frames before it go into self.plays, models, log switches and boosts are skipped
        (model(), log_switch() and boost() read them)."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            if is_model(msg) or log_switch(msg) is not None or boost_value(msg) is not None:
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
        """The next frame other than a `log_stream` or a `volume_boost` must start a `model`: it
        and the 0x03 payload that follows."""
        msg = await asyncio.wait_for(self.ws.receive(), timeout)
        while log_switch(msg) is not None or boost_value(msg) is not None:
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

    async def log_switch(self, timeout=5.0) -> bool:
        """The `on` of the next `log_stream`; only models and boosts may come before it."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            if not is_model(msg) and boost_value(msg) is None:
                on = log_switch(msg)
                assert on is not None, (msg.type, msg.data)
                return on

    async def boost(self, timeout=5.0) -> int:
        """The `value` of the next `volume_boost`; only models and log switches may come
        before it."""
        while True:
            msg = await asyncio.wait_for(self.ws.receive(), timeout)
            if not is_model(msg) and log_switch(msg) is None:
                value = boost_value(msg)
                assert value is not None, (msg.type, msg.data)
                return value

    async def closed_with(self) -> int:
        msg = await asyncio.wait_for(self.ws.receive(), 5.0)
        while is_model(msg) or log_switch(msg) is not None or boost_value(msg) is not None:
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
                        "no_speech_timeout_ms": 200, "trim_start_ms": 0, "wake_pause_ms": 0},
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
                      capture=None, follow_up=False) -> FakeDevice:
        ws = await self.http.ws_connect(f"http://127.0.0.1:{self.manager.port}/device")
        hello = {"type": "hello", "id": id, "fw": "voicepe 0.1.0 (a1b2c3d)",
                 "caps": {"speaker": speaker, "screen": False, "follow_up": follow_up},
                 "mic": {"channels": 2}}
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
                              ("RELEASE_VAD_MODEL", "vad.tflite", 3000),
                              ("RELEASE_STOP_MODEL", "stop.tflite", 2000)):
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


async def test_hello_gets_config_and_the_models(srv):
    dev = await srv.connect()
    assert await dev.recv() == {"type": "config", "wake_cutoff": 80, "vad_cutoff": 30, "echo_cancel": 1,
                                "echo_cancel_mode": 2, "echo_cancel_filter": 4, "play_prime_ms": 700,
                                "auto_volume": False}
    assert await dev.boost() == 0  # a new connection has no `ambient` yet
    # wake_model / vad_model are empty: the release's files go out, and the stop model always.
    for kind, path in (("wake", device_server.RELEASE_WAKE_MODEL), ("vad", device_server.RELEASE_VAD_MODEL),
                       ("stop", device_server.RELEASE_STOP_MODEL)):
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
    for _ in range(3):
        await dev.model()

    # A changed path is sent; the kind whose path stayed is not.
    other = tmp_path / "other.tflite"
    other.write_bytes(b"\x07" * 100)
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "vad_model": str(other)}]}})
    await srv.manager.reconfigure()
    assert await dev.model() == ({"type": "model", "kind": "vad", "name": str(other), "size": 100}, b"\x07" * 100)
    with pytest.raises(asyncio.TimeoutError):
        await dev.model(timeout=0.2)

    # The reload command re-reads and sends them all; a file that cannot be read sends nothing.
    missing = str(tmp_path / "missing.tflite")
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "wake_model": missing, "vad_model": str(other)}]}})
    assert srv.manager.reload_model("kitchen") == {
        "device": "kitchen", "models": {"vad": str(other), "stop": device_server.RELEASE_STOP_MODEL}}
    assert (await dev.model())[0] == {"type": "model", "kind": "vad", "name": str(other), "size": 100}


async def test_a_wake_model_name_sends_its_file(srv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "microWakeWord/v28/model/zakhar.tflite"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x05" * 100)
    dev = await srv.connect()
    await dev.recv()  # config
    for _ in range(3):
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


def test_wake_models_are_named_and_listed_in_version_order_without_vad_or_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in ("v10/candidates/angular_s10_hardneg.tflite", "v10/candidates/angular_s08_hardneg.tflite",
                 "v31/model/zakhar.tflite", "v8/model/zakhar.tflite", "v19/zakhar.tflite", "v1/model/zakhar.tflite",
                 "esphome/okay_nabu.tflite", "vad/vad.tflite", "stop/stop.tflite", "v8/model/notes.txt"):
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


async def test_wake_session_end_to_end(srv, tmp_path):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
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
    await dev.send(type="played", id=stream, dry=1, gap_ms=40)
    assert await dev.recv() == {"type": "state", "session": 17, "value": "idle"}
    # The run log keeps how the reply played.
    await until(lambda: srv.rt.runs_store.list())
    [row] = srv.rt.runs_store.list()
    assert (row["play_dry"], row["play_gap_ms"]) == (1, 40)


async def test_new_wake_drops_the_older_session(srv):
    # Session 1 is mid-playback (paced, never reported played) when wake 2 arrives:
    # nothing of session 1 may reach the device after that.
    srv.tts.clips.append(b"\x02\x00" * 32000)  # 2 s
    dev = await srv.connect(buffer_ms=1400)
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


class HeldVerifier(WakewordVerifier):
    """A wake-word verifier with a 100 ms window: it rules on audio whose first sample is
    `sample` once the test resolves verdict(sample)."""

    timeout_ms = 10000
    fail_open = True
    window_ms = 100

    def __init__(self):
        self.calls = []
        self._verdicts = {}

    def verdict(self, sample: int) -> asyncio.Future:
        return self._verdicts.setdefault(sample, asyncio.get_running_loop().create_future())

    async def verify(self, pcm):
        self.calls.append(pcm)
        return await self.verdict(int(np.frombuffer(pcm[:2], dtype="<i2")[0]))


async def answer_playing(srv, dev, seconds: int) -> int:
    """Session 1 runs (no verifier yet) up to the play_start of its `seconds`-long answer;
    returns the stream id."""
    srv.tts.clips.append(b"\x02\x00" * 16000 * seconds)
    await dev.recv()  # config
    await dev.wake(1)
    await dev.phrase()
    got = [await dev.recv() for _ in range(6)]
    assert got[-1]["type"] == "play_start"
    return got[-1]["id"]


async def wake_over(dev, session):
    """A wake and the two 60 ms mic frames that fill the verifier's window."""
    await dev.send(type="wake", session=session, word="Захар")
    await dev.mic(stereo(11, -11, 60))
    await dev.mic(stereo(12, -12, 60))


async def test_a_wake_over_the_answer_drops_it_only_once_confirmed(srv):
    dev = await srv.connect(buffer_ms=1400)
    old = await answer_playing(srv, dev, 10)
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    pipeline = srv.device().pipeline
    on_start, starts, release = pipeline.on_start, [], asyncio.Event()

    async def held_on_start(*args, **kw):
        await on_start(*args, **kw)
        starts.append(kw)
        await release.wait()

    pipeline.on_start = held_on_start
    await wake_over(dev, 2)
    await until(lambda: verifier.calls)
    # Until the verdict the answer plays on: no play_stop, no state. (recv's own timeout is per
    # frame, and the answer's frames keep coming.)
    sent = len(dev.plays[old])
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(dev.recv(), 0.5)
    assert len(dev.plays[old]) > sent

    verifier.verdict(11).set_result(WakewordVerdict(accepted=True, score=0.9))
    assert await dev.recv() == {"type": "play_stop"}
    await dev.send(type="played", id=old, dropped=True)
    await until(lambda: starts)
    assert starts[0]["verified"][0] == 0.9
    # A frame that comes while on_start runs is held too and reaches the run after the others.
    await dev.mic(stereo(13, -13, 60))
    await until(lambda: len(srv.device()._pending_wake.frames) == 3)
    release.set()
    # The arbitration rules once on_start returned and the held wake counts as registered.
    assert await dev.recv() == state(2, "listening")
    got = await dev.finish()
    stream = got[5]["id"]
    assert got == session_msgs(
        2,
        {"type": "play_start", "id": stream, "session": 2},
        {"type": "play_end", "id": stream, "session": 2},
        state(2, "idle"),
    )
    assert np.frombuffer(srv.stt.pcm[-1], dtype="<i2").tolist() == [11] * 960 + [12] * 960 + [13] * 960
    assert len(verifier.calls) == 1  # the confirmed wake is not checked again


async def test_a_rejected_wake_leaves_the_answer_playing_under_the_new_session(srv, tmp_path):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    dev = await srv.connect(buffer_ms=1400)
    old = await answer_playing(srv, dev, 1)
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    verifier.verdict(11).set_result(WakewordVerdict(accepted=False, score=0.0))
    await wake_over(dev, 2)
    # No play_stop: the answer's text and last state, stamped with session 2, reach the device
    # (its mic stops); the answer plays to its end and its run then ends session 2.
    assert [await dev.recv() for _ in range(3)] == [
        {"type": "transcript", "session": 2, "text": "включи свет"},
        {"type": "reply", "session": 2, "text": "готово"},
        state(2, "speaking"),
    ]
    assert await dev.recv() == {"type": "play_end", "id": old, "session": 1}
    assert len(dev.plays[old]) == 32000
    await dev.send(type="played", id=old)
    assert await dev.recv() == state(2, "idle")
    await until(lambda: len(srv.rt.runs_store.list()) == 2)
    [row] = srv.rt.runs_store.list(result="rejected")
    assert (row["reason"], row["wakeword_score"], row["has_audio"]) == ("wakeword_reject", 0.0, 1)


async def test_a_rejected_wake_over_an_announcement_ends_its_session(srv):
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    verifier.verdict(11).set_result(WakewordVerdict(accepted=False, score=0.1))
    srv.tts.clips.append(b"\x02\x00" * 16000)  # 1 s
    dev = await srv.connect(buffer_ms=1400)
    await dev.recv()  # config
    say = asyncio.create_task(srv.manager.announce("kitchen", "молоко"))
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    start = await dev.recv()
    await wake_over(dev, 1)
    assert await dev.recv() == state(1, "idle")
    assert (await dev.recv())["type"] == "play_end"
    assert len(dev.plays[start["id"]]) == 32000
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(say, 2)


async def test_a_wake_with_nothing_to_protect_is_taken_at_once(srv):
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    dev = await srv.connect()
    await dev.recv()  # config
    await wake_over(dev, 1)
    assert await dev.recv() == {"type": "play_stop"}
    await until(lambda: verifier.calls)
    assert srv.device()._pending_wake is None  # the run's own gate checks it
    verifier.verdict(11).set_result(WakewordVerdict(accepted=True, score=0.9))
    assert await dev.recv() == state(1, "listening")


@pytest.mark.parametrize("outcome", ["confirmed", "rejected", "dropped"])
async def test_a_held_wake_keeps_the_arbitration_open_until_its_check_rules(srv, outcome):
    # The kitchen speaker answers and holds a wake when the panel (no speaker) wakes too and
    # its check rules first: no device keeps the phrase until the speaker's check rules.
    srv.svc.apply({"core": {"arbitration": {"window_ms": 100, "cooldown_ms": 300}}})
    speaker = await srv.connect(buffer_ms=1400)
    panel = await srv.connect("panel", speaker=False)
    await panel.recv()  # config
    old = await answer_playing(srv, speaker, 10)
    await asyncio.sleep(0.3)  # past the cooldown of session 1's phrase
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    verifier.verdict(500).set_result(WakewordVerdict(accepted=True, score=0.9))
    await wake_over(speaker, 2)
    await until(lambda: verifier.calls)
    await panel.send(type="wake", session=3, word="Захар")
    assert await panel.recv() == {"type": "play_stop"}
    await panel.mic(stereo(500, -500, 60))
    await panel.mic(stereo(500, -500, 60))
    await until(lambda: len(verifier.calls) == 2)
    with pytest.raises(asyncio.TimeoutError):  # the panel's window runs out, the group stays open
        await panel.recv(timeout=0.3)

    if outcome == "dropped":
        # The speaker's link goes before its check rules: the held wake is dropped and
        # the re-check rules the panel's group without it.
        await lose_link(srv, speaker)
        assert await panel.recv() == state(3, "listening")
        return
    confirmed = outcome == "confirmed"
    verifier.verdict(11).set_result(WakewordVerdict(accepted=confirmed, score=0.9 if confirmed else 0.0))
    if confirmed:
        # The speaker wins; the panel streams its phrase on silently and is let go.
        assert await speaker.recv() == {"type": "play_stop"}
        await speaker.send(type="played", id=old, dropped=True)
        assert await speaker.recv() == state(2, "listening")
        await panel.phrase()
        assert await panel.recv() == state(3, "idle")
    else:
        # The re-check rules the panel's group without it: the panel keeps the phrase.
        assert await panel.recv() == state(3, "listening")
        assert await speaker.recv() == {"type": "transcript", "session": 2, "text": "включи свет"}


async def test_a_confirmed_wake_holds_the_arbitration_until_its_run_is_registered(srv):
    # The speaker's held wake is confirmed and the panel's check rules while the speaker's
    # run is still starting: the panel's group waits for the speaker, which then wins.
    srv.svc.apply({"core": {"arbitration": {"window_ms": 100, "cooldown_ms": 300}}})
    speaker = await srv.connect(buffer_ms=1400)
    panel = await srv.connect("panel", speaker=False)
    await panel.recv()  # config
    old = await answer_playing(srv, speaker, 10)
    await asyncio.sleep(0.3)  # past the cooldown of session 1's phrase
    verifier = srv.rt.wakeword_backend = HeldVerifier()
    pipeline = srv.device().pipeline
    reset_run, release = pipeline._reset_run, asyncio.Event()

    async def slow_reset_run(*args, **kw):
        # An await inside on_start after it cancelled the older run, before the new run
        # registers with the arbiter (closing a stale streaming STT session).
        await release.wait()
        await reset_run(*args, **kw)

    pipeline._reset_run = slow_reset_run
    await wake_over(speaker, 2)
    await until(lambda: verifier.calls)
    await panel.send(type="wake", session=3, word="Захар")
    assert await panel.recv() == {"type": "play_stop"}
    await panel.mic(stereo(500, -500, 60))
    await panel.mic(stereo(500, -500, 60))
    await until(lambda: len(verifier.calls) == 2)
    verifier.verdict(11).set_result(WakewordVerdict(accepted=True, score=0.9))
    assert await speaker.recv() == {"type": "play_stop"}
    verifier.verdict(500).set_result(WakewordVerdict(accepted=True, score=0.9))
    with pytest.raises(asyncio.TimeoutError):  # the speaker's run is not registered yet
        await panel.recv(timeout=0.3)
    release.set()
    await speaker.send(type="played", id=old, dropped=True)
    assert await speaker.recv() == state(2, "listening")
    await panel.phrase()
    assert await panel.recv() == state(3, "idle")


async def test_stop_word_in_a_session_ends_it(srv):
    # Session 1 is mid-playback when the stop word comes: the playback stops, the run is
    # cancelled and the session ends with exactly one `state idle`.
    srv.tts.clips.append(b"\x02\x00" * 32000)  # 2 s
    dev = await srv.connect(buffer_ms=1400)
    await dev.recv()  # config
    await dev.wake(1)
    await dev.phrase()
    got = [await dev.recv() for _ in range(6)]
    assert got[-1]["type"] == "play_start"
    stream = got[-1]["id"]
    run = srv.device().pipeline._run_task

    await dev.send(type="stop", session=1)
    assert await dev.recv() == {"type": "play_stop"}
    sent = len(dev.plays[stream])
    assert await dev.recv() == state(1, "idle")
    await dev.send(type="played", id=stream, dropped=True)
    await until(run.done)
    assert run.cancelled()
    with pytest.raises(asyncio.TimeoutError):
        await dev.recv(timeout=0.5)
    assert len(dev.plays[stream]) == sent


async def test_stop_word_outside_a_session_stops_the_playback_only(srv):
    srv.tts.clips.append(b"\x02\x00" * 32000)  # 2 s
    dev = await srv.connect(buffer_ms=1400)
    await dev.recv()  # config
    say = asyncio.create_task(srv.manager.announce("kitchen", "молоко"))
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    start = await dev.recv()
    assert start["type"] == "play_start"

    await dev.send(type="stop")
    assert await dev.recv() == {"type": "play_stop"}
    await dev.send(type="played", id=start["id"], dropped=True)
    await asyncio.wait_for(say, 2)
    with pytest.raises(asyncio.TimeoutError):
        await dev.recv(timeout=0.3)


async def test_a_reminder_is_a_run_and_a_line_of_the_dialog(srv, tmp_path):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    dev = await srv.connect()
    await dev.recv()  # config
    say = asyncio.create_task(srv.manager.announce("kitchen", "молоко"))
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    start = await dev.recv()
    assert (await dev.recv())["type"] == "play_end"
    await dev.send(type="played", id=start["id"], dry=0, gap_ms=0)
    await asyncio.wait_for(say, 2)

    [row] = srv.rt.runs_store.list()
    run = srv.rt.runs_store.get(row["id"])
    assert (run["device"], run["reason"], run["result"], run["stt_text"], run["llm_text"]) == (
        "kitchen", "reminder", "ok", None, "молоко")
    assert (run["play_dry"], run["play_gap_ms"]) == (0, 0)
    assert [(a["kind"], a["outcome"]) for a in run["attempts"]] == [("tts", "spoke")]
    assert run["t_tts"] is not None
    assert context.load_context(str(tmp_path / "context_kitchen.txt")) == [
        {"role": "user", "content": "[На колонке была произнесена фраза: «молоко»]"}]


async def test_announce_on_an_offline_device_is_an_error_run(srv, tmp_path):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    with pytest.raises(RuntimeError, match="offline"):
        await srv.device().announce("молоко", "say")
    [row] = srv.rt.runs_store.list()
    assert (row["reason"], row["result"]) == ("say", "error")
    assert srv.rt.runs_store.get(row["id"])["error_stage"] == "device"
    assert not (tmp_path / "context_kitchen.txt").exists()


async def test_announce_whose_speech_fails_is_a_tts_error_run(srv, tmp_path, monkeypatch):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    dev = await srv.connect()
    await dev.recv()  # config

    async def down(text, lang="ru"):
        raise RuntimeError("tts down")

    monkeypatch.setattr(srv.tts, "synthesize", down)
    with pytest.raises(Exception, match="tts down"):
        await srv.device().announce("молоко", "say")
    [row] = srv.rt.runs_store.list()
    assert (row["reason"], row["result"]) == ("say", "error")
    assert srv.rt.runs_store.get(row["id"])["error_stage"] == "TTS"
    assert not (tmp_path / "context_kitchen.txt").exists()


async def test_follow_up_listens_after_a_played_answer_only(srv):
    srv.rt.followup_backend = FakeGate(accepted=False, probability=0.1)
    dev = await srv.connect(follow_up=True)
    await dev.recv()  # config
    await dev.wake(1)
    await dev.phrase()
    got = [await dev.recv() for _ in range(7)]
    await dev.send(type="played", id=got[-1]["id"])
    # The answer played out: the session stays open and the microphone reopens.
    assert await dev.recv() == state(1, "listening")
    # A phrase the follow-up check drops ends the session without a transcript.
    await dev.phrase()
    assert await dev.recv() == state(1, "transcribing")
    assert await dev.recv() == state(1, "idle")

    # An answer the device dropped opens no window.
    await dev.wake(2)
    await dev.phrase()
    got = [await dev.recv() for _ in range(7)]
    await dev.send(type="played", id=got[-1]["id"], dropped=True)
    assert await dev.recv() == state(2, "idle")


async def test_run_ends_when_played_never_comes(srv, monkeypatch):
    monkeypatch.setattr(device_server, "PLAYED_WAIT_MARGIN_S", 0.2)
    dev = await srv.connect(buffer_ms=1400)
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


@pytest.mark.parametrize("prime", [None, 300])
async def test_playback_is_paced_to_the_device_buffer(srv, prime):
    # buffer_ms - play_prime_ms (the default 700, or the one the panel set and the last
    # `config` sent) - 500 -> at most 200 ms of audio ahead of real-time playback.
    dev = await srv.connect(buffer_ms=(prime or 700) + 700)
    await dev.recv()  # config
    if prime is not None:
        srv.manager.set_device_control("kitchen", "play_prime_ms", prime)
        await srv.manager.reconfigure()
        assert (await dev.recv())["play_prime_ms"] == prime
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
    # The last frame leaves once 0.6 s of the 0.8 s has played.
    assert 0.35 <= dev.frame_times[-1][0] - t0 < 0.8
    assert not play.done()  # still waiting for `played`
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(play, 2)


async def test_a_smaller_reserve_mid_stream_keeps_the_pacing_of_the_larger(srv):
    # 1400 - 700 - 500 -> 200 ms ahead. The device primed this stream with 700 ms, so a reserve
    # lowered to 0 while it plays must not let the server run 900 ms ahead and overflow it.
    dev = await srv.connect(buffer_ms=1400)
    await dev.recv()  # config

    async def chunks():
        for _ in range(8):
            yield b"\x03\x00" * 1600  # 0.1 s at 16 kHz

    async def lower():
        await asyncio.sleep(0.1)
        srv.manager.set_device_control("kitchen", "play_prime_ms", 0)
        await srv.manager.reconfigure()

    play = asyncio.create_task(srv.device().play(pcm_mime(16000), chunks()))
    start = await dev.recv()
    assert start["type"] == "play_start"
    # Lowered in the background, so the frames are read (and timed) as they arrive.
    lowering = asyncio.create_task(lower())
    assert (await dev.recv())["play_prime_ms"] == 0
    await lowering
    assert (await dev.recv())["type"] == "play_end"
    t0 = dev.frame_times[0][0]
    received = 0
    for t, n in dev.frame_times:
        received += n
        assert received / 32000 - (t - t0) <= 0.2 + 0.05
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(play, 2)

    # The next stream starts on an empty buffer: paced by the reserve 0, 900 ms ahead (once the
    # server's estimate of the first stream's playback has run out too).
    await asyncio.sleep(0.4)
    dev.frame_times.clear()
    play = asyncio.create_task(srv.device().play(pcm_mime(16000), b"\x03\x00" * 19200))  # 1.2 s
    start = await dev.recv()
    assert (await dev.recv())["type"] == "play_end"
    t0 = dev.frame_times[0][0]
    ahead = max(sum(n for _, n in dev.frame_times[:i + 1]) / 32000 - (t - t0)
                for i, (t, _) in enumerate(dev.frame_times))
    assert 0.8 <= ahead <= 0.9 + 0.05
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(play, 2)


async def test_played_reports_how_the_stream_ran_dry(srv):
    # A stream that played out returns the device's dry/gap_ms; a dropped one and a
    # device that does not count them return None for both.
    dev = await srv.connect()
    await dev.recv()  # config
    for reply, stats in (
            ({"dry": 2, "gap_ms": 90}, {"outcome": "played", "dry": 2, "gap_ms": 90}),
            ({}, {"outcome": "played", "dry": None, "gap_ms": None}),
            ({"dropped": True}, {"outcome": "dropped", "dry": None, "gap_ms": None})):
        play = asyncio.create_task(srv.device().play(pcm_mime(16000), b"\x01\x00" * 1600))
        start = await dev.recv()
        await dev.recv()  # play_end
        await dev.send(type="played", id=start["id"], **reply)
        assert await asyncio.wait_for(play, 2) == stats


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
    assert [(c["id"], c["value"], c["min"], c["max"], c["step"]) for c in controls] == [
        ("wake_probability_cutoff", 80, 0, 100, 1), ("vad_cutoff", 30, 0, 100, 1), ("echo_cancel", 1, 0, 3, 1),
        ("echo_cancel_mode", 2, 0, 3, 1), ("echo_cancel_filter", 4, 1, 8, 1), ("play_prime_ms", 700, 0, 2000, 50),
        ("auto_volume_threshold", -60, -120, 0, 1), ("auto_volume_gain", 2.0, 0, 10, 0.1),
        ("speaker_volume", 60, 0, 100, 1),
    ]

    # A cutoff is written to the config; the reconfigure that follows sends `config`.
    srv.manager.set_device_control("kitchen", "wake_probability_cutoff", 55)
    assert srv.svc.core.devices[0].wake_cutoff == 55
    await srv.manager.reconfigure()
    rest = {"echo_cancel": 1, "echo_cancel_mode": 2, "echo_cancel_filter": 4, "play_prime_ms": 700,
            "auto_volume": False}
    assert await dev.recv() == {"type": "config", "wake_cutoff": 55, "vad_cutoff": 30, **rest}
    # So are the echo cancellation settings and the playback reserve, each clamped to its range.
    for control, value, stored in (("echo_cancel", 7, 3), ("echo_cancel_mode", -2, 0), ("echo_cancel_filter", 0, 1),
                                   ("echo_cancel_filter", 9, 8), ("play_prime_ms", 450, 450),
                                   ("play_prime_ms", 2500, 2000)):
        srv.manager.set_device_control("kitchen", control, value)
        assert getattr(srv.svc.core.devices[0], control) == stored
        await srv.manager.reconfigure()
        rest[control] = stored
        assert await dev.recv() == {"type": "config", "wake_cutoff": 55, "vad_cutoff": 30, **rest}

    srv.manager.set_device_control("kitchen", "speaker_volume", 40)
    assert await dev.recv() == {"type": "set_volume", "value": 40}
    await dev.send(type="volume", value=70)
    await until(lambda: srv.manager.device_controls("kitchen")["controls"][8]["value"] == 70)

    srv.manager.set_wake_prob_stream("kitchen", True)
    assert await dev.recv() == {"type": "wake_peak_stream", "on": True}
    await dev.send(type="wake_peak", value=42)
    await until(lambda: srv.manager.wake_prob("kitchen")["value"] == 42)


async def test_auto_volume_boosts_by_the_median_noise_over_the_threshold(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    assert await dev.boost() == 0

    def readings():
        return [r["value"] for r in srv.manager.device_controls("kitchen")["readings"]]

    assert readings() == [None, 0]
    # Off: the noise is kept and shown, the boost stays 0.
    await dev.send(type="ambient", value=-50.0)
    await until(lambda: readings() == [-50.0, 0])
    # On: `config` carries it, then the boost: (-50 - -60) x 2.
    devices = [d.model_dump(mode="json") for d in srv.svc.core.devices]
    srv.svc.apply({"core": {"devices": [{**devices[0], "auto_volume": True}]}})
    await srv.manager.reconfigure()
    assert (await dev.recv())["auto_volume"] is True
    assert await dev.boost() == 20
    # The median of the last four values: the fifth drops the first (-50).
    for value, boost in ((-40.0, 30), (-70.0, 20), (-30.0, 30), (-30.0, 50)):
        await dev.send(type="ambient", value=value)
        assert await dev.boost() == boost
    assert readings() == [-35.0, 50]
    # A microphone reading all zeros sends null (-inf): not a level, the link stays up.
    await dev.send(type="ambient", value=None)
    await dev.send(type="volume", value=41)
    await until(lambda: srv.device().conn.volume == 41)
    assert readings() == [-35.0, 50]
    # The threshold and the gain stay on the server: a change sends the new boost, no `config`.
    srv.manager.set_device_control("kitchen", "auto_volume_threshold", -40)
    await srv.manager.reconfigure()
    assert await dev.boost() == 10
    srv.manager.set_device_control("kitchen", "auto_volume_gain", 3.04)
    assert srv.svc.core.devices[0].auto_volume_gain == 3.0
    await srv.manager.reconfigure()
    assert await dev.boost() == 15
    srv.manager.set_device_control("kitchen", "auto_volume_gain", 12)
    await srv.manager.reconfigure()
    assert await dev.boost() == 50
    # A boost past 100 is 100, and a boost that stays as it is is not sent again.
    await dev.send(type="ambient", value=0.0)  # the median -30: (-30 - -40) x 10
    assert await dev.boost() == 100
    await dev.send(type="ambient", value=0.0)  # the median -15: 250
    with pytest.raises(asyncio.TimeoutError):
        await dev.boost(timeout=0.2)
    assert readings() == [-15.0, 100]
    # Off again: the boost goes back to 0 and the stale noise is dropped.
    devices = [d.model_dump(mode="json") for d in srv.svc.core.devices]
    srv.svc.apply({"core": {"devices": [{**devices[0], "auto_volume": False}]}})
    await srv.manager.reconfigure()
    assert (await dev.recv())["auto_volume"] is False
    assert await dev.boost() == 0
    assert readings() == [None, 0]


async def test_set_volume_to_a_level_or_by_a_change(srv):
    dev = await srv.connect()
    await dev.recv()  # config
    device = srv.device()
    assert device.set_volume(level=30) == 30
    assert await dev.recv() == {"type": "set_volume", "value": 30}
    # A change counts from the volume the device reported last (e.g. turned on the dial).
    await dev.send(type="volume", value=50)
    await until(lambda: device.conn.volume == 50)
    assert srv.manager.set_volume("kitchen", change=-10) == 40
    assert await dev.recv() == {"type": "set_volume", "value": 40}
    assert device.set_volume(change=100) == 100  # clamped
    assert await dev.recv() == {"type": "set_volume", "value": 100}
    for args in ({}, {"level": 10, "change": 5}):
        with pytest.raises(ValueError):
            device.set_volume(**args)
    assert srv.manager.get_volume("kitchen") == 100
    device.conn.volume = None
    with pytest.raises(RuntimeError, match="has not reported its volume"):
        device.set_volume(change=10)
    with pytest.raises(RuntimeError, match="has not reported its volume"):
        srv.manager.get_volume("kitchen")


async def test_set_volume_refuses_offline_unknown_and_speakerless_devices(srv):
    for args in ({"level": 30}, {"change": 10}):
        with pytest.raises(RuntimeError, match="offline"):
            srv.manager.set_volume("kitchen", **args)
    with pytest.raises(RuntimeError, match="offline"):
        srv.manager.get_volume("kitchen")
    with pytest.raises(LookupError, match="unknown device"):
        srv.manager.set_volume("nope", level=30)
    with pytest.raises(LookupError, match="unknown device"):
        srv.manager.get_volume("nope")
    dev = await srv.connect("panel", speaker=False)
    await dev.recv()  # config
    for args in ({"level": 30}, {"change": 10}):
        with pytest.raises(LookupError, match="no speaker"):
            srv.manager.set_volume("panel", **args)
    with pytest.raises(LookupError, match="no speaker"):
        srv.manager.get_volume("panel")


async def test_say_speaks_on_an_online_device_only(srv, tmp_path):
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    with pytest.raises(DeviceOfflineError):
        await srv.manager.say("kitchen", "проверка громкости")
    assert srv.rt.runs_store.list() == []

    dev = await srv.connect()
    await dev.recv()  # config
    say = asyncio.create_task(srv.manager.say("kitchen", "проверка громкости"))
    assert await dev.recv() == {"type": "reply", "text": "проверка громкости"}
    start = await dev.recv()
    assert (await dev.recv())["type"] == "play_end"
    await dev.send(type="played", id=start["id"])
    await asyncio.wait_for(say, 2)
    [row] = srv.rt.runs_store.list()
    assert (row["reason"], row["result"], row["llm_text"]) == ("say", "ok", "проверка громкости")


async def test_set_detect_sends_switch_and_refuses_offline(srv):
    with pytest.raises(RuntimeError, match="offline"):
        srv.manager.set_detect("kitchen", False)
    dev = await srv.connect()
    await dev.recv()  # config
    assert srv.manager.set_detect("kitchen", False) == {"device": "kitchen", "detect": False}
    assert await dev.recv() == {"type": "detect", "on": False}
    srv.manager.set_detect("kitchen", True)
    assert await dev.recv() == {"type": "detect", "on": True}


async def test_the_log_switch_reaches_the_device_and_its_log_the_disk(srv, tmp_path):
    # Every hello gets `log_stream` with the device's setting, off as well, before the models.
    dev = await srv.connect()
    assert (await dev.recv())["type"] == "config"
    assert await dev.log_switch() is False
    # Switched on and off while connected: sent at once; a reconfigure that changes nothing
    # else sends nothing.
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "log": True}]}})
    await srv.manager.reconfigure()
    assert await dev.log_switch() is True
    await srv.manager.reconfigure()
    with pytest.raises(asyncio.TimeoutError):
        await dev.log_switch(timeout=0.2)

    # The device's lines land in today's file, each behind the server's time.
    await dev.send(type="log", text="I (1200) zakhar_link: connected\nW (1300) x: тихо\n")
    await dev.send(type="log", text="I (1400) y: z\n")
    path = tmp_path / "device_logs" / "kitchen" / f"{date.today():%Y-%m-%d}.log"
    await until(lambda: path.exists() and len(path.read_text(encoding="utf-8").splitlines()) == 3)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line[13:] for line in lines] == [
        "I (1200) zakhar_link: connected", "W (1300) x: тихо", "I (1400) y: z"]
    assert all(re.fullmatch(r"\d\d:\d\d:\d\d\.\d{3} ", line[:13]) for line in lines)

    # A new connection gets the setting as it is now.
    again = await srv.connect()
    assert (await again.recv())["type"] == "config"
    assert await again.log_switch() is True
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "log": False}]}})
    await srv.manager.reconfigure()
    assert await again.log_switch() is False


async def test_an_unwritable_device_log_keeps_the_link(srv, monkeypatch):
    tried = []

    def full(*args):
        tried.append(args)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(device_server.device_logs, "append", full)
    dev = await srv.connect()
    assert (await dev.recv())["type"] == "config"
    assert await dev.log_switch() is False
    await dev.send(type="log", text="I (1200) x: y\n")
    await until(lambda: tried)
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "log": True}]}})
    await srv.manager.reconfigure()
    assert await dev.log_switch() is True


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

    plays = []

    async def spy():
        disconnects.append(1)
        # A play that starts while the old connection is wound down goes out after `config`.
        if not plays:
            plays.append(asyncio.create_task(srv.device().play(pcm_mime(16000), b"\x01\x00" * 1600)))
            await asyncio.sleep(0)
        await on_disconnect()

    pipeline.on_disconnect = spy
    second = await srv.connect()
    assert (await second.recv())["type"] == "config"
    start = await second.recv()
    assert start["type"] == "play_start"
    assert (await second.recv())["type"] == "play_end"
    await second.send(type="played", id=start["id"])
    await asyncio.wait_for(plays[0], 2)
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
    # No stop model: it listens only while the device plays.
    assert [(await dev.model())[0]["kind"] for _ in range(2)] == ["wake", "vad"]
    assert srv.manager.reload_model("panel")["models"].keys() == {"wake", "vad"}
    # A stop model sent on connect would be read here first.
    assert [(await dev.model())[0]["kind"] for _ in range(2)] == ["wake", "vad"]
    await dev.wake(5)
    await dev.phrase()
    got = [await dev.recv() for _ in range(6)]
    assert got == session_msgs(5, {"type": "state", "session": 5, "value": "idle"})

    await srv.manager.announce("panel", "молоко")
    assert await dev.recv() == {"type": "reply", "text": "молоко"}
    assert srv.tts.calls == 0
    assert dev.plays == {}


async def test_one_phrase_heard_by_three_devices_runs_on_the_first_speaker(srv, tmp_path):
    # The panel (no speaker) hears the phrase first, then the kitchen speaker, then the
    # hall speaker: only the kitchen's session goes on. The others stream their phrase
    # on silently, are let go at its end and are logged as duplicates with what they
    # heard; the panel is then shown the kitchen's answer, once — not its follow-up.
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    srv.rt.followup_backend = FakeGate(accepted=True)
    speaker = await srv.connect(follow_up=True)
    hall = await srv.connect("hall")
    panel = await srv.connect("panel", speaker=False)
    for dev in (speaker, hall, panel):
        await dev.recv()  # config

    for dev, name, session in ((panel, "panel", 3), (speaker, "kitchen", 8), (hall, "hall", 5)):
        await dev.send(type="wake", session=session, word="Захар")
        assert await dev.recv() == {"type": "play_stop"}
        if dev is not hall:
            await dev.mic(stereo(500, -500, 60))  # the start of its pre-roll
            await until(lambda: srv.device(name).pipeline.preroll_pcm)
    assert await speaker.recv() == state(8, "listening")
    for dev, session in ((panel, 3), (hall, 5)):
        await dev.phrase()
        assert await dev.recv() == state(session, "idle")  # no `listening`, no `transcript`

    await speaker.phrase()
    got = [await speaker.recv() for _ in range(7)]
    await speaker.send(type="played", id=got[5]["id"])
    assert await speaker.recv() == state(8, "listening")  # the follow-up window
    await speaker.phrase()
    got = []
    while not got or got[-1]["type"] != "play_end":
        got.append(await speaker.recv())
    assert {"type": "reply", "session": 8, "text": "готово"} in got
    await speaker.send(type="played", id=got[-1]["id"], dropped=True)
    assert await speaker.recv() == state(8, "idle")

    assert await panel.recv() == {"type": "reply", "text": "готово"}
    for dev in (panel, hall):
        with pytest.raises(asyncio.TimeoutError):
            await dev.recv(timeout=0.3)
    assert len(srv.stt.pcm) == 4
    rows = {row["device"]: row for row in srv.rt.runs_store.list()}
    for name in ("panel", "hall"):
        assert (rows[name]["result"], rows[name]["reason"], rows[name]["stt_text"]) == (
            "rejected", "duplicate", "включи свет")
    assert rows["kitchen"]["result"] == "ok"
