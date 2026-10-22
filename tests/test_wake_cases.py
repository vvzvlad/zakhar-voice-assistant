"""Replays of the multi-device wake cases of docs/wake-cases.md, from what the run log,
the recordings and the server log kept of them (tests/fixtures/wake_cases/<case>.json).

A fixture group names the devices online, what each was doing before the group (idle,
answering, announcing, listening in a follow-up window), when each wake reached the server
relative to the group's first wake, each run's recording and recognised text, and the
outcome the doc expects. The replay drives the real DeviceManager, Pipeline and WakeArbiter
through fake WebSocket devices: each device streams its recording as its mic — the 2 s
pre-roll at its wake, then the phrase as it was spoken — so each pre-roll's SNR is the real
one; the STT answers each phrase with the text recognised in its recording; the LLM answers
«ответ <device>». Time runs `scale` times the real one, WINDOW_MS and COOLDOWN_MS with it.

The VAD is a stand-in: a frame is speech unless it is digital silence, and the replay sends
silence after each recording, so a phrase ends where its recording ends, where production's
VAD ended it. Once the group's last logged event has happened, the rest of the audio goes at
once: the outcome no longer depends on its pace."""

import asyncio
import json
import wave
from pathlib import Path

import aiohttp
import numpy as np
import pytest
from loguru import logger

from src.plugins.vad.webrtc import WebRtcVadBackend
from src.runs_store import RunsStore
from src.voice_run import FOLLOWUP_SKIP_MS, PREROLL_MS
from src.wake_arbiter import COOLDOWN_MS, WINDOW_MS
from tests.test_device_server import FakeDevice, srv, until  # noqa: F401  (srv is a fixture)
from tests.test_pipeline import make_llm_result
from tests.test_pipeline_followup import FakeGate
from tests.test_pipeline_wakeword_gate import FakeWakewordVerifier

CASES = Path(__file__).parent / "fixtures" / "wake_cases"
# Replay seconds per real second.
SCALE = 0.25
CHUNK_MS = 60
# Production's core.vad, except the pause: each recording already ends with the pause that
# ended it, and the replay's silence after it ends the phrase within 100 ms.
REPLAY_VAD = {"silence_ms": 100, "min_speech_ms": 200, "max_utterance_ms": 13000,
              "no_speech_timeout_ms": 4000, "trim_start_ms": 0}
# The phrase that makes the speaker answer, or open a follow-up window, before a group: the
# session's real earlier phrases were not recorded.
STAND_IN_MS = 2500
STAND_IN_TEXT = "захар который час"
STAND_IN_KEY = np.full(1600, 1000, "<i2").tobytes()  # 100 ms of the stand-in phrase's samples
# An answer or announcement that plays through the whole replay.
LONG_CLIP = b"\x02\x00" * 16000 * 30


def ms_bytes(ms: int, channels: int) -> int:
    return ms * 16 * 2 * channels


class SilenceIsZeros:
    """The VAD double: a frame is speech unless it is digital silence."""

    def is_speech(self, frame, rate):
        return any(frame)


class RecordedStt:
    """Answers a phrase with the text recognised in the recording it came from, known by
    100 ms of its channel 0 right after the pre-roll; a stand-in phrase gets STAND_IN_TEXT."""

    def __init__(self):
        self.texts = {}

    async def transcribe(self, pcm):
        text = next((text for key, text in self.texts.items() if key in pcm), None)
        if text is not None:
            return text
        # Only the stand-in phrase (constant samples) gets the stand-in text: a recording whose
        # key was lost would otherwise pass the wake-word check with it.
        if STAND_IN_KEY in pcm:
            return STAND_IN_TEXT
        pytest.fail("the STT got a phrase from no known recording")


class DeviceNamedLlm:
    def __init__(self, backend, hub, cfg):
        pass

    async def respond(self, req, *, on_filler=None, has_next=False):
        return make_llm_result(f"ответ {req.device}")


@pytest.fixture
def server(srv, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr("src.pipeline.PREROLL_MS", PREROLL_MS)  # srv drops it
    monkeypatch.setattr("src.llm.LlmStage", DeviceNamedLlm)
    # Each LLM request names the wake word through wake_model_files, which globs
    # microWakeWord/: a checkout holding the training data takes tens of milliseconds of the
    # event loop for it, and the open case's late wake has 138 ms to spare in the replay (69 ms
    # in real time).
    monkeypatch.setattr("src.device_server.wake_model_files", lambda: {})
    srv.svc.apply({"core": {"vad": REPLAY_VAD}})
    srv.rt.vad_backend = WebRtcVadBackend(vad_factory=SilenceIsZeros)
    srv.rt.stt_backends["groq"] = RecordedStt()
    # A wake over the speaker's answer waits for this check; production's had confirmed it.
    srv.rt.wakeword_backend = FakeWakewordVerifier(accepted=True, score=1.0, window_ms=2500)
    srv.rt.followup_backend = FakeGate(window_ms=5000)
    srv.rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    return srv


def load(slug: str) -> dict:
    return json.loads((CASES / f"{slug}.json").read_text(encoding="utf-8"))


def recording(name: str) -> bytes:
    with wave.open(str(CASES / name)) as w:
        return w.readframes(w.getnframes())


def stt_key(audio: bytes, channels: int) -> bytes:
    """100 ms of the recording's channel 0 right after its pre-roll (RecordedStt)."""
    channel0 = np.frombuffer(audio, "<i2")[::channels].tobytes()
    start = ms_bytes(PREROLL_MS, 1)
    return channel0[start:start + ms_bytes(100, 1)]


async def connect(srv, device_id: str, spec: dict) -> FakeDevice:
    ws = await srv.http.ws_connect(f"http://127.0.0.1:{srv.manager.port}/device")
    hello = {"type": "hello", "id": device_id, "fw": "voicepe 0.1.0 (a1b2c3d)", "uptime": 0,
             "caps": {"speaker": spec["speaker"], "screen": False, "follow_up": spec["follow_up"]},
             "mic": {"channels": spec["mic_channels"]}}
    if spec["speaker"]:
        hello.update(play={"rate": 16000, "buffer_ms": 4000}, volume=60)
    await ws.send_str(json.dumps(hello))
    return FakeDevice(ws)


async def listen(dev: FakeDevice, got: list) -> None:
    """Collect the device's text messages, reporting every stream played to its end. Reads
    the socket without FakeDevice.recv's wait_for, which can swallow the cancellation."""
    async for frame in dev.ws:
        if frame.type == aiohttp.WSMsgType.TEXT:
            got.append(msg := json.loads(frame.data))
            if msg["type"] == "play_end":
                await dev.send(type="played", id=msg["id"], dry=0, gap_ms=0)


async def stream(dev: FakeDevice, audio: bytes, channels: int, burst: int, scale: float,
                 done: asyncio.Event) -> None:
    """Send `audio` (interleaved mic samples) in 60 ms chunks: its first `burst` bytes at
    once, the rest as it was spoken, `scale` times faster, and all at once after `done`."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    chunk = ms_bytes(CHUNK_MS, channels)
    head, tail = audio[:burst], audio[burst:]
    for pos in range(0, len(head), chunk):
        await dev.mic(head[pos:pos + chunk])
    for pos in range(0, len(tail), chunk):
        due = start + (pos + chunk) / ms_bytes(1000, channels) * scale
        if not done.is_set() and due > loop.time():
            await asyncio.sleep(due - loop.time())
        await dev.mic(tail[pos:pos + chunk])


async def prepare(srv, device_id: str, state: str, dev: FakeDevice, channels: int,
                  got: list, cooldown: float, tasks: list) -> None:
    """Put the device in its state before the group: `answering` plays the answer to a
    stand-in phrase through the replay, `follow_up` answers one and plays it out, which opens
    the follow-up window, `announcing` plays an announcement through the replay."""
    if state in ("answering", "announcing"):
        srv.tts.clips.append(LONG_CLIP)
    if state == "announcing":
        tasks.append(asyncio.create_task(srv.manager.announce(device_id, "объявление")))
    if state in ("answering", "follow_up"):
        await dev.send(type="wake", session=1, word="Захар")
        phrase = np.full(ms_bytes(STAND_IN_MS, channels) // 2, 1000, "<i2").tobytes()
        phrase += bytes(ms_bytes(300, channels))
        await stream(dev, phrase, channels, len(phrase), 1, asyncio.Event())
        await until(lambda: srv.rt.runs_store.list(device=device_id))  # its answer
    if state in ("answering", "announcing"):
        await until(lambda: any(m["type"] == "play_start" for m in got))
    if state == "answering":
        # Past the cooldown of its own wake, which was ruled before it answered: the real
        # wake was long before the group.
        await asyncio.sleep(cooldown)


async def replay(srv, monkeypatch, case: dict, group: dict, scale: float = SCALE) -> None:
    """Replay `group` of `case` and check every run's outcome against its `expected`."""
    monkeypatch.setattr("src.wake_arbiter.WINDOW_MS", WINDOW_MS * scale)
    monkeypatch.setattr("src.wake_arbiter.COOLDOWN_MS", COOLDOWN_MS * scale)
    loop = asyncio.get_running_loop()
    specs, runs = case["devices"], group["runs"]
    for run in runs.values():
        key = stt_key(recording(run["recording"]), specs[run["device"]]["mic_channels"])
        srv.rt.stt_backends["groq"].texts[key] = run["stt_text"]
    rulings = []
    sink = logger.add(rulings.append, format="{message}",
                      filter=lambda r: r["message"].startswith("wake arbitration:"))
    devices, got, tasks = {}, {}, []
    done = asyncio.Event()
    try:
        for device_id in group["online"]:
            devices[device_id] = await connect(srv, device_id, specs[device_id])
            got[device_id] = []
            tasks.append(asyncio.create_task(listen(devices[device_id], got[device_id])))
        await until(lambda: all(any(m["type"] == "config" for m in got[d]) for d in devices))
        for device_id, before in group["before"].items():
            await prepare(srv, device_id, before["state"], devices[device_id],
                          specs[device_id]["mic_channels"], got[device_id],
                          COOLDOWN_MS * scale / 1000, tasks)
        base = max((r["id"] for r in srv.rt.runs_store.list()), default=0)
        ruled_before = len(rulings)

        origin = None
        for session, event in enumerate(group["timeline"], start=2):
            at = event["at_ms"] * scale / 1000
            if event["event"] == "follow_up_window":
                # It opened once the stand-in answer played out.
                listening = {"type": "state", "session": 1, "value": "listening"}
                await until(lambda: got[event["device"]].count(listening) == 2)
                origin = loop.time() - at
            elif origin is None:
                origin = loop.time() - at
            else:
                await asyncio.sleep(max(0.0, origin + at - loop.time()))
            if event["event"] == "ruled":
                # The events after it in the log wait for it here.
                await until(lambda: len(rulings) > ruled_before)
                continue
            run = runs[str(event["run"])]
            dev, channels = devices[event["device"]], specs[event["device"]]["mic_channels"]
            audio = recording(run["recording"]) + bytes(ms_bytes(300, channels))
            if event["event"] == "follow_up_window":
                # The part the window drops, then the recording, silent up to its speech.
                silent = ms_bytes(run["speech_from_ms"], channels)
                audio = bytes(ms_bytes(FOLLOWUP_SKIP_MS, channels) + silent) + audio[silent:]
                burst = 0
            else:
                await dev.send(type="wake", session=session, word="Захар")
                burst = ms_bytes(PREROLL_MS, channels)
            tasks.append(asyncio.create_task(stream(dev, audio, channels, burst, scale, done)))
        done.set()

        def rows(device_id):
            return [(r["result"], r["reason"]) for r in reversed(srv.rt.runs_store.list())
                    if r["id"] > base and r["device"] == device_id]

        def shown(device_id):
            return [m["text"] for m in got[device_id] if m["type"] == "reply" and "session" not in m]

        await until(lambda: all(rows(run["device"]) for run in runs.values()), timeout=5)
        for run_id, run in runs.items():
            assert rows(run["device"]) == [tuple(run["expected"])], (run_id, run["device"])
            if run.get("answer_stopped"):
                assert not srv.device(run["device"]).conn.playing, run_id
        # A speakerless loser is shown the winner's answer once it is spoken.
        deadline = loop.time() + 2
        while loop.time() < deadline and not all(
                shown(run["device"]) for run in runs.values() if run.get("shown_answer_of")):
            await asyncio.sleep(0.01)
        for run_id, run in runs.items():
            winner = run.get("shown_answer_of")
            expected = [f"ответ {winner}"] if winner else []
            assert shown(run["device"]) == expected, (run_id, run["device"])
    finally:
        logger.remove(sink)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_the_speakers_follow_up_window_keeps_the_phrase_a_panel_woke_on(server, monkeypatch):
    # 06:10, runs 4438 and 4439: both devices answered.
    case = load("followup-window-and-panel-wake")
    await replay(server, monkeypatch, case, case["groups"][0])


async def test_a_wake_over_the_speakers_answer_after_the_ruling_stops_it(server, monkeypatch):
    # 07:01, runs 4478-4480: the answer played to its end. 80f1b2d12c91's phrase is the
    # answer's voice around «захар стоп» and is dropped.
    case = load("late-wake-over-the-playing-answer")
    await replay(server, monkeypatch, case, case["groups"][0])


async def test_panels_that_recorded_the_speakers_announcement_around_the_wake_word(
        server, monkeypatch):
    # 02:22, runs 4321 and 4322; run 4480 is replayed by the test above.
    case = load("panels-recorded-the-speakers-answer")
    await replay(server, monkeypatch, case, case["groups"][0])


@pytest.mark.parametrize("group", ["07:22:27", "07:23:13", "07:24:42", "06:59:55"])
async def test_the_speaker_keeps_a_phrase_whose_text_lost_the_wake_word(server, monkeypatch, group):
    # Runs 4509-4511, 4515-4517, 4525-4527, 4465-4467: nobody answered.
    case = load("speaker-lost-the-wake-word-in-its-text")
    await replay(server, monkeypatch, case, next(g for g in case["groups"] if g["name"] == group))


async def test_the_speakers_wake_that_reached_the_server_late_is_a_duplicate(server, monkeypatch):
    # 07:18, runs 4492-4494. Open in docs/wake-cases.md: no outcome is decided, so this asserts
    # today's — the speaker's wake lands in the panels' cooldown and is dropped — and a fix
    # flips it. At half speed: the wake came 69 ms before the cooldown ended, just after the
    # panel's answer starts here, and a loaded runner's event loop eats a 69 ms margin; at
    # scale 2 the margin is 138 ms.
    case = load("speaker-wake-late-to-the-server")
    await replay(server, monkeypatch, case, case["groups"][0], scale=2)
