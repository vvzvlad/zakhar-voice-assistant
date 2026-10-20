"""MQTT telemetry (src/telemetry.py) against a fake paho client: what is published, on
which topic, in which shape — from the payload cleaning to the device server's events,
the runs and the 30 s state."""

import asyncio
import json
import time
import types

import pytest
from loguru import logger
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from src import telemetry
from src.core_config import TelemetryConfig
from src.reconfig import Reconfigurator
from src.run_events import record_run
from src.runs_store import RunsStore, new_record
from src.telemetry import Telemetry, clean, publish_run
from tests.test_device_server import srv, until  # noqa: F401  (srv is a fixture)
from tests.test_pipeline import FakeRunsStore, make_llm_result, make_pipeline, patch_llm, say


class FakeClient:
    """paho's Client: records the calls; the published JSON is parsed back."""

    def __init__(self, api):
        assert api is CallbackAPIVersion.VERSION2
        self.calls = []
        self.published = []
        self.fail = False
        self.connected = True

    def is_connected(self):
        return self.connected

    def connect_async(self, host, port):
        self.calls.append(("connect_async", host, port))

    def loop_start(self):
        self.calls.append("loop_start")

    def disconnect(self):
        self.calls.append("disconnect")

    def loop_stop(self):
        self.calls.append("loop_stop")

    def publish(self, topic, payload, qos):
        if self.fail:
            raise ValueError("broken")
        self.published.append((topic, json.loads(payload), qos))

    def payloads(self, topic):
        """The payloads published to `topic`, without the `ts` every one carries."""
        return [{k: v for k, v in p.items() if k != "ts"}
                for t, p, _ in self.published if t == f"zakhar/{topic}"]


class Air:
    """Starts a real Telemetry on a runtime, over FakeClient; stops them all at teardown."""

    def __init__(self):
        self.clients = []
        self.started = []

    def client(self, api):
        self.clients.append(FakeClient(api))
        return self.clients[-1]

    async def start(self, rt, **cfg) -> FakeClient:
        rt.core.telemetry = TelemetryConfig(**{"enabled": True, "host": "broker", **cfg})
        rt.telemetry = Telemetry(rt, started_at=time.time())
        await rt.telemetry.start()
        self.started.append(rt.telemetry)
        return self.clients[-1]


@pytest.fixture
async def air(monkeypatch):
    air = Air()
    monkeypatch.setattr(telemetry.mqtt, "Client", air.client)
    yield air
    for t in air.started:
        await t.stop()


def _rt():
    """A runtime with what the telemetry, record_run and apply_job read."""
    return types.SimpleNamespace(
        core=types.SimpleNamespace(telemetry=TelemetryConfig()),
        manager=types.SimpleNamespace(clients=[]), telemetry=None,
        runs_store=None, run_events=None, llm_backends={},
    )


def test_clean_keeps_every_leaf_one_type():
    # A text that would pass for a number or a bool is quoted, so it stays a text in InfluxDB;
    # lists become objects; null and empty strings are left out at every depth.
    assert clean({
        "none": None, "empty": "", "text": "включи свет", "n": 5, "f": -1.5, "b": True,
        "num": "100", "exp": "1e3", "nan": "nan", "word": "True",
        "list": ["a", None, " 7 "], "nested": {"x": "", "y": {"z": "false"}},
    }) == {
        "text": "включи свет", "n": 5, "f": -1.5, "b": True,
        "num": '"100"', "exp": '"1e3"', "nan": '"nan"', "word": '"True"',
        "list": {"0": "a", "2": '" 7 "'}, "nested": {"y": {"z": '"false"'}},
    }


async def test_off_or_without_a_broker_nothing_is_published(air):
    for cfg in (TelemetryConfig(enabled=False, host="broker"), TelemetryConfig(enabled=True)):
        rt = _rt()
        rt.core.telemetry = cfg
        rt.telemetry = Telemetry(rt, started_at=0)
        await rt.telemetry.start()
        rt.telemetry.publish("x", {"a": 1})
        publish_run(rt, new_record("kitchen", "say"))
        await rt.telemetry.stop()
    assert air.clients == []


async def test_publishes_cleaned_json_at_qos1_until_stopped(air):
    rt = _rt()
    client = await air.start(rt, port=1884)
    assert client.calls == [("connect_async", "broker", 1884), "loop_start"]
    before = time.time()
    rt.telemetry.publish("x/y", {"a": "1", "b": None})
    [(topic, payload, qos)] = client.published[-1:]
    # Every payload carries the time it was published: the bridge stamps the time it got it.
    assert before - 0.001 <= payload.pop("ts") <= time.time() + 0.001  # ms-rounded
    assert (topic, payload, qos) == ("zakhar/x/y", {"a": '"1"'}, 1)
    # A failure is logged, never raised into the caller.
    client.fail = True
    rt.telemetry.publish("x/y", {"a": 2})
    publish_run(rt, {"attempts": 5})
    client.fail = False
    await rt.telemetry.stop()
    assert client.calls[-2:] == ["disconnect", "loop_stop"]
    published = len(client.published)
    rt.telemetry.publish("x/y", {"a": 3})
    assert len(client.published) == published


def test_a_lost_broker_is_warned_once_per_outage():
    t = Telemetry(_rt(), started_at=0)
    warnings = []
    sink = logger.add(warnings.append, level="WARNING")
    try:
        lost = ReasonCode(PacketTypes.DISCONNECT, "Unspecified error")
        t._on_connect_fail(None, None)
        t._on_connect_fail(None, None)          # still the same outage
        t._on_connect(None, None, None, ReasonCode(PacketTypes.CONNACK, "Success"), None)
        t._on_disconnect(None, None, None, lost, None)
        t._on_disconnect(None, None, None, ReasonCode(PacketTypes.DISCONNECT, "Success"), None)
    finally:
        logger.remove(sink)
    assert len(warnings) == 2


async def test_a_run_is_published_with_its_context(air):
    rt = _rt()
    client = await air.start(rt)
    record = new_record(
        None, "assistant", result="error", stt_text="100", error_stage="TTS",
        error_text="timeout", chain="main", model="m", tokens=12, t_tts=300, t_total=1200,
        request={"system_prompt": "P"},
        attempts=[
            {"kind": "followup", "provider": "jev", "text": "да", "outcome": "accepted",
             "probability": 0.9, "t": 40},
            {"kind": "chain", "chain": "main", "stt": "groq", "handler": "openrouter",
             "text": "100", "outcome": "answered", "error": None, "detail": None,
             "t_stt": 200, "t_handler": 500},
            {"kind": "tts", "provider": "yandex", "outcome": "error", "error": "503", "t": 30},
            {"kind": "tts", "provider": "yandex", "outcome": "spoke", "error": None, "t": 90},
            {"kind": "probe", "outcome": "done"},
        ],
        rounds=[
            {"round": 1, "calls": [{"name": "set_light", "args": {"room": "кухня"}, "result": "ok"},
                                   {"name": "get_time", "args": {}, "result": "12"}]},
            {"round": 2, "calls": []},
        ],
    )
    publish_run(rt, record, 7)
    assert client.payloads("devices/server/run/error") == [{
        "run_id": 7, "reason": "assistant", "chain": "main", "model": "m", "tokens": 12,
        "stt_text": '"100"', "error_stage": "TTS", "error_text": "timeout",
        "t_tts": 300, "t_total": 1200,
        "attempts": {
            "followup_jev": {"kind": "followup", "provider": "jev", "text": "да",
                             "outcome": "accepted", "probability": 0.9, "t": 40},
            "chain_main": {"kind": "chain", "chain": "main", "stt": "groq", "handler": "openrouter",
                           "text": '"100"', "outcome": "answered", "t_stt": 200, "t_handler": 500},
            "tts_yandex": {"kind": "tts", "provider": "yandex", "outcome": "error",
                           "error": '"503"', "t": 30},
            "tts_yandex_2": {"kind": "tts", "provider": "yandex", "outcome": "spoke", "t": 90},
            "probe": {"kind": "probe", "outcome": "done"},
        },
        "tools": {
            "0": {"round": 1, "name": "set_light", "args": '{"room": "кухня"}', "result": "ok"},
            "1": {"round": 1, "name": "get_time", "args": "{}", "result": '"12"'},
        },
    }]


async def test_record_run_publishes_whether_or_not_the_log_is_on(air, tmp_path):
    rt = _rt()
    client = await air.start(rt)
    rt.runs_store = RunsStore(str(tmp_path / "runs.db"))
    await record_run(rt, new_record("kitchen", "say", result="ok", llm_text="привет"))
    rt.runs_store = None
    await record_run(rt, new_record("kitchen", "reminder", result="ok", llm_text="пора"))
    assert [(p["reason"], p.get("run_id")) for p in client.payloads("devices/kitchen/run/ok")] == [
        ("say", 1), ("reminder", None)]


async def test_a_pipeline_run_is_published_with_the_log_off(air, tmp_path, monkeypatch):
    patch_llm(monkeypatch, reply="готово")
    pipeline, _ = make_pipeline(tmp_path, monkeypatch, stt_text="включи свет")
    pipeline.rt.manager = types.SimpleNamespace(clients=[])
    client = await air.start(pipeline.rt)
    await say(pipeline)
    [run] = client.payloads("devices/dev/run/ok")
    assert "run_id" not in run
    assert (run["stt_text"], run["llm_text"], run["reason"]) == ("включи свет", "готово", "endpoint")
    assert run["t_total"] >= 0 and run["attempts"]["chain_main"]["outcome"] == "answered"


async def test_a_config_change_restarts_the_publisher(air):
    rt = _rt()
    old = await air.start(rt, host="old")
    reconf = Reconfigurator(rt, types.SimpleNamespace(tts_timeout=30), asyncio.Queue())
    rt.core.telemetry = TelemetryConfig(enabled=True, host="new")
    reconf.on_config_change({"core.telemetry.host"})
    await reconf.apply_job(reconf.queue.get_nowait())
    assert old.calls == [("connect_async", "old", 1883), "loop_start", "disconnect", "loop_stop"]
    assert air.clients[-1].calls == [("connect_async", "new", 1883), "loop_start"]
    rt.core.telemetry = TelemetryConfig(enabled=False, host="new")
    await reconf.apply_job({"core.telemetry.enabled"})
    assert air.clients[-1].calls[-2:] == ["disconnect", "loop_stop"]
    assert len(air.clients) == 2


# --- through the device server ----------------------------------------------------

@pytest.fixture
async def broker(srv, air):  # noqa: F811
    srv.svc.apply({"core": {"devices": [{"name": "kitchen", "label": "Кухня"}]}})
    return await air.start(srv.rt)


async def _hello(srv, uptime, **extra):  # noqa: F811
    """A device without a speaker says hello; returns its socket."""
    ws = await srv.http.ws_connect(f"http://127.0.0.1:{srv.manager.port}/device")
    await ws.send_str(json.dumps({"type": "hello", "id": "kitchen", "fw": "fw 1", "uptime": uptime,
                                  "caps": {"speaker": False}, **extra}))
    return ws


async def test_connections_publish_connected_and_disconnected(srv, broker):  # noqa: F811
    connected, gone = "devices/kitchen/event/connected", "devices/kitchen/event/disconnected"
    first = await _hello(srv, 100, reset="task_wdt")
    await until(lambda: len(broker.payloads(connected)) == 1)
    second = await _hello(srv, 101)          # the same boot: a reconnect that replaces the first
    await until(lambda: len(broker.payloads(connected)) == 2)
    await second.close()
    await until(lambda: broker.payloads(gone))
    third = await _hello(srv, 5)             # booted again meanwhile
    await until(lambda: len(broker.payloads(connected)) == 3)
    assert broker.payloads(connected) == [
        {"fw": "fw 1", "uptime": 100, "reset": "task_wdt", "speaker": False, "replaced": False,
         "rebooted": False},
        {"fw": "fw 1", "uptime": 101, "speaker": False, "replaced": True, "rebooted": False},
        {"fw": "fw 1", "uptime": 5, "speaker": False, "replaced": False, "rebooted": True},
    ]
    # Only the live connection's end is an event: the replaced first one is not.
    [event] = broker.payloads(gone)
    assert event["code"] == 1000 and event["connected_s"] >= 0 and "error" not in event
    await first.close()
    await third.close()


async def test_a_session_publishes_its_events_and_its_run(srv, broker, monkeypatch):  # noqa: F811
    rounds = [{"round": 1, "calls": [{"name": "set_light", "args": {"on": True}, "result": "ok"}]}]
    patch_llm(monkeypatch, result=make_llm_result("готово", rounds=rounds, tool_used=True))
    srv.rt.runs_store = FakeRunsStore()
    dev = await srv.connect()
    await dev.recv()  # config
    await dev.wake(17)
    await dev.phrase()
    await dev.finish()  # answers `played` with dry 0, gap_ms 0
    await dev.send(type="volume", value=40)
    await dev.send(type="ambient", value=-50.5)
    await dev.send(type="ambient", value=None)
    await dev.send(type="stats", rssi=-61, cpu0=12, temp=41.5, heap=123456, wifi_reason=201)
    await dev.send(type="stop")
    await until(lambda: broker.payloads("devices/kitchen/event/stop"))
    assert broker.payloads("devices/kitchen/event/wake") == [
        {"session": 17, "word": "Захар", "over_answer": False}]
    assert broker.payloads("devices/kitchen/event/played") == [{"dry": 0, "gap_ms": 0}]
    assert broker.payloads("devices/kitchen/event/volume") == [{"value": 40}]
    assert broker.payloads("devices/kitchen/event/ambient") == [
        {"value": -50.5, "zeros": False}, {"zeros": True}]
    assert broker.payloads("devices/kitchen/stats") == [
        {"rssi": -61, "cpu0": 12, "temp": 41.5, "heap": 123456, "wifi_reason": 201}]
    assert broker.payloads("devices/kitchen/event/stop") == [{"in_session": False}]
    [run] = broker.payloads("devices/kitchen/run/tool")
    assert (run["run_id"], run["stt_text"], run["llm_text"]) == (1, "включи свет", "готово")
    assert (run["play_dry"], run["play_gap_ms"]) == (0, 0)
    assert run["tools"] == {"0": {"round": 1, "name": "set_light", "args": '{"on": true}',
                                  "result": "ok"}}
    assert run["attempts"]["tts_yandex"]["outcome"] == "spoke"


async def test_the_state_covers_the_server_and_every_enabled_device(srv, broker):  # noqa: F811
    # The fake broker is connected from the start, so the first state goes out at once, the
    # kitchen still offline (a real one is skipped until paho connects: the next tick, 30 s on).
    await until(lambda: broker.payloads("devices/kitchen/state"))
    assert broker.payloads("devices/kitchen/state") == [{"online": False, "label": "Кухня"}]
    [server] = broker.payloads("server/state")
    assert (server["version"], server["devices"], server["devices_online"]) == (
        telemetry.__version__, 1, 0)
    dev = await srv.connect(uptime=3725)
    await dev.recv()  # config
    await dev.send(type="model_loaded", kind="wake", name="zakhar/zakhar_v31", ok=True)
    await dev.send(type="ambient", value=-50)
    await dev.send(type="ambient", value=-40)
    await until(lambda: len(srv.device().conn.ambient) == 2)
    srv.rt.telemetry.publish_state()
    state = broker.payloads("devices/kitchen/state")[-1]
    assert 3725 <= state.pop("uptime") <= 3727
    assert state == {"online": True, "label": "Кухня", "fw": "voicepe 0.1.0 (a1b2c3d)",
                     "model": "zakhar/zakhar_v31", "volume": 60, "ambient": -45.0, "boost": 0}
    assert broker.payloads("server/state")[-1]["devices_online"] == 1
    # While the broker is away the state is not queued: it would land late, all at once.
    broker.connected = False
    published = len(broker.published)
    srv.rt.telemetry.publish_state()
    assert len(broker.published) == published
