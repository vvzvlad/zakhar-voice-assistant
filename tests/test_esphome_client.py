import asyncio
import io
import types
import wave

import pytest

import src.esphome_client as esphome_client
from src.core_config import DeviceConfig
from src.esphome_client import DeviceManager


class _Cfg:
    def __init__(self, name): self.name = name


class _FakeClient:
    def __init__(self, name, *, fail=False):
        self.cfg = _Cfg(name)
        self.fail = fail
        self.started = False
        self.stopped = False

    async def start(self):
        if self.fail:
            raise RuntimeError("boom")
        self.started = True

    async def stop(self):
        if self.fail:
            raise RuntimeError("boom")
        self.stopped = True


def _manager(clients):
    mgr = DeviceManager.__new__(DeviceManager)  # bypass __init__ (needs real deps)
    mgr.clients = clients
    return mgr


# --- manual capture (DeviceClient.capture + DeviceManager.capture) ------------

class _Ent:
    def __init__(self, object_id, key):
        self.object_id = object_id
        self.key = key


class _CaptureCli:
    """Fake APIClient recording number_command/button_command/switch_command calls."""

    def __init__(self):
        self.numbers = []  # (key, state)
        self.buttons = []  # key
        self.switches = []  # (key, state)

    def number_command(self, key, state, device_id=0):
        self.numbers.append((key, state))

    def button_command(self, key, device_id=0):
        self.buttons.append(key)

    def switch_command(self, key, state, device_id=0):
        self.switches.append((key, state))


def _wav_bytes(pcm=b"\x01\x02" * 8):
    """Build a tiny valid 16k/mono/16-bit WAV for capture-return assertions."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return buf.getvalue()


class _CapturePipeline:
    """Fake pipeline recording the armed seconds and returning a resolvable Future.

    arm_capture() returns the Future DeviceClient.capture() awaits. By default it is
    pre-resolved with a tiny WAV (the device "recorded" instantly); set `resolve` to
    None to leave it pending so the wait_for timeout path can be exercised.
    """

    def __init__(self, resolve=_wav_bytes()):
        self.armed = []
        self.disarmed = 0
        self._resolve = resolve

    def arm_capture(self, seconds):
        self.armed.append(seconds)
        fut = asyncio.get_event_loop().create_future()
        if self._resolve is not None:
            fut.set_result(self._resolve)
        self._future = fut
        return fut

    def disarm_capture(self):
        self.disarmed += 1
        if getattr(self, "_future", None) is not None and not self._future.done():
            self._future.set_exception(RuntimeError("cancelled"))


def _capture_client(name="dev", *, online=True, btn_key=11, sec_key=22,
                    resolve=_wav_bytes()):
    """Build a DeviceClient bypassing __init__, wired with capture fakes.

    `resolve` is the WAV bytes the fake pipeline's Future resolves with (None ->
    leave it pending so the capture() timeout path can be tested)."""
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg(name)
    c.online = online
    c.cli = _CaptureCli()
    c.pipeline = _CapturePipeline(resolve=resolve)
    c._capture_button_key = btn_key
    c._capture_seconds_key = sec_key
    return c


def test_discover_capture_keys_maps_by_object_id():
    # These object_ids are exactly what the firmware sends over the Native API:
    # object_id = slugify(name) for the "Capture Seconds" / "Capture Sample"
    # entities in esphome/zakhar-voice-preroll.yaml (NOT the YAML `id:` field). They
    # MUST stay equal to slugify(name) or discovery silently fails on real hardware.
    c = _capture_client(btn_key=None, sec_key=None)
    c._discover_capture_keys([
        _Ent("some_other", 1),
        _Ent("capture_seconds", 22),
        _Ent("capture_sample", 11),
    ])
    assert c._capture_button_key == 11
    assert c._capture_seconds_key == 22


def test_discover_capture_keys_absent_leaves_none():
    c = _capture_client(btn_key=5, sec_key=6)
    c._discover_capture_keys([_Ent("unrelated", 1)])  # no capture entities
    assert c._capture_button_key is None
    assert c._capture_seconds_key is None


async def test_capture_arms_commands_and_returns_wav_bytes():
    wav = _wav_bytes(b"\x09\x0a" * 12)
    c = _capture_client(btn_key=11, sec_key=22, resolve=wav)
    out = await c.capture(7)
    # Pipeline armed BEFORE the device commands.
    assert c.pipeline.armed == [7]
    # seconds set first (as float), then the button pressed.
    assert c.cli.numbers == [(22, 7.0)]
    assert c.cli.buttons == [11]
    # capture() returns the WAV bytes the pipeline's Future resolved with.
    assert out == wav
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == 16000


async def test_capture_times_out_and_disarms(monkeypatch):
    # The pipeline Future never resolves (device never streamed). capture() must
    # raise TimeoutError and disarm the pipeline so a later run isn't hijacked.
    import src.esphome_client as ec
    monkeypatch.setattr(ec, "CAPTURE_WAIT_MARGIN", 0.0)
    c = _capture_client(btn_key=11, sec_key=22, resolve=None)

    async def fast_wait_for(fut, timeout):
        # Don't actually wait the requested seconds in the test.
        raise asyncio.TimeoutError

    monkeypatch.setattr(ec.asyncio, "wait_for", fast_wait_for)
    with pytest.raises(TimeoutError):
        await c.capture(2)
    assert c.pipeline.disarmed == 1


async def test_capture_raises_when_offline():
    c = _capture_client(online=False)
    with pytest.raises(RuntimeError):
        await c.capture(5)
    assert c.pipeline.armed == []  # never armed when offline


async def test_capture_raises_when_keys_missing():
    c = _capture_client(btn_key=None, sec_key=None)
    with pytest.raises(RuntimeError):
        await c.capture(5)
    assert c.pipeline.armed == []


# --- live Wake Probability (DeviceClient + DeviceManager) ---------------------

def _wake_prob_client(name="dev", *, online=True, switch_key=33, sensor_key=44):
    """Build a DeviceClient bypassing __init__, wired for the wake-prob tests."""
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg(name)
    c.online = online
    c.cli = _CaptureCli()
    c._wake_prob_switch_key = switch_key
    c._wake_prob_sensor_key = sensor_key
    c._wake_prob_value = None
    # _on_state also touches the control/version key sets — keep them empty here.
    c._control_keys = set()
    c._control_value = {}
    c._version_keys = set()
    c._version_value = {}
    return c


class _State:
    """Fake subscribe_states state object (key/state/missing_state)."""

    def __init__(self, key, state, missing_state=False):
        self.key = key
        self.state = state
        self.missing_state = missing_state


def test_discover_wake_prob_keys_maps_by_object_id():
    # object_id = slugify(name) for "Wake Probability Stream" / "Wake Probability";
    # these MUST stay equal to slugify(name) or discovery silently fails on hardware.
    c = _wake_prob_client(switch_key=None, sensor_key=None)
    c._discover_wake_prob_keys([
        _Ent("unrelated", 1),
        _Ent("wake_probability_stream", 33),
        _Ent("wake_probability", 44),
    ])
    assert c._wake_prob_switch_key == 33
    assert c._wake_prob_sensor_key == 44


def test_discover_wake_prob_keys_absent_leaves_none():
    c = _wake_prob_client(switch_key=7, sensor_key=8)
    c._discover_wake_prob_keys([_Ent("unrelated", 1)])
    assert c._wake_prob_switch_key is None
    assert c._wake_prob_sensor_key is None


def test_on_state_caches_wake_prob_sensor_value():
    c = _wake_prob_client(sensor_key=44)
    c._on_state(_State(44, "62.0"))
    assert c._wake_prob_value == 62.0
    # A missing_state push must not overwrite the cached value.
    c._on_state(_State(44, "99.0", missing_state=True))
    assert c._wake_prob_value == 62.0
    # An unrelated key is ignored.
    c._on_state(_State(999, "10.0"))
    assert c._wake_prob_value == 62.0


def test_set_wake_prob_stream_issues_switch_command():
    c = _wake_prob_client(switch_key=33)
    c._wake_prob_value = 50.0
    c.set_wake_prob_stream(True)
    assert c.cli.switches == [(33, True)]
    # Enabling leaves the cached value intact.
    assert c._wake_prob_value == 50.0
    # Disabling issues the off command AND clears the cached value (no stale read).
    c.set_wake_prob_stream(False)
    assert c.cli.switches == [(33, True), (33, False)]
    assert c._wake_prob_value is None


def test_set_wake_prob_stream_missing_switch_raises_lookup():
    c = _wake_prob_client(switch_key=None)
    with pytest.raises(LookupError):
        c.set_wake_prob_stream(True)
    assert c.cli.switches == []


def test_set_wake_prob_stream_offline_raises_runtime():
    c = _wake_prob_client(online=False)
    with pytest.raises(RuntimeError):
        c.set_wake_prob_stream(True)
    assert c.cli.switches == []


def test_wake_prob_snapshot_shape():
    c = _wake_prob_client(switch_key=33, sensor_key=44)
    c._wake_prob_value = 71.0
    assert c.wake_prob() == {"available": True, "value": 71.0}
    # available is False unless BOTH entities are present.
    c._wake_prob_sensor_key = None
    assert c.wake_prob() == {"available": False, "value": 71.0}


class _MgrWakeProbClient:
    """Fake DeviceClient for DeviceManager wake-prob routing tests."""

    def __init__(self, name, online=True, available=True, value=None):
        self.cfg = _Cfg(name)
        self.online = online
        self._available = available
        self._value = value
        self.stream_calls = []  # enabled flags

    def set_wake_prob_stream(self, enabled):
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        self.stream_calls.append(enabled)

    def wake_prob(self):
        return {"available": self._available, "value": self._value}


def test_manager_set_wake_prob_stream_routes_and_returns_snapshot():
    a = _MgrWakeProbClient("a")
    b = _MgrWakeProbClient("b", value=33.0)
    mgr = _manager([a, b])
    out = mgr.set_wake_prob_stream("b", True)
    assert b.stream_calls == [True] and a.stream_calls == []
    assert out == {"device": "b", "online": True, "available": True, "value": 33.0}


def test_manager_set_wake_prob_stream_unknown_device_raises_lookup():
    mgr = _manager([_MgrWakeProbClient("a")])
    with pytest.raises(LookupError):
        mgr.set_wake_prob_stream("nope", True)


def test_manager_set_wake_prob_stream_offline_propagates_runtime():
    off = _MgrWakeProbClient("a", online=False)
    mgr = _manager([off])
    with pytest.raises(RuntimeError):
        mgr.set_wake_prob_stream("a", True)


def test_manager_wake_prob_online_returns_client_snapshot():
    a = _MgrWakeProbClient("a", available=True, value=42.0)
    mgr = _manager([a])
    assert mgr.wake_prob("a") == {"device": "a", "online": True, "available": True, "value": 42.0}


def test_manager_wake_prob_offline_reports_unavailable_without_call():
    off = _MgrWakeProbClient("a", online=False, available=True, value=42.0)
    mgr = _manager([off])
    # An offline speaker reports unavailable/None without touching the client snapshot.
    assert mgr.wake_prob("a") == {"device": "a", "online": False, "available": False, "value": None}


def test_manager_wake_prob_unknown_device_raises_lookup():
    mgr = _manager([_MgrWakeProbClient("a")])
    with pytest.raises(LookupError):
        mgr.wake_prob("nope")


class _MgrCaptureClient:
    """Fake DeviceClient for DeviceManager.capture routing tests."""

    def __init__(self, name, online=True):
        self.cfg = _Cfg(name)
        self.online = online
        self.captured = []

    async def capture(self, seconds):
        self.captured.append(seconds)
        return _wav_bytes(b"\x0b\x0c" * 10)


async def test_manager_capture_routes_to_named_online_client_and_returns_bytes():
    a = _MgrCaptureClient("a")
    b = _MgrCaptureClient("b")
    mgr = _manager([a, b])
    out = await mgr.capture("b", 9)
    assert b.captured == [9] and a.captured == []
    # The WAV bytes flow straight back through the manager to the caller.
    assert out == _wav_bytes(b"\x0b\x0c" * 10)


async def test_manager_capture_unknown_device_raises_lookup():
    mgr = _manager([_MgrCaptureClient("a")])
    with pytest.raises(LookupError):
        await mgr.capture("nope", 5)


async def test_manager_capture_offline_raises_runtime():
    off = _MgrCaptureClient("a", online=False)
    mgr = _manager([off])
    with pytest.raises(RuntimeError):
        await mgr.capture("a", 5)
    assert off.captured == []


async def test_start_isolates_failing_client():
    a, bad, c = _FakeClient("a"), _FakeClient("bad", fail=True), _FakeClient("c")
    await _manager([a, bad, c]).start()
    assert a.started and c.started  # a failing client doesn't abort the others


async def test_stop_isolates_failing_client():
    a, bad, c = _FakeClient("a"), _FakeClient("bad", fail=True), _FakeClient("c")
    await _manager([a, bad, c]).stop()
    assert a.stopped and c.stopped


# --- reconfigure (Tier 3c: hot device reconcile) -----------------------------

class _ReconcileClient:
    """Fake DeviceClient recording start/stop, carrying a real DeviceConfig as .cfg.
    Its constructor matches the real (cfg, zc, runtime) signature so the manager's
    reconfigure() can instantiate it via the monkeypatched DeviceClient."""

    def __init__(self, cfg, zc, runtime):
        self.cfg = cfg
        self.zc = zc
        self.rt = runtime
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def _device(name, host="10.0.0.1", psk="psk0", enabled=True):
    return DeviceConfig(name=name, host=host, psk=psk, enabled=enabled)


def _reconfigure_runtime(devices, *, port=6053):
    """Runtime stub exposing the bits reconfigure() reads: core.devices/core.esphome.port."""
    return types.SimpleNamespace(
        core=types.SimpleNamespace(
            devices=devices,
            esphome=types.SimpleNamespace(port=port),
        )
    )


def _make_manager(devices, *, port=6053, monkeypatch):
    """Build a real DeviceManager whose clients are fake (DeviceClient monkeypatched)."""
    monkeypatch.setattr(esphome_client, "DeviceClient", _ReconcileClient)
    rt = _reconfigure_runtime(devices, port=port)
    mgr = DeviceManager(object(), rt)   # zc is an opaque sentinel for the fakes
    return mgr, rt


@pytest.mark.asyncio
async def test_reconfigure_adds_only_new_client(monkeypatch):
    a = _device("a")
    mgr, rt = _make_manager([a], monkeypatch=monkeypatch)
    existing = mgr.clients[0]
    # Add a second device.
    b = _device("b", host="10.0.0.2", psk="psk-b")
    rt.core.devices = [a, b]
    await mgr.reconfigure()
    assert len(mgr.clients) == 2
    assert existing in mgr.clients and existing.stopped is False  # kept, not touched
    new = next(c for c in mgr.clients if c.cfg.name == "b")
    assert new.started is True and existing.started is False       # only the new one started


@pytest.mark.asyncio
async def test_reconfigure_removes_only_dropped_client(monkeypatch):
    a, b = _device("a"), _device("b", host="10.0.0.2", psk="psk-b")
    mgr, rt = _make_manager([a, b], monkeypatch=monkeypatch)
    client_a = next(c for c in mgr.clients if c.cfg.name == "a")
    client_b = next(c for c in mgr.clients if c.cfg.name == "b")
    rt.core.devices = [a]   # drop b
    await mgr.reconfigure()
    assert mgr.clients == [client_a]
    assert client_b.stopped is True and client_a.stopped is False  # only b stopped+dropped


@pytest.mark.asyncio
async def test_reconfigure_changed_host_psk_recreates(monkeypatch):
    a = _device("a", host="10.0.0.1", psk="psk0")
    mgr, rt = _make_manager([a], monkeypatch=monkeypatch)
    old = mgr.clients[0]
    # Same name, different host/psk -> different key -> stop old, start new.
    a2 = _device("a", host="10.0.0.9", psk="psk-new")
    rt.core.devices = [a2]
    await mgr.reconfigure()
    assert len(mgr.clients) == 1
    new = mgr.clients[0]
    assert new is not old
    assert old.stopped is True                       # old key stopped
    assert new.started is True and new.cfg.host == "10.0.0.9" and new.cfg.psk == "psk-new"


@pytest.mark.asyncio
async def test_reconfigure_port_change_rebuilds_all(monkeypatch):
    a, b = _device("a"), _device("b", host="10.0.0.2", psk="psk-b")
    mgr, rt = _make_manager([a, b], port=6053, monkeypatch=monkeypatch)
    old_clients = list(mgr.clients)
    rt.core.esphome.port = 7000   # global port change -> rebuild EVERY client
    await mgr.reconfigure()
    assert mgr._esphome_port == 7000
    assert all(c.stopped for c in old_clients)        # every old client stopped
    assert len(mgr.clients) == 2
    assert all(c not in old_clients for c in mgr.clients)  # all fresh instances
    assert all(c.started for c in mgr.clients)        # every new client started


def test_init_skips_disabled_devices(monkeypatch):
    # A disabled device must get NO client at construction time.
    a, off = _device("a"), _device("off", host="10.0.0.9", enabled=False)
    mgr, _rt = _make_manager([a, off], monkeypatch=monkeypatch)
    assert [c.cfg.name for c in mgr.clients] == ["a"]


@pytest.mark.asyncio
async def test_reconfigure_disabling_stops_and_removes_client(monkeypatch):
    a, b = _device("a"), _device("b", host="10.0.0.2", psk="psk-b")
    mgr, rt = _make_manager([a, b], monkeypatch=monkeypatch)
    client_b = next(c for c in mgr.clients if c.cfg.name == "b")
    # Flip b to disabled (same name/host/psk) -> its client is stopped + removed.
    rt.core.devices = [a, _device("b", host="10.0.0.2", psk="psk-b", enabled=False)]
    await mgr.reconfigure()
    assert [c.cfg.name for c in mgr.clients] == ["a"]
    assert client_b.stopped is True


@pytest.mark.asyncio
async def test_reconfigure_reenabling_creates_and_starts_client(monkeypatch):
    a = _device("a")
    off = _device("b", host="10.0.0.2", psk="psk-b", enabled=False)
    mgr, rt = _make_manager([a, off], monkeypatch=monkeypatch)
    assert [c.cfg.name for c in mgr.clients] == ["a"]  # disabled -> no client
    # Re-enable b -> a fresh client is created and started.
    rt.core.devices = [a, _device("b", host="10.0.0.2", psk="psk-b", enabled=True)]
    await mgr.reconfigure()
    assert sorted(c.cfg.name for c in mgr.clients) == ["a", "b"]
    new = next(c for c in mgr.clients if c.cfg.name == "b")
    assert new.started is True


@pytest.mark.asyncio
async def test_reconfigure_noop_when_unchanged(monkeypatch):
    a = _device("a")
    mgr, _rt = _make_manager([a], monkeypatch=monkeypatch)
    existing = mgr.clients[0]
    await mgr.reconfigure()   # config unchanged
    assert mgr.clients == [existing]
    assert existing.started is False and existing.stopped is False  # untouched


# --- DeviceManager.statuses --------------------------------------------------

class _StatusClient:
    """Minimal client exposing the .cfg.name/.online/.versions() statuses() reads."""

    def __init__(self, name, online, versions=None):
        self.cfg = types.SimpleNamespace(name=name)
        self.online = online
        self._versions = versions or []

    def versions(self):
        return self._versions


def _status_manager(devices, clients):
    """Manager via __new__ with a fake runtime carrying the configured devices."""
    mgr = _manager(clients)
    mgr.rt = types.SimpleNamespace(core=types.SimpleNamespace(devices=devices))
    return mgr


def test_statuses_reports_each_configured_device_in_order():
    devices = [
        DeviceConfig(name="a", host="10.0.0.1", psk="p1"),
        DeviceConfig(name="b", host="10.0.0.2", psk="p2"),
    ]
    versions = [{"id": "config_version", "name": "Config Version", "value": "v16"}]
    a = _StatusClient("a", online=True, versions=versions)
    b = _StatusClient("b", online=False)
    mgr = _status_manager(devices, [a, b])
    # One dict per CONFIGURED device, in config order. versions flow through only
    # for online clients (the offline one reads []).
    assert mgr.statuses() == [
        {"name": "a", "host": "10.0.0.1", "enabled": True, "online": True,
         "versions": versions},
        {"name": "b", "host": "10.0.0.2", "enabled": True, "online": False,
         "versions": []},
    ]


def test_statuses_disabled_device_reads_offline_without_client():
    # A disabled speaker has no client at all but is still reported (offline).
    devices = [
        DeviceConfig(name="a", host="10.0.0.1", psk="p1"),
        DeviceConfig(name="off", host="10.0.0.9", psk="p9", enabled=False),
    ]
    mgr = _status_manager(devices, [_StatusClient("a", online=True)])
    assert mgr.statuses() == [
        {"name": "a", "host": "10.0.0.1", "enabled": True, "online": True,
         "versions": []},
        {"name": "off", "host": "10.0.0.9", "enabled": False, "online": False,
         "versions": []},
    ]


def test_statuses_empty_config_is_empty():
    assert _status_manager([], []).statuses() == []


# --- DeviceClient.announce ----------------------------------------------------

class _AnnouncePipeline:
    """Fake pipeline recording speak() calls (the single text->speaker entry)."""

    def __init__(self):
        self.spoken = []  # text

    async def speak(self, text):
        self.spoken.append(text)


def _announce_client(name="dev", *, online=True):
    """Build a DeviceClient via __new__ wired with the announce fakes."""
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg(name)
    c.online = online
    c.pipeline = _AnnouncePipeline()
    return c


async def test_announce_delegates_to_pipeline_speak():
    # The device layer no longer touches pipeline internals (tts_backend /
    # audio_server): announce is online-check + pipeline.speak(text).
    c = _announce_client()
    await c.announce("привет")
    assert c.pipeline.spoken == ["привет"]


async def test_announce_offline_raises_without_speaking():
    c = _announce_client(online=False)
    with pytest.raises(RuntimeError):
        await c.announce("hi")
    # The offline guard must short-circuit before any synthesis / playback.
    assert c.pipeline.spoken == []


# --- DeviceClient.play_media ---------------------------------------------------

class _PlayMediaPipeline:
    """Fake pipeline recording serve_audio() positional args, returning (ext, url, nbytes)."""

    def __init__(self, url="<url>"):
        self.served = []  # positional-args tuples, exactly as serve_audio received them
        self._url = url

    async def serve_audio(self, *args):
        self.served.append(args)
        return ("mp3", self._url, len(args[1]))


class _PlayMediaCli:
    """Fake APIClient recording the announcement-await kwargs."""

    def __init__(self):
        self.announcements = []  # kwargs dicts

    async def send_voice_assistant_announcement_await_response(self, **kwargs):
        self.announcements.append(kwargs)


def _play_media_client(name="dev", *, online=True):
    """Build a DeviceClient via __new__ wired with the play_media fakes."""
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg(name)
    c.online = online
    c.pipeline = _PlayMediaPipeline()
    c.cli = _PlayMediaCli()
    return c


async def test_play_media_serves_audio_and_announces_url():
    c = _play_media_client()
    await c.play_media(audio=b"AUD", mime="audio/mpeg")
    # serve_audio takes (mime, audio) positionally, in EXACTLY that order — the
    # pipeline.serve_audio contract (swapped args would serve garbage audio).
    assert c.pipeline.served == [("audio/mpeg", b"AUD")]
    # The served URL is played through the assist-satellite announcement channel.
    assert c.cli.announcements == [{"media_id": "<url>", "timeout": 30.0, "text": ""}]


async def test_play_media_offline_raises_without_serving():
    c = _play_media_client(online=False)
    with pytest.raises(RuntimeError):
        await c.play_media(audio=b"AUD", mime="audio/mpeg")
    # The offline guard must short-circuit before serving or announcing anything.
    assert c.pipeline.served == []
    assert c.cli.announcements == []


# --- DeviceManager.announce routing ------------------------------------------

class _MgrAnnounceClient:
    """Fake DeviceClient for DeviceManager.announce routing tests."""

    def __init__(self, name, online=True):
        self.cfg = _Cfg(name)
        self.online = online
        self.announced = []  # text

    async def announce(self, text):
        self.announced.append(text)


async def test_manager_announce_none_with_no_online_client_is_noop():
    # device_name=None and every client offline -> no client.announce, no raise.
    a = _MgrAnnounceClient("a", online=False)
    b = _MgrAnnounceClient("b", online=False)
    mgr = _manager([a, b])
    await mgr.announce(None, "hi")  # must not raise
    assert a.announced == [] and b.announced == []


# --- DeviceManager.play_chime --------------------------------------------------

class _ChimeClient:
    """Fake DeviceClient for DeviceManager.play_chime tests, recording play_media."""

    def __init__(self, name, online=True, fail=False):
        self.cfg = _Cfg(name)
        self.online = online
        self.fail = fail
        self.played = []  # (audio, mime) — positional, matching play_chime's call

    async def play_media(self, audio, mime):
        if self.fail:
            raise RuntimeError("play boom")
        self.played.append((audio, mime))


async def test_play_chime_plays_synthesized_chime_on_online_skips_offline():
    a = _ChimeClient("a")
    b = _ChimeClient("b")
    c = _ChimeClient("c", online=False)
    mgr = _manager([a, b, c])
    result = await mgr.play_chime("")  # empty path -> synthesized chime
    assert result == {"played": ["a", "b"], "offline": ["c"]}
    # Every online client received the same MP3 clip; the offline one nothing.
    for cl in (a, b):
        assert len(cl.played) == 1
        audio, mime = cl.played[0]
        assert mime == "audio/mpeg"
        assert audio  # non-empty clip bytes
    assert c.played == []


async def test_play_chime_failing_client_lands_in_offline_others_still_play():
    ok = _ChimeClient("ok")
    bad = _ChimeClient("bad", fail=True)
    mgr = _manager([ok, bad])
    result = await mgr.play_chime("")
    # The per-device failure is reported, never raised, and does not block others.
    assert result == {"played": ["ok"], "offline": ["bad"]}
    assert len(ok.played) == 1


async def test_play_chime_unknown_device_raises_lookup():
    mgr = _manager([_ChimeClient("a")])
    with pytest.raises(LookupError):
        await mgr.play_chime("", device_name="nope")


async def test_play_chime_wav_file_is_transcoded_end_to_end(tmp_path):
    # Seam test: a real on-disk WAV goes through build_ack_clip -> play_media,
    # so the client must receive MP3 bytes (transcoded), not the raw WAV.
    wav = _wav_bytes()
    p = tmp_path / "chime.wav"
    p.write_bytes(wav)
    a = _ChimeClient("a")
    mgr = _manager([a])
    result = await mgr.play_chime(str(p))
    assert result == {"played": ["a"], "offline": []}
    audio, mime = a.played[0]
    assert mime == "audio/mpeg"
    assert audio            # non-empty
    assert audio != wav     # transcoded, not the raw WAV bytes
    assert audio[0] == 0xFF  # MP3 frame sync byte


# --- DeviceClient._on_connect / _on_disconnect -------------------------------

class _ConnectPipeline:
    """Fake pipeline whose send_event/send_audio/send_announcement get rebound by _on_connect."""

    def __init__(self):
        self.send_event = None
        self.send_audio = None
        self.send_announcement = None


class _ConnectCli:
    """Fake APIClient for the _on_connect path.

    device_info_and_list_entities() returns (info, entities, services); set
    `fail=True` to make it raise so the swallowed-failure branch is exercised.
    subscribe_voice_assistant() returns a sentinel unsub token.
    """

    def __init__(self, entities, *, fail=False, unsub="UNSUB"):
        self._entities = entities
        self._fail = fail
        self._unsub = unsub
        self.subscribed = False

    async def device_info_and_list_entities(self):
        if self._fail:
            raise RuntimeError("device_info boom")
        info = types.SimpleNamespace(name="Zakhar", esphome_version="2024.1")
        return info, self._entities, []

    # Bound onto pipeline.send_event/send_audio by _on_connect.
    async def send_voice_assistant_event(self, *a, **k):
        return None

    async def send_voice_assistant_audio(self, *a, **k):
        return None

    # Bound onto pipeline.send_announcement by _on_connect (early-filler channel).
    async def send_voice_assistant_announcement_await_response(self, *a, **k):
        return None

    def subscribe_voice_assistant(self, *, handle_start, handle_stop, handle_audio):
        self.subscribed = True
        return self._unsub


def _connect_client(name="dev", *, entities=None, fail=False, unsub="UNSUB"):
    """Build a DeviceClient via __new__ wired for _on_connect/_on_disconnect."""
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg(name)
    c.online = False
    c._unsub = None
    c._capture_button_key = None
    c._capture_seconds_key = None
    c.pipeline = _ConnectPipeline()
    c.cli = _ConnectCli(entities if entities is not None else [], fail=fail, unsub=unsub)
    return c


async def test_on_connect_wires_subscribes_and_discovers_keys():
    ents = [
        _Ent("capture_seconds", 22),
        _Ent("capture_sample", 11),
    ]
    c = _connect_client(entities=ents, unsub="TOKEN")
    await c._on_connect()
    # Emitters rebound to the live connection: stage events go through the
    # StageEvent -> VAET translator, audio straight to the client.
    assert c.pipeline.send_event == c._send_stage_event
    assert c.pipeline.send_audio == c.cli.send_voice_assistant_audio
    # Subscribed and the unsub token retained.
    assert c.cli.subscribed is True
    assert c._unsub == "TOKEN"
    # Capture keys discovered from the entity list.
    assert c._capture_button_key == 11
    assert c._capture_seconds_key == 22
    assert c.online is True


async def test_on_connect_swallows_device_info_failure_but_subscribes():
    # A transient device_info failure must NOT leave the speaker unsubscribed/offline.
    c = _connect_client(fail=True, unsub="TOKEN")
    await c._on_connect()  # must not raise
    assert c.cli.subscribed is True
    assert c._unsub == "TOKEN"
    assert c.online is True
    # Discovery never ran (the failing call aborted before it), keys stay None.
    assert c._capture_button_key is None and c._capture_seconds_key is None


async def test_on_disconnect_clears_unsub_and_online():
    c = _connect_client()
    c._unsub = "TOKEN"
    c.online = True
    await c._on_disconnect(expected=False)
    assert c._unsub is None
    assert c.online is False


# --- StageEvent -> VAET translation ------------------------------------------

def test_event_to_vaet_covers_every_stage_event():
    # Pins table completeness: every transport-neutral StageEvent must have an
    # ESPHome wire mapping, or a pipeline emit would KeyError at runtime.
    from src.esphome_client import _EVENT_TO_VAET
    from src.pipeline_events import StageEvent
    assert set(_EVENT_TO_VAET) == set(StageEvent)


def test_send_stage_event_translates_to_vaet():
    from aioesphomeapi import VoiceAssistantEventType as VAET

    from src.esphome_client import DeviceClient
    from src.pipeline_events import StageEvent

    sent = []
    c = DeviceClient.__new__(DeviceClient)
    c.cli = types.SimpleNamespace(
        send_voice_assistant_event=lambda et, data: sent.append((et, data))
    )
    c._send_stage_event(StageEvent.STT_END, {"text": "hi"})
    assert sent == [(VAET.VOICE_ASSISTANT_STT_END, {"text": "hi"})]


# --- DeviceClient.stop failure isolation -------------------------------------

class _StopReconnect:
    def __init__(self, fail=False):
        self.fail = fail
        self.stopped = False

    async def stop(self):
        self.stopped = True
        if self.fail:
            raise RuntimeError("reconnect.stop boom")


class _StopCli:
    def __init__(self, fail=False):
        self.fail = fail
        self.disconnect_calls = []  # force kwarg

    async def disconnect(self, force=False):
        self.disconnect_calls.append(force)
        if self.fail:
            raise RuntimeError("disconnect boom")


async def test_stop_swallows_failures_and_still_disconnects():
    from src.esphome_client import DeviceClient
    c = DeviceClient.__new__(DeviceClient)
    c.cfg = _Cfg("dev")
    c.reconnect = _StopReconnect(fail=True)
    c.cli = _StopCli(fail=True)
    await c.stop()  # neither failure may propagate
    # disconnect(force=True) is attempted even after reconnect.stop raised.
    assert c.reconnect.stopped is True
    assert c.cli.disconnect_calls == [True]
