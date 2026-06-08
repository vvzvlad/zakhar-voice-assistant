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
    """Fake APIClient recording number_command/button_command calls."""

    def __init__(self):
        self.numbers = []  # (key, state)
        self.buttons = []  # key

    def number_command(self, key, state, device_id=0):
        self.numbers.append((key, state))

    def button_command(self, key, device_id=0):
        self.buttons.append(key)


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
    # object_id = slugify(name) for the "Zakhar Capture Seconds" / "Zakhar Capture
    # Sample" entities in esphome/zakhar-voice.yaml (NOT the YAML `id:` field). They
    # MUST stay equal to slugify(name) or discovery silently fails on real hardware.
    c = _capture_client(btn_key=None, sec_key=None)
    c._discover_capture_keys([
        _Ent("some_other", 1),
        _Ent("zakhar_capture_seconds", 22),
        _Ent("zakhar_capture_sample", 11),
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


def _device(name, host="10.0.0.1", psk="psk0"):
    return DeviceConfig(name=name, host=host, psk=psk)


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


@pytest.mark.asyncio
async def test_reconfigure_noop_when_unchanged(monkeypatch):
    a = _device("a")
    mgr, _rt = _make_manager([a], monkeypatch=monkeypatch)
    existing = mgr.clients[0]
    await mgr.reconfigure()   # config unchanged
    assert mgr.clients == [existing]
    assert existing.started is False and existing.stopped is False  # untouched
