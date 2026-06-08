import types

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
