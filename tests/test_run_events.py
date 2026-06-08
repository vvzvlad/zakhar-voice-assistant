"""Unit tests for the in-memory run-events broadcast hub (src.run_events)."""

import json

from src.run_events import RunEventsHub


class FakeWS:
    """WebSocket double: records every JSON string sent over send_str()."""

    def __init__(self):
        self.sent = []

    async def send_str(self, data):
        self.sent.append(data)


class RaisingWS:
    """WebSocket double whose send_str always fails (a dead client)."""

    async def send_str(self, data):
        raise RuntimeError("socket is dead")


async def test_broadcast_reaches_all_clients():
    hub = RunEventsHub()
    a, b = FakeWS(), FakeWS()
    hub.register(a)
    hub.register(b)
    assert hub.count() == 2

    payload = {"type": "run", "run": {"id": 1}}
    await hub.broadcast(payload)

    assert [json.loads(s) for s in a.sent] == [payload]
    assert [json.loads(s) for s in b.sent] == [payload]


async def test_unregister_removes_client():
    hub = RunEventsHub()
    a = FakeWS()
    hub.register(a)
    assert hub.count() == 1
    hub.unregister(a)
    assert hub.count() == 0

    await hub.broadcast({"type": "run", "run": {"id": 2}})
    assert a.sent == []


async def test_dead_client_is_dropped_and_does_not_block_others():
    hub = RunEventsHub()
    dead = RaisingWS()
    alive = FakeWS()
    hub.register(dead)
    hub.register(alive)
    assert hub.count() == 2

    payload = {"type": "run", "run": {"id": 3}}
    await hub.broadcast(payload)

    # The healthy client still received the payload despite the raising one.
    assert [json.loads(s) for s in alive.sent] == [payload]
    # The dead client was discarded from the set.
    assert hub.count() == 1


async def test_broadcast_with_no_clients_is_noop():
    hub = RunEventsHub()
    # Must not raise with zero registered clients.
    await hub.broadcast({"type": "run", "run": {"id": 4}})
    assert hub.count() == 0
