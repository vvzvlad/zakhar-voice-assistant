"""In-memory broadcast hub: pushes finalized pipeline runs to live panel WS clients."""

import asyncio
import contextlib
import json

from loguru import logger

from src.runs_store import summary_row


async def record_run(rt, record: dict) -> None:
    """Insert a finished run into the run log and push it to live panel
    subscribers; a failure is logged, never raised. No-op when the log is off."""
    store, events = rt.runs_store, rt.run_events
    if store is None:
        return
    try:
        run_id = await asyncio.to_thread(store.insert, record)
        if events is not None:
            await events.broadcast({"type": "run", "run": summary_row(record, run_id)})
    except Exception as e:
        logger.error(f"run record failed: {e}")


class RunEventsHub:
    """Fan-out of finalized runs to connected WebSocket panel clients.

    Created once in the app composition root and shared by the pipeline (producer,
    via broadcast()) and the panel API (consumers register/unregister their sockets).
    Two producers broadcast (pipeline runs + the panel system heartbeat), so the
    per-client send loop is serialized by an asyncio.Lock to keep their sends from
    interleaving on one socket. A send failure drops that client; it never propagates
    to the caller.
    """

    def __init__(self):
        self._clients: set = set()
        # Two producers now broadcast to the same sockets: pipeline-run frames and the
        # panel's periodic system heartbeat. Serialize the per-client send loop so their
        # `await ws.send_str(...)` calls can never interleave on one socket. In Python
        # 3.10+ asyncio.Lock() needs no running loop at construction, so building the hub
        # in the composition root is safe.
        self._send_lock = asyncio.Lock()

    def register(self, ws) -> None:
        self._clients.add(ws)

    def unregister(self, ws) -> None:
        self._clients.discard(ws)

    def count(self) -> int:
        return len(self._clients)

    async def broadcast(self, payload: dict) -> None:
        """Send `payload` (JSON) to every client; silently drop dead ones."""
        if not self._clients:
            return
        data = json.dumps(payload, ensure_ascii=False)
        # Hold the lock for the whole send loop so a concurrent pipeline-run broadcast
        # and a heartbeat broadcast never interleave their sends on the same socket.
        async with self._send_lock:
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_str(data)
                except Exception as e:  # noqa: BLE001 - a broken client must not break others
                    logger.debug(f"run-events: dropping dead client: {e}")
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    async def close_all(self) -> None:
        """Close every registered client and clear the set, used on panel shutdown.

        The panel's `_runs_stream` handlers block in `async for _msg in ws`, so on
        shutdown aiohttp's `AppRunner.cleanup()` would otherwise wait the full
        `shutdown_timeout` for them to finish — making the process hang on Ctrl+C.
        Closing the sockets here unblocks those handlers immediately. Closing one
        client must not prevent closing the others, so each close is suppressed.
        """
        for ws in list(self._clients):
            with contextlib.suppress(Exception):
                await ws.close()
        self._clients.clear()
