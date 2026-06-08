"""In-memory broadcast hub: pushes finalized pipeline runs to live panel WS clients."""

import contextlib
import json

from loguru import logger


class RunEventsHub:
    """Fan-out of finalized runs to connected WebSocket panel clients.

    Created once in the app composition root and shared by the pipeline (producer,
    via broadcast()) and the panel API (consumers register/unregister their sockets).
    Everything runs on the asyncio event loop, so no lock is needed. A send failure
    drops that client; it never propagates to the caller.
    """

    def __init__(self):
        self._clients: set = set()

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
