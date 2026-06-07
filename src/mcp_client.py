"""MCP client hub: short-lived per-operation sessions to the smart-home MCP server."""

import asyncio

from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class McpToolHub:
    """MCP client for the smart-home MCP server.

    Opens a fresh short-lived session per operation (list_tools / call_tool). The
    smart-home MCP server runs stateless_http and is an independently auto-updated
    container, so its restarts are routine; reconnecting per call makes the
    integration self-healing (no stale persistent session to break) and keeps every
    context manager entered and exited in the caller's own task (anyio-safe).
    """

    def __init__(self, url: str, token: str | None = None):
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"} if token else None
        self._tools: list = []
        self._lock = asyncio.Lock()  # serialize tool calls across concurrent speakers

    async def _list_tools(self) -> list:
        async with streamablehttp_client(self._url, headers=self._headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.list_tools()
                return [self._to_groq_tool(t) for t in resp.tools]

    async def start(self) -> None:
        # Graceful: a smart-home outage must NOT crash the assistant. On failure log
        # and continue with an empty tool list (refreshed lazily on first use).
        try:
            self._tools = await self._list_tools()
            logger.info(f"MCP connected: {[t['function']['name'] for t in self._tools]}")
        except Exception as e:
            logger.error(f"MCP connect failed ({self._url}): {e}; will retry on demand")
            self._tools = []

    async def ensure_tools(self) -> None:
        # Self-heal the startup race: if the server was unreachable when start() ran,
        # pick the tools up on the first utterance after it becomes reachable.
        if self._tools:
            return
        try:
            self._tools = await self._list_tools()
            if self._tools:
                logger.info(f"MCP tools loaded: {[t['function']['name'] for t in self._tools]}")
        except Exception as e:
            logger.warning(f"MCP tools still unavailable ({self._url}): {e}")

    @staticmethod
    def _to_groq_tool(tool) -> dict:
        return {"type": "function", "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}}}}

    @property
    def tools(self) -> list:
        return self._tools

    async def call(self, name: str, arguments: dict) -> str:
        async with self._lock:
            try:
                async with streamablehttp_client(self._url, headers=self._headers) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(name, arguments)
            except Exception as e:
                return f"error calling {name}: {e}"
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []))
        return text or "(no output)"

    async def stop(self) -> None:
        # Sessions are per-operation; nothing persistent to close. Kept for API symmetry.
        return None
