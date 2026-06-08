"""Multi-source tool hub: aggregates several tool sources behind one interface.

The LLM tool-loop only ever touches `hub.tools` (the merged, namespaced tool list)
and `hub.call(name, args)` (routed to the owning source). This lets the model see,
through ONE interface, both the external smart-home MCP server (HttpMcpSource) and
in-process built-in MCP servers (BuiltinMcpSource, e.g. weather).

Tool names are namespaced as `f"{source.id}__{raw_name}"` so (a) names never collide
across sources and (b) the hub knows which source owns each advertised tool. The
prefix is stripped before the call reaches the owning source.

Failure isolation mirrors McpToolHub: one source failing to start/refresh must NOT
break the others, and a tool call never raises — it returns an error string instead.
"""

import copy

from loguru import logger
from mcp.server.fastmcp import FastMCP

from src.mcp_client import McpToolHub


class ToolSource:
    """Base interface for a source of tools the hub can aggregate.

    A source exposes un-prefixed, groq-shape tool dicts and executes calls by their
    raw (un-prefixed) name. The hub owns namespacing and routing.
    """

    id: str

    async def start(self) -> None:
        """Initial load of this source's tools."""
        raise NotImplementedError

    async def ensure(self) -> None:
        """Refresh this source's tools (may be a no-op for static sources)."""
        raise NotImplementedError

    def raw_tools(self) -> list[dict]:
        """Un-prefixed groq-shape tool dicts:
        {"type": "function", "function": {"name", "description", "parameters"}}."""
        raise NotImplementedError

    async def call(self, raw_name: str, args: dict) -> str:
        """Execute a tool by its raw (un-prefixed) name; return plain text."""
        raise NotImplementedError


class HttpMcpSource(ToolSource):
    """Wraps an existing McpToolHub (the external smart-home MCP server).

    McpToolHub already handles short-lived sessions, graceful start and error-as-string
    calls; this adapter just delegates and is intentionally thin.
    """

    def __init__(self, id: str, hub: McpToolHub):
        self.id = id
        self._hub = hub

    async def start(self) -> None:
        await self._hub.start()

    async def ensure(self) -> None:
        await self._hub.ensure_tools()

    def raw_tools(self) -> list[dict]:
        return self._hub.tools

    async def call(self, raw_name: str, args: dict) -> str:
        return await self._hub.call(raw_name, args)

    async def stop(self) -> None:
        await self._hub.stop()


class BuiltinMcpSource(ToolSource):
    """Wraps an in-process FastMCP instance, dialed directly (no transport/session).

    Tools are static, so they are listed once on start() and cached. Calls go through
    FastMCP.call_tool, whose return shape varies across SDK versions; we normalize it
    to plain text defensively here.
    """

    def __init__(self, id: str, server: FastMCP):
        self.id = id
        self._server = server
        self._tools: list[dict] = []

    async def start(self) -> None:
        # FastMCP.list_tools() returns list[mcp.types.Tool] (.name/.description/
        # .inputSchema); reuse McpToolHub's converter for an identical groq shape.
        tools = await self._server.list_tools()
        self._tools = [McpToolHub._to_groq_tool(t) for t in tools]

    async def ensure(self) -> None:
        # Built-in tools are static: nothing to refresh.
        return None

    def raw_tools(self) -> list[dict]:
        return self._tools

    async def call(self, raw_name: str, args: dict) -> str:
        try:
            res = await self._server.call_tool(raw_name, args)
            return self._normalize(res)
        except Exception as e:
            # Never raise: mirror McpToolHub.call so a tool error reaches the model
            # as text rather than crashing the run.
            return f"error calling {raw_name}: {e}"

    @staticmethod
    def _normalize(res) -> str:
        """Flatten a FastMCP call_tool result to text.

        This SDK version returns a tuple ([TextContent(...), ...], {"result": ...}).
        Other versions may return just the content sequence or a dict. Handle all:
        - tuple -> take element 0 (the content list);
        - dict  -> stringify a "result" key if present, else the whole dict;
        - sequence of content blocks -> join their .text attributes.
        """
        if isinstance(res, tuple):
            res = res[0] if res else []
        if isinstance(res, dict):
            return str(res.get("result", res))
        try:
            return "\n".join(getattr(c, "text", "") for c in res)
        except TypeError:
            return str(res)


class ToolHub:
    """Drop-in replacement for the single MCP hub: aggregates N ToolSources.

    Exposes the same surface the LLM loop expects: a `tools` property (merged,
    namespaced advertised tool list) and an async `call(name, args)` that routes by
    the advertised name to the owning source with the RAW name.
    """

    def __init__(self, sources: list[ToolSource]):
        self._sources = sources
        self._advertised: list[dict] = []
        # advertised_name -> (source, raw_name)
        self._routes: dict[str, tuple[ToolSource, str]] = {}

    def _rebuild(self) -> None:
        """Recompute the merged advertised list + routing map from current sources.

        Each raw tool dict is deep-copied and its function name overwritten with the
        prefixed name, so the original source dicts are never mutated.
        """
        advertised: list[dict] = []
        routes: dict[str, tuple[ToolSource, str]] = {}
        for source in self._sources:
            for tool in source.raw_tools():
                raw_name = tool["function"]["name"]
                advertised_name = f"{source.id}__{raw_name}"
                clone = copy.deepcopy(tool)
                clone["function"]["name"] = advertised_name
                advertised.append(clone)
                routes[advertised_name] = (source, raw_name)
        self._advertised = advertised
        self._routes = routes

    async def start(self) -> None:
        # Start each source; one failing must NOT break the others (graceful, like
        # McpToolHub.start). Log and continue.
        for source in self._sources:
            try:
                await source.start()
            except Exception as e:
                logger.error(f"tool source {source.id!r} start failed: {e}")
        self._rebuild()

    async def ensure_tools(self) -> None:
        # Called by the LLM loop each run. Refresh each source independently.
        for source in self._sources:
            try:
                await source.ensure()
            except Exception as e:
                logger.warning(f"tool source {source.id!r} refresh failed: {e}")
        self._rebuild()

    @property
    def tools(self) -> list:
        return self._advertised or []

    async def call(self, name: str, args: dict) -> str:
        route = self._routes.get(name)
        if route is None:
            return f"error: unknown tool {name}"
        source, raw_name = route
        return await source.call(raw_name, args)

    async def stop(self) -> None:
        # Best-effort: stop each source that supports it.
        for source in self._sources:
            stop = getattr(source, "stop", None)
            if stop is None:
                continue
            try:
                await stop()
            except Exception as e:
                logger.warning(f"tool source {source.id!r} stop failed: {e}")
