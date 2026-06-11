"""Agent-facing MCP server (FastMCP over streamable HTTP).

Exposes the assistant ITSELF as an MCP server so an external agent (e.g. Claude)
can read the voice run log, view/change the live config, speak arbitrary text on
a speaker and send full text requests through the assistant pipeline. Runs in a
TRUSTED zone — there is NO authentication, same posture as the admin panel. The
enabled flag and bind host/port come from `core.agent_mcp` in data/config.json
(panel-editable on the System page, hot-applied via the reconfigurator — no
restart). The endpoint is http://<host>:<port>/mcp (streamable HTTP).

Everything is read live THROUGH the Runtime holder (rt.svc / rt.runs_store /
rt.manager) at call time, so hot-reloads and runs-store swaps take effect without
a rebuild. Every tool catches its exceptions and returns an error payload (a dict
with an "error" key) — it never raises, mirroring the builtin-tool contract in
src/builtin_mcp/reminders.py.
"""

import asyncio
import contextlib

import uvicorn
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError


def _find_client(manager, name: str):
    """Live DeviceClient by configured name, or None when unknown."""
    return next((c for c in manager.clients if c.cfg.name == name), None)


def _first_online(manager):
    """First currently-online DeviceClient, or None."""
    return next((c for c in manager.clients if c.online), None)


def build_agent_mcp(rt) -> FastMCP:
    """Build the FastMCP server exposing the assistant to external agents.

    `rt` is the live Runtime holder; every tool reads rt.svc / rt.runs_store /
    rt.manager at CALL time (never captured), so hot-swapped subsystems apply.
    """
    mcp = FastMCP("zakhar-voice-assistant", stateless_http=True, json_response=True)

    @mcp.tool(
        name="list_runs",
        description=(
            "List recent voice interactions (newest first): what the user said "
            "(stt_text), what the assistant replied (llm_text), result and stage "
            "timings. Optional filters: device (exact speaker name), search "
            "(substring match on request/reply text), limit (max rows)."
        ),
    )
    async def list_runs(
        limit: int = 20, device: str | None = None, search: str | None = None
    ) -> dict:
        try:
            store = rt.runs_store  # read at call time: may be None or hot-swapped
            if store is None:
                return {"error": "run log is disabled (core.runs.enabled = false)"}
            runs = await asyncio.to_thread(
                store.list, device=device, search=search, limit=limit
            )
            return {"runs": runs}
        except Exception as e:
            return {"error": f"failed to list runs: {e}"}

    @mcp.tool(
        name="get_run",
        description=(
            "Full record of one voice interaction by id: user text, assistant "
            "reply, the agentic tool-calling rounds, stage timings and any error."
        ),
    )
    async def get_run(run_id: int) -> dict:
        try:
            store = rt.runs_store
            if store is None:
                return {"error": "run log is disabled (core.runs.enabled = false)"}
            run = await asyncio.to_thread(store.get, run_id)
            if run is None:
                return {"error": f"run {run_id} not found"}
            return run
        except Exception as e:
            return {"error": f"failed to get run: {e}"}

    @mcp.tool(
        name="get_config",
        description=(
            "The full live configuration document of the assistant (stage "
            "providers, core settings, devices, tool sources). Plain JSON, the "
            "same document the admin panel edits."
        ),
    )
    async def get_config() -> dict:
        try:
            return rt.svc.document()
        except Exception as e:
            return {"error": f"failed to read config: {e}"}

    @mcp.tool(
        name="update_config",
        description=(
            "Apply a deep-merge patch to the assistant configuration (same "
            "semantics as the admin panel). Pass only the keys to change, e.g. "
            '{"core": {"log_level": "DEBUG"}}. Changes apply LIVE (hot-reload, '
            "no restart). Returns {ok: true, config: <new document>} or "
            "{ok: false, error: <validation message>}."
        ),
    )
    async def update_config(patch: dict) -> dict:
        try:
            rt.svc.apply(patch)
        except (ValidationError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"failed to apply config patch: {e}"}
        return {"ok": True, "config": rt.svc.document()}

    @mcp.tool(
        name="list_devices",
        description=(
            "List the configured speakers with their live status "
            "(name, host, enabled, online)."
        ),
    )
    async def list_devices() -> dict:
        try:
            return {"devices": rt.manager.statuses()}
        except Exception as e:
            return {"error": f"failed to list devices: {e}"}

    @mcp.tool(
        name="say",
        description=(
            "Speak arbitrary text out loud on a speaker (TTS + announce channel; "
            "ducks any current audio). device: speaker name from list_devices; "
            "omitted = the first online speaker."
        ),
    )
    async def say(text: str, device: str | None = None) -> dict:
        try:
            manager = rt.manager
            if device is not None:
                client = _find_client(manager, device)
                if client is None:
                    return {"error": f"unknown device {device!r}"}
            else:
                client = _first_online(manager)
                if client is None:
                    return {"error": "no online speakers"}
            await client.announce(text)
            return {"ok": True, "device": client.cfg.name}
        except Exception as e:
            return {"error": f"say failed: {e}"}

    @mcp.tool(
        name="ask",
        description=(
            "Send a text request to the assistant as if the user spoke it: a "
            "full LLM turn with the smart-home tools and the device's "
            "conversation context. The reply is spoken on the speaker when "
            "speak=true and the device is online, and is always returned as "
            "text. device: speaker name from list_devices; omitted = the first "
            "online (else first configured) speaker."
        ),
    )
    async def ask(text: str, device: str | None = None, speak: bool = True) -> dict:
        try:
            manager = rt.manager
            if device is not None:
                client = _find_client(manager, device)
                if client is None:
                    return {"error": f"unknown device {device!r}"}
            else:
                # Prefer an online speaker; fall back to any configured one so the
                # turn still runs (text-only) when everything is offline.
                client = _first_online(manager)
                if client is None and manager.clients:
                    client = manager.clients[0]
                if client is None:
                    return {"error": "no devices configured"}
            do_speak = speak and client.online
            result = await client.pipeline.run_text(text, speak=do_speak)
            # "Actually spoken": speaking was requested AND there was a reply AND
            # the TTS/announce step did not fail. (When an earlier stage already
            # owns error_stage a TTS failure is invisible here — acceptable.)
            spoken = (
                do_speak
                and bool(result.get("reply"))
                and result.get("error_stage") != "TTS"
            )
            return {**result, "device": client.cfg.name, "spoken": spoken}
        except Exception as e:
            return {"error": f"ask failed: {e}"}

    return mcp


class _NoSignalUvicornServer(uvicorn.Server):
    # Uvicorn must not steal the process SIGINT/SIGTERM handlers: the app owns
    # shutdown (app.py relies on KeyboardInterrupt/CancelledError out of
    # asyncio.run), so the stock capture_signals() would break Ctrl+C.
    @contextlib.contextmanager
    def capture_signals(self):
        yield


class AgentMcpServer:
    """Serves the agent-facing FastMCP app with uvicorn inside the app's event loop.

    start/stop mirror PanelServer/AudioServer naming. The uvicorn lifespan runs
    the FastMCP streamable-HTTP session manager (the ASGI app provides its own
    lifespan), so nothing else needs to be started.
    """

    def __init__(self, rt, host: str, port: int):
        self.rt = rt
        self.host = host
        self.port = port
        self._server: _NoSignalUvicornServer | None = None
        self._task: asyncio.Task | None = None

    @staticmethod
    async def _serve(server: "_NoSignalUvicornServer") -> None:
        """Run uvicorn's serve(), converting SystemExit into a plain exception.

        uvicorn calls sys.exit(1) when the bind fails; a SystemExit escaping an
        asyncio Task is re-raised into the event loop and would kill the whole
        app, so it must be downgraded to an exception start() can retrieve."""
        try:
            await server.serve()
        except SystemExit as e:
            raise RuntimeError(f"uvicorn exited (code {e.code})") from e

    async def start(self) -> None:
        """Start serving; raises RuntimeError if the server fails to bind."""
        app = build_agent_mcp(self.rt).streamable_http_app()
        config = uvicorn.Config(
            app, host=self.host, port=self.port, log_level="warning", lifespan="on"
        )
        server = _NoSignalUvicornServer(config)
        self._server = server
        task = asyncio.create_task(self._serve(server))
        self._task = task
        # serve() is fire-and-forget, so a taken port would "succeed" silently and
        # the task would die later. Wait until uvicorn reports it is listening
        # (Server.started flips True) or the task completes/times out, so callers
        # see bind failures as an exception.
        deadline = asyncio.get_running_loop().time() + 5.0
        while not server.started:
            if task.done() or asyncio.get_running_loop().time() > deadline:
                self._task = None
                self._server = None
                if task.done():
                    exc = task.exception()
                    reason = str(exc) if exc is not None else "server task exited"
                else:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    reason = "timed out waiting for the server to start"
                logger.error(f"agent MCP server failed to start: {reason}")
                raise RuntimeError(
                    f"agent MCP server failed to bind {self.host}:{self.port}: {reason}"
                )
            await asyncio.sleep(0.05)
        logger.info(f"agent MCP server on http://{self.host}:{self.port}/mcp")

    async def stop(self) -> None:
        """Idempotent shutdown; safe when start() failed or never ran."""
        task, self._task = self._task, None
        server, self._server = self._server, None
        if server is not None:
            server.should_exit = True
        if task is None:
            return
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            # The graceful exit flag was ignored (e.g. a stuck connection):
            # cancel the serve task outright.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
