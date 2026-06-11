"""Tests for the agent-facing MCP server (src/agent_mcp.py).

Tools are exercised through a REAL MCP client session connected in-memory (no
sockets) via mcp.shared.memory.create_connected_server_and_client_session, so
the schema/serialization path is the same one an external agent hits. The
Runtime holder is a SimpleNamespace with stubs for svc/manager and (where the
test needs it) a real RunsStore on a tmp sqlite file.
"""

import json
from types import SimpleNamespace

from mcp.shared.memory import create_connected_server_and_client_session

from src.agent_mcp import build_agent_mcp
from src.runs_store import RunsStore


class StubSvc:
    """ConfigService double: canned document(), apply() records or raises."""

    def __init__(self, doc=None, apply_error=None):
        self.doc = doc if doc is not None else {"core": {"log_level": "INFO"}}
        self.applied = []
        self.apply_error = apply_error

    def document(self):
        return self.doc

    def apply(self, patch):
        if self.apply_error is not None:
            raise self.apply_error
        self.applied.append(patch)


class StubPipeline:
    """Pipeline double exposing only the async run_text the ask tool calls."""

    def __init__(self, reply="готово"):
        self.reply = reply
        self.calls = []  # (text, speak) per run_text call

    async def run_text(self, text, speak=True):
        self.calls.append((text, speak))
        return {
            "reply": self.reply,
            "result": "ok",
            "error_stage": None,
            "error_text": None,
        }


class StubClient:
    """DeviceClient double: cfg.name, online flag, announce recorder, pipeline."""

    def __init__(self, name, online=True):
        self.cfg = SimpleNamespace(name=name)
        self.online = online
        self.announced = []
        self.pipeline = StubPipeline()

    async def announce(self, text):
        if not self.online:
            raise RuntimeError(f"{self.cfg.name} is offline")
        self.announced.append(text)


def make_rt(clients=(), runs_store=None, svc=None):
    """Fake Runtime holder: only the attributes the agent MCP tools read."""
    clients = list(clients)
    manager = SimpleNamespace(
        clients=clients,
        statuses=lambda: [
            {"name": c.cfg.name, "host": "h", "enabled": True, "online": c.online}
            for c in clients
        ],
    )
    return SimpleNamespace(svc=svc or StubSvc(), runs_store=runs_store, manager=manager)


def payload(result):
    """Decode a CallToolResult back into the dict the tool returned."""
    assert result.isError is False  # tools never raise; errors are payloads
    if result.structuredContent is not None:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def call(rt, name, args=None):
    """Call one tool over an in-memory client session and return its payload."""
    server = build_agent_mcp(rt)._mcp_server
    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool(name, args or {})
    return payload(result)


def make_store(tmp_path):
    """Real RunsStore on a tmp sqlite file, pre-filled with two runs."""
    store = RunsStore(str(tmp_path / "runs.db"))
    store.insert({
        "ts": 1.0, "device": "kitchen", "result": "ok", "reason": "endpoint",
        "stt_text": "включи свет", "llm_text": "готово",
        "rounds": [{"round": 1, "note": "final answer", "calls": []}],
    })
    store.insert({
        "ts": 2.0, "device": "hall", "result": "error", "reason": "endpoint",
        "stt_text": "какая погода", "llm_text": "",
        "error_stage": "LLM", "error_text": "boom",
    })
    return store


# --- run log ------------------------------------------------------------------

async def test_list_runs_happy_path(tmp_path):
    rt = make_rt(runs_store=make_store(tmp_path))

    out = await call(rt, "list_runs", {"limit": 10})
    assert [r["device"] for r in out["runs"]] == ["hall", "kitchen"]  # newest first

    # Filters pass through to RunsStore.list.
    out = await call(rt, "list_runs", {"device": "kitchen"})
    assert [r["stt_text"] for r in out["runs"]] == ["включи свет"]
    out = await call(rt, "list_runs", {"search": "погода"})
    assert [r["device"] for r in out["runs"]] == ["hall"]


async def test_list_runs_disabled_store(tmp_path):
    rt = make_rt(runs_store=None)
    out = await call(rt, "list_runs", {})
    assert out == {"error": "run log is disabled (core.runs.enabled = false)"}


async def test_get_run_full_record_and_not_found(tmp_path):
    rt = make_rt(runs_store=make_store(tmp_path))

    out = await call(rt, "get_run", {"run_id": 1})
    assert out["stt_text"] == "включи свет"
    assert out["llm_text"] == "готово"
    assert out["rounds"] == [{"round": 1, "note": "final answer", "calls": []}]

    out = await call(rt, "get_run", {"run_id": 999})
    assert out == {"error": "run 999 not found"}


# --- config -------------------------------------------------------------------

async def test_get_config_returns_document():
    doc = {"core": {"log_level": "DEBUG"}, "llm": {"selected": "openrouter"}}
    rt = make_rt(svc=StubSvc(doc=doc))
    assert await call(rt, "get_config", {}) == doc


async def test_update_config_happy_path():
    svc = StubSvc(doc={"core": {"log_level": "DEBUG"}})
    rt = make_rt(svc=svc)
    patch = {"core": {"log_level": "DEBUG"}}

    out = await call(rt, "update_config", {"patch": patch})

    assert out == {"ok": True, "config": svc.doc}
    assert svc.applied == [patch]


async def test_update_config_validation_error_never_raises():
    rt = make_rt(svc=StubSvc(apply_error=ValueError("unknown key core.bogus")))

    out = await call(rt, "update_config", {"patch": {"core": {"bogus": 1}}})

    assert out == {"ok": False, "error": "unknown key core.bogus"}


# --- devices / say ------------------------------------------------------------

async def test_list_devices():
    rt = make_rt(clients=[StubClient("kitchen"), StubClient("hall", online=False)])
    out = await call(rt, "list_devices", {})
    assert [(d["name"], d["online"]) for d in out["devices"]] == [
        ("kitchen", True), ("hall", False),
    ]


async def test_say_on_named_device():
    kitchen, hall = StubClient("kitchen"), StubClient("hall")
    rt = make_rt(clients=[kitchen, hall])

    out = await call(rt, "say", {"text": "привет", "device": "hall"})

    assert out == {"ok": True, "device": "hall"}
    assert hall.announced == ["привет"]
    assert kitchen.announced == []


async def test_say_unknown_device():
    rt = make_rt(clients=[StubClient("kitchen")])
    out = await call(rt, "say", {"text": "привет", "device": "nope"})
    assert out == {"error": "unknown device 'nope'"}


async def test_say_defaults_to_first_online():
    offline, online = StubClient("a", online=False), StubClient("b")
    rt = make_rt(clients=[offline, online])

    out = await call(rt, "say", {"text": "привет"})

    assert out == {"ok": True, "device": "b"}
    assert online.announced == ["привет"]


async def test_say_no_online_device():
    rt = make_rt(clients=[StubClient("a", online=False)])
    out = await call(rt, "say", {"text": "привет"})
    assert out == {"error": "no online speakers"}


async def test_say_named_offline_device_returns_error():
    # announce() raises RuntimeError for an offline device; the tool catches it.
    rt = make_rt(clients=[StubClient("a", online=False)])
    out = await call(rt, "say", {"text": "привет", "device": "a"})
    assert "offline" in out["error"]


# --- ask ----------------------------------------------------------------------

async def test_ask_routes_to_named_client():
    kitchen, hall = StubClient("kitchen"), StubClient("hall")
    rt = make_rt(clients=[kitchen, hall])

    out = await call(rt, "ask", {"text": "включи свет", "device": "hall"})

    assert hall.pipeline.calls == [("включи свет", True)]
    assert kitchen.pipeline.calls == []
    assert out["device"] == "hall"
    assert out["reply"] == "готово"
    assert out["spoken"] is True


async def test_ask_offline_target_passes_speak_false():
    # An offline named target still answers (full LLM turn) but is never spoken to.
    target = StubClient("kitchen", online=False)
    rt = make_rt(clients=[target])

    out = await call(rt, "ask", {"text": "привет", "device": "kitchen"})

    assert target.pipeline.calls == [("привет", False)]
    assert out["spoken"] is False
    assert out["reply"] == "готово"


async def test_ask_speak_false_requested():
    target = StubClient("kitchen")
    rt = make_rt(clients=[target])

    out = await call(rt, "ask", {"text": "привет", "speak": False})

    assert target.pipeline.calls == [("привет", False)]
    assert out["spoken"] is False


async def test_ask_prefers_online_else_first_client():
    offline, online = StubClient("a", online=False), StubClient("b")
    rt = make_rt(clients=[offline, online])
    out = await call(rt, "ask", {"text": "привет"})
    assert out["device"] == "b"
    assert online.pipeline.calls == [("привет", True)]

    # All offline: falls back to the first configured client, text-only.
    only = StubClient("a", online=False)
    rt = make_rt(clients=[only])
    out = await call(rt, "ask", {"text": "привет"})
    assert out["device"] == "a"
    assert only.pipeline.calls == [("привет", False)]
    assert out["spoken"] is False


async def test_ask_no_devices_configured():
    rt = make_rt(clients=[])
    out = await call(rt, "ask", {"text": "привет"})
    assert out == {"error": "no devices configured"}
