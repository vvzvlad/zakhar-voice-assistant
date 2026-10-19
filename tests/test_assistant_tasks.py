"""Tests for jobs handed off to the ZeroClaw assistant (src.assistant_tasks) and its
built-in MCP tools: the signed webhook POST, the ledger, and speaking the replies."""

import asyncio
import hashlib
import hmac
import json
import types
from datetime import datetime

import httpx
import pytest
import respx

from src import context
from src.assistant_tasks import AssistantTasks, AssistantTasksStore
from src.builtin_mcp.assistant_tasks import build_assistant_tasks_server
from src.core_config import AssistantTasksConfig, ContextConfig
from src.run_context import current_device
from src.runs_store import RunsStore
from src.tool_hub import BuiltinMcpSource

URL = "http://zeroclaw.lan/webhook"


class _FakeDevice:
    """A DeviceManager client: name, online, pipeline.has_speaker and announce()."""

    def __init__(self, name, online=True, speaker=True, fail=False):
        self.name = name
        self.online = online
        self.pipeline = types.SimpleNamespace(has_speaker=speaker)
        self.fail = fail
        self.announced = []

    async def announce(self, text, attempts=None):
        if self.fail:
            raise RuntimeError("tts down")
        self.announced.append(text)
        if attempts is not None:
            attempts.append({"kind": "tts", "provider": "fake", "outcome": "spoke",
                             "error": None, "t": 120})


def _tasks(tmp_path, devices=()):
    tasks = AssistantTasks(AssistantTasksStore(str(tmp_path / "assistant_tasks.db")))
    cfg = AssistantTasksConfig(url=URL, secret="s3cret", task_suffix="Коротко.")
    for d in devices:
        d.pipeline._context_path = str(tmp_path / f"context_{d.name}.txt")
    tasks.rt = types.SimpleNamespace(
        core=types.SimpleNamespace(assistant_tasks=cfg, context=ContextConfig()),
        manager=types.SimpleNamespace(clients=list(devices)),
        runs_store=RunsStore(str(tmp_path / "runs.db")), run_events=None)
    return tasks


async def _reply_and_speak(tasks, payload):
    """on_reply, then wait until the replies it started speaking are done."""
    result = await tasks.on_reply(payload)
    await asyncio.gather(*list(tasks._speaking))
    return result


async def _call(tasks, tool, args):
    source = BuiltinMcpSource("assistant_tasks", build_assistant_tasks_server(tasks))
    await source.start()
    return await source.call(tool, args)


def test_delegate_task_posts_a_signed_task_from_this_speaker(tmp_path):
    tasks = _tasks(tmp_path)
    token = current_device.set("kitchen")
    try:
        with respx.mock:
            route = respx.post(URL).mock(return_value=httpx.Response(200))
            out = asyncio.run(_call(tasks, "delegate_task", {"text": "найди режим обжига"}))
    finally:
        current_device.reset(token)

    assert out == "Задача №1 передана ассистенту, ответ скажу, когда придёт."
    request = route.calls.last.request
    sent = request.content
    assert request.headers["X-Webhook-Signature"] == (
        "sha256=" + hmac.new(b"s3cret", sent, hashlib.sha256).hexdigest())
    assert json.loads(sent) == {"sender": "kitchen", "content": "найди режим обжига\n\nКоротко.",
                                "thread_id": "zakhar-1"}
    assert [(r["id"], r["device"], r["answer"]) for r in tasks.recent(5)] == [(1, "kitchen", None)]


@pytest.mark.parametrize("mock", [{"return_value": httpx.Response(503)},
                                  {"side_effect": httpx.ConnectError("refused")}])
def test_failed_handoff_leaves_no_task(tmp_path, mock):
    tasks = _tasks(tmp_path)
    with respx.mock:
        respx.post(URL).mock(**mock)
        out = asyncio.run(_call(tasks, "delegate_task", {"text": "x"}))

    assert out.startswith("Не удалось передать задачу ассистенту")
    assert tasks.recent(5) == []


def test_reply_is_stored_and_spoken_on_the_task_speaker(tmp_path):
    kitchen, bedroom = _FakeDevice("kitchen"), _FakeDevice("bedroom")
    tasks = _tasks(tmp_path, [kitchen, bedroom])
    task_id = tasks.store.insert("погода в Токио", "bedroom")
    sent = []

    class _Events:
        async def broadcast(self, payload):
            sent.append(payload)

    tasks.rt.run_events = _Events()

    async def main():
        first = await _reply_and_speak(tasks, {"content": "В Токио +20.", "thread_id": f"zakhar-{task_id}"})
        second = await _reply_and_speak(tasks, {"content": "И дождь.", "recipient": f"zakhar-{task_id}"})
        return first, second

    assert asyncio.run(main()) == ((200, {"id": task_id}),) * 2
    assert bedroom.announced == ["В Токио +20.", "И дождь."]
    assert kitchen.announced == []
    row = tasks.store.get(task_id)
    assert row["answer"] == "В Токио +20.\n\nИ дождь."
    assert row["delivered"] == 1
    assert row["answered_ts"] is not None
    # Each reply is a run in the log, newest first, and a line of the speaker's dialog.
    runs = tasks.rt.runs_store.list()
    assert [(r["device"], r["reason"], r["result"], r["stt_text"], r["llm_text"], r["t_tts"])
            for r in runs] == [("bedroom", "assistant", "ok", "погода в Токио", "И дождь.", 120),
                               ("bedroom", "assistant", "ok", "погода в Токио", "В Токио +20.", 120)]
    # ...and reaches an open panel at once.
    assert [(e["type"], e["run"]["id"], e["run"]["llm_text"]) for e in sent] == [
        ("run", runs[1]["id"], "В Токио +20."), ("run", runs[0]["id"], "И дождь.")]
    assert context.load_context(bedroom.pipeline._context_path) == [
        {"role": "assistant", "content": "В Токио +20."},
        {"role": "assistant", "content": "И дождь."}]


@pytest.mark.parametrize("device,shown,run", [
    ({"online": False}, [], ("error", "device")),
    ({"speaker": False}, ["ответ"], ("ok", None)),
    ({"fail": True}, [], ("error", "TTS")),
])
def test_reply_that_cannot_be_spoken_is_kept_undelivered(tmp_path, device, shown, run):
    kitchen = _FakeDevice("kitchen", **device)
    tasks = _tasks(tmp_path, [kitchen])
    task_id = tasks.store.insert("x", "kitchen")

    result = asyncio.run(_reply_and_speak(tasks, {"content": "ответ", "thread_id": f"zakhar-{task_id}"}))

    assert result == (200, {"id": task_id})
    assert kitchen.announced == shown  # a device without a speaker still shows the text
    row = tasks.store.get(task_id)
    assert (row["answer"], row["delivered"]) == ("ответ", 0)
    # The log keeps the reply either way; only a reply the device got joins the dialog.
    [logged] = tasks.rt.runs_store.list()
    assert (logged["result"], logged["llm_text"]) == (run[0], "ответ")
    assert tasks.rt.runs_store.get(logged["id"])["error_stage"] == run[1]
    assert context.load_context(kitchen.pipeline._context_path) == (
        [{"role": "assistant", "content": "ответ"}] if shown else [])


def test_reply_is_answered_before_it_is_spoken(tmp_path):
    # ZeroClaw re-sends a reply whose request fails: the request must not wait for the speech.
    kitchen = _FakeDevice("kitchen")
    tasks = _tasks(tmp_path, [kitchen])
    task_id = tasks.store.insert("x", "kitchen")

    async def main():
        release = asyncio.Event()
        original = kitchen.announce

        async def slow_announce(text, attempts=None):
            await release.wait()
            await original(text, attempts)

        kitchen.announce = slow_announce
        result = await asyncio.wait_for(tasks.on_reply({"content": "ответ", "thread_id": f"zakhar-{task_id}"}), 1)
        spoken_before = list(kitchen.announced)
        release.set()
        await asyncio.gather(*list(tasks._speaking))
        return result, spoken_before

    assert asyncio.run(main()) == ((200, {"id": task_id}), [])
    assert kitchen.announced == ["ответ"]
    assert tasks.store.get(task_id)["delivered"] == 1


def test_an_unspoken_newer_reply_keeps_the_task_undelivered(tmp_path, monkeypatch):
    # The second reply arrives before the first is spoken and then fails to speak: the
    # first one's success must not mark the task delivered.
    clock = iter(range(1000, 2000))
    monkeypatch.setattr("src.assistant_tasks.time.time", lambda: next(clock))
    kitchen = _FakeDevice("kitchen")
    tasks = _tasks(tmp_path, [kitchen])
    task_id = tasks.store.insert("x", "kitchen")
    original = kitchen.announce

    async def announce_once(text, attempts=None):
        kitchen.announce = _FakeDevice("kitchen", fail=True).announce
        await original(text, attempts)

    kitchen.announce = announce_once

    async def main():
        await tasks.on_reply({"content": "первый", "thread_id": f"zakhar-{task_id}"})
        await tasks.on_reply({"content": "второй", "thread_id": f"zakhar-{task_id}"})
        await asyncio.gather(*list(tasks._speaking))

    asyncio.run(main())
    assert kitchen.announced == ["первый"]
    assert tasks.store.get(task_id)["delivered"] == 0


@pytest.mark.parametrize("payload", [{"content": "x", "thread_id": "zakhar-99"},
                                     {"content": "x", "thread_id": "telegram-1"},
                                     {"content": "x"},
                                     ["not", "an", "object"]])
def test_reply_for_an_unknown_thread_is_404(tmp_path, payload):
    tasks = _tasks(tmp_path)
    tasks.store.insert("x", None)  # task 1 exists; other threads still miss
    assert asyncio.run(tasks.on_reply(payload)) == (404, {"error": "unknown task"})


def test_list_tasks_newest_first_with_status(tmp_path):
    tasks = _tasks(tmp_path)
    assert asyncio.run(_call(tasks, "list_tasks", {})) == "Задач ассистенту ещё не было."
    store = tasks.store
    first = store.insert("первая", "kitchen")
    store.mark_delivered(first, store.add_answer(first, "готово"))
    second = store.insert("вторая", None)
    store.add_answer(second, "не услышан")
    store.insert("третья", "bedroom")

    lines = asyncio.run(_call(tasks, "list_tasks", {"limit": 5})).splitlines()

    when = datetime.fromtimestamp(store.get(3)["created_ts"]).strftime("%Y-%m-%d %H:%M")
    assert lines[0] == f"№3 {when} [bedroom]: третья — в работе"
    assert lines[1].startswith("№2 ") and lines[1].endswith(": вторая — ответ: не услышан (не прозвучал)")
    assert lines[2].startswith("№1 ") and lines[2].endswith("[kitchen]: первая — ответ: готово")
    assert asyncio.run(_call(tasks, "list_tasks", {"limit": 1})).splitlines() == lines[:1]
