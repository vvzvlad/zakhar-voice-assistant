"""Tests for the voice-reminders subsystem.

Layers covered:
  * RemindersStore (sync SQLite): insert/list ordering, next_due_ts, pop_due,
    drop_overdue, delete, NULL device.
  * ReminderScheduler (async): fires after due, drops overdue on start, cancel.
  * Built-in reminders MCP server driven through BuiltinMcpSource.call_tool:
    set_reminder reads current_device, list/cancel text output.
  * call_llm_api contextvar wiring: device reaches an in-process tool and resets.
  * DeviceManager.announce routing: named-online / named-offline / None.

Async code is driven with asyncio.run(...) (config-independent), matching the
existing test style.
"""

import asyncio
import time
from datetime import datetime

from src.builtin_mcp.reminders import build_reminders_server
from src.core_config import CoreConfig, PromptConfig
from src.llm import call_llm_api
from src.plugins.llm.base import LlmConfig
from src.reminders import (
    ReminderScheduler,
    RemindersStore,
    _format_ago,
    _format_reminder_speech,
)
from src.run_context import current_device
from src.tool_hub import BuiltinMcpSource


def _store(tmp_path):
    return RemindersStore(str(tmp_path / "reminders.db"))


# --- RemindersStore (sync) ---------------------------------------------------


def test_insert_and_list_ordered_by_due_ts(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    store.insert("late", now + 100, "kitchen")
    store.insert("soon", now + 10, "bedroom")
    store.insert("mid", now + 50, None)

    pending = store.list_pending()
    assert [r["text"] for r in pending] == ["soon", "mid", "late"]
    # created_ts is stamped on insert.
    assert all(r["created_ts"] is not None for r in pending)
    store.close()


def test_next_due_ts(tmp_path):
    store = _store(tmp_path)
    assert store.next_due_ts() is None  # empty
    now = time.time()
    store.insert("a", now + 100, "d")
    store.insert("b", now + 10, "d")
    assert store.next_due_ts() == now + 10
    store.close()


def test_pop_due_returns_and_deletes_only_due(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    store.insert("past", now - 5, "d")
    store.insert("now", now, "d")
    store.insert("future", now + 100, "d")

    popped = store.pop_due(now)
    assert {r["text"] for r in popped} == {"past", "now"}
    # Only the future one survives.
    assert [r["text"] for r in store.list_pending()] == ["future"]
    store.close()


def test_drop_overdue_deletes_only_overdue_and_returns_count(tmp_path):
    store = _store(tmp_path)
    now = time.time()
    store.insert("old1", now - 10, "d")
    store.insert("old2", now - 1, "d")
    store.insert("fresh", now + 100, "d")

    count = store.drop_overdue(now)
    assert count == 2
    assert [r["text"] for r in store.list_pending()] == ["fresh"]
    store.close()


def test_delete_returns_true_false(tmp_path):
    store = _store(tmp_path)
    rid = store.insert("x", time.time() + 100, "d")
    assert store.delete(rid) is True
    assert store.delete(rid) is False  # already gone
    assert store.delete(99999) is False
    store.close()


def test_device_stored_and_returned_including_none(tmp_path):
    store = _store(tmp_path)
    store.insert("named", time.time() + 10, "kitchen")
    store.insert("anon", time.time() + 20, None)
    by_text = {r["text"]: r["device"] for r in store.list_pending()}
    assert by_text["named"] == "kitchen"
    assert by_text["anon"] is None
    store.close()


# --- ReminderScheduler (async) -----------------------------------------------


def test_scheduler_fires_deliver_after_due(tmp_path):
    store = _store(tmp_path)
    captured = []
    fired = asyncio.Event()

    async def fake_deliver(device, text):
        captured.append((device, text))
        fired.set()

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = fake_deliver
        await sched.start()
        sched.add("молоко", time.time() + 0.05, "kitchen")
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        await sched.stop()

    asyncio.run(main())
    assert len(captured) == 1
    device, text = captured[0]
    assert device == "kitchen"
    assert "молоко" in text
    assert "вы просили напомнить вам" in text
    store.close()


def test_scheduler_drops_overdue_on_start(tmp_path):
    store = _store(tmp_path)
    # Pre-insert a reminder already past due.
    store.insert("missed", time.time() - 60, "kitchen")
    captured = []

    async def fake_deliver(device, text):
        captured.append((device, text))

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = fake_deliver
        await sched.start()
        # Give the loop a couple of ticks; the overdue row must never be delivered.
        await asyncio.sleep(0.1)
        await sched.stop()

    asyncio.run(main())
    assert captured == []
    assert store.list_pending() == []
    store.close()


def test_scheduler_cancel_prevents_fire(tmp_path):
    store = _store(tmp_path)
    captured = []

    async def fake_deliver(device, text):
        captured.append((device, text))

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = fake_deliver
        await sched.start()
        rid = sched.add("отменяемое", time.time() + 0.2, "kitchen")
        assert sched.cancel(rid) is True
        # Wait past the original due time; nothing should fire.
        await asyncio.sleep(0.35)
        await sched.stop()

    asyncio.run(main())
    assert captured == []
    store.close()


# --- Built-in reminders MCP server -------------------------------------------


class FakeScheduler:
    """Records scheduler calls so the tool layer can be tested in isolation."""

    def __init__(self, pending=None):
        self.added = []  # list of dict(text, due_ts, device)
        self.cancelled = []  # list of reminder_id
        self._pending = pending or []
        self._next_id = 1
        self.cancel_result = True

    def add(self, text, due_ts, device):
        rid = self._next_id
        self._next_id += 1
        self.added.append({"text": text, "due_ts": due_ts, "device": device})
        return rid

    def cancel(self, reminder_id):
        self.cancelled.append(reminder_id)
        return self.cancel_result

    def pending(self):
        return self._pending


def _source(scheduler):
    return BuiltinMcpSource("reminders", build_reminders_server(scheduler))


def test_set_reminder_in_minutes_uses_current_device():
    sched = FakeScheduler()
    token = current_device.set("kitchen")
    try:

        async def main():
            source = _source(sched)
            await source.start()
            return await source.call(
                "set_reminder", {"text": "купить молоко", "in_minutes": 30}
            )

        out = asyncio.run(main())
    finally:
        current_device.reset(token)

    assert "№1" in out
    assert len(sched.added) == 1
    added = sched.added[0]
    assert added["text"] == "купить молоко"
    assert added["device"] == "kitchen"
    # due_ts is ~30 min in the future.
    assert abs(added["due_ts"] - (time.time() + 30 * 60)) < 5


def test_set_reminder_without_device_stores_none():
    sched = FakeScheduler()
    # ContextVar not set -> default None.

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("set_reminder", {"text": "x", "in_minutes": 5})

    out = asyncio.run(main())
    assert "№1" in out
    assert sched.added[0]["device"] is None


def test_set_reminder_at_iso_datetime():
    sched = FakeScheduler()
    # A clearly future ISO datetime so the future-time guard always passes.
    future_at = datetime.fromtimestamp(time.time() + 3600).strftime("%Y-%m-%dT%H:%M")

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call(
            "set_reminder", {"text": "встреча", "at": future_at}
        )

    out = asyncio.run(main())
    assert "№1" in out
    assert len(sched.added) == 1


def test_set_reminder_past_at_returns_error_string():
    sched = FakeScheduler()

    async def main():
        source = _source(sched)
        await source.start()
        # A clearly past ISO datetime must be rejected, not scheduled.
        return await source.call(
            "set_reminder", {"text": "встреча", "at": "2000-01-01T10:00"}
        )

    out = asyncio.run(main())
    assert "уже прошло" in out
    assert sched.added == []


def test_set_reminder_bad_at_returns_error_string():
    sched = FakeScheduler()

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("set_reminder", {"text": "x", "at": "not-a-date"})

    out = asyncio.run(main())
    assert "Не понял время" in out
    assert sched.added == []


def test_set_reminder_no_time_asks_for_it():
    sched = FakeScheduler()

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("set_reminder", {"text": "x"})

    out = asyncio.run(main())
    assert "in_minutes" in out or "at" in out
    assert sched.added == []


def test_set_reminder_non_positive_minutes():
    sched = FakeScheduler()

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("set_reminder", {"text": "x", "in_minutes": 0})

    out = asyncio.run(main())
    assert "положительное" in out
    assert sched.added == []


def test_list_reminders_text_output():
    now = time.time()
    pending = [
        {"id": 1, "text": "молоко", "due_ts": now + 600, "device": "kitchen",
         "created_ts": now},
        {"id": 2, "text": "позвонить", "due_ts": now + 1200, "device": None,
         "created_ts": now},
    ]
    sched = FakeScheduler(pending=pending)

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("list_reminders", {})

    out = asyncio.run(main())
    assert "№1" in out and "молоко" in out and "kitchen" in out
    assert "№2" in out and "позвонить" in out


def test_list_reminders_empty():
    sched = FakeScheduler(pending=[])

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("list_reminders", {})

    out = asyncio.run(main())
    assert out == "Активных напоминаний нет."


def test_cancel_reminder_found_and_not_found():
    sched = FakeScheduler()

    async def main():
        source = _source(sched)
        await source.start()
        ok = await source.call("cancel_reminder", {"reminder_id": 7})
        sched.cancel_result = False
        missing = await source.call("cancel_reminder", {"reminder_id": 8})
        return ok, missing

    ok, missing = asyncio.run(main())
    assert "№7" in ok and "отменено" in ok
    assert "№8" in missing and "не найдено" in missing
    assert sched.cancelled == [7, 8]


# --- call_llm_api contextvar wiring ------------------------------------------


class _DeviceRecordingHub:
    """Tool hub double whose call() records the current_device at call time."""

    def __init__(self):
        self.tools = [{
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "set a reminder",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        self.seen_device = "UNSET"

    async def ensure_tools(self):
        return None

    async def call(self, name, args):
        self.seen_device = current_device.get()
        return "ok"


class _ScriptedBackend:
    """Returns one tool call then a final answer."""

    def __init__(self):
        self._i = 0

    async def complete(self, messages, tools):
        self._i += 1
        if self._i == 1:
            return {
                "choices": [{"message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "set_reminder", "arguments": "{}"},
                    }],
                }}],
                "model": "x", "usage": {},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": "Готово."}}],
            "model": "x", "usage": {},
        }


def test_call_llm_api_publishes_device_and_resets(tmp_path):
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("PROMPT", encoding="utf-8")
    core = CoreConfig(prompt=PromptConfig(system_prompt_path=str(prompt_path)))
    hub = _DeviceRecordingHub()
    backend = _ScriptedBackend()

    async def main():
        return await call_llm_api(
            backend, hub, "напомни",
            core=core, llm_cfg=LlmConfig(max_tool_rounds=5),
            device="kitchen",
        )

    asyncio.run(main())
    # The tool saw the device via the ContextVar.
    assert hub.seen_device == "kitchen"
    # After the call the ContextVar is back to its default.
    assert current_device.get() is None


# --- DeviceManager.announce routing ------------------------------------------


class _FakeClientCfg:
    def __init__(self, name):
        self.name = name


class _FakeClient:
    def __init__(self, name, online):
        self.cfg = _FakeClientCfg(name)
        self.online = online
        self.announced = []

    async def announce(self, text):
        self.announced.append(text)


def _manager_with(clients):
    """Build a DeviceManager without running its __init__ (avoids real APIClient)."""
    from src.esphome_client import DeviceManager
    mgr = DeviceManager.__new__(DeviceManager)
    mgr.clients = clients
    return mgr


def test_announce_routes_to_named_online_client():
    kitchen = _FakeClient("kitchen", online=True)
    bedroom = _FakeClient("bedroom", online=True)
    mgr = _manager_with([kitchen, bedroom])

    asyncio.run(mgr.announce("kitchen", "молоко"))
    assert kitchen.announced == ["молоко"]
    assert bedroom.announced == []


def test_announce_drops_named_offline_client():
    kitchen = _FakeClient("kitchen", online=False)
    mgr = _manager_with([kitchen])

    asyncio.run(mgr.announce("kitchen", "молоко"))
    assert kitchen.announced == []  # offline -> dropped, no announce


def test_announce_none_picks_first_online():
    offline = _FakeClient("a", online=False)
    online = _FakeClient("b", online=True)
    mgr = _manager_with([offline, online])

    asyncio.run(mgr.announce(None, "привет"))
    assert online.announced == ["привет"]
    assert offline.announced == []


# --- Spoken-phrase helpers ---------------------------------------------------


def test_format_ago_granularity():
    assert _format_ago(30) == "минуту назад"
    assert _format_ago(60) == "минуту назад"
    assert _format_ago(120) == "2 минуты назад"
    assert _format_ago(300) == "5 минут назад"
    assert _format_ago(3600) == "час назад"
    assert _format_ago(7200) == "2 часа назад"
    assert _format_ago(86400) == "день назад"


def test_format_reminder_speech_full_phrase_and_lowercased():
    now = time.time()
    out = _format_reminder_speech("Вытащить стирку", created_ts=now - 60, now=now)
    assert out == "минуту назад вы просили напомнить вам вытащить стирку"
    # The reminder body is lower-cased on its first letter.
    assert "вытащить" in out


def test_format_reminder_speech_falls_back_when_created_ts_none():
    now = time.time()
    assert _format_reminder_speech("что-то", created_ts=None, now=now) == "что-то"
