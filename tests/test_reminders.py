"""Tests for the voice-reminders subsystem.

Layers covered:
  * RemindersStore (sync SQLite): insert/list ordering, next_due_ts, pop_due (a
    repeating alarm moves on, the rest is deleted), delete, NULL device.
  * next_occurrence and the spoken phrases of reminders, timers and alarms.
  * ReminderScheduler (async): fires the row after due, takes overdue rows on start
    without firing, cancel.
  * Built-in reminders MCP server driven through BuiltinMcpSource.call_tool:
    set_reminder reads current_device, the timer/alarm/command tools, list/cancel text
    output.
  * LlmStage device wiring: LlmRequest.device travels via hub.call's ContextVar
    publication to an in-process tool and resets after the run.
  * DeviceManager.fire routing: each kind, named-offline, None.

Async code is driven with asyncio.run(...) (config-independent), matching the
existing test style.
"""

import asyncio
import time
from datetime import datetime

from src.builtin_mcp.reminders import _format_due, build_reminders_server
from src.device_server import ALARM_RING_S, TIMER_RING_S
from src.llm import LlmRequest, LlmStage
from src.plugins.llm.base import LlmConfig
from src.reminders import (
    ReminderScheduler,
    RemindersStore,
    _format_ago,
    format_alarm_speech,
    format_duration,
    format_reminder_speech,
    format_timer_speech,
    next_occurrence,
)
from src.run_context import current_device, in_command
from src.tool_hub import BuiltinMcpSource, ToolHub


def _store(tmp_path):
    return RemindersStore(str(tmp_path / "schedule.db"))


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


def test_pop_due_moves_a_repeating_alarm_on_and_deletes_the_rest(tmp_path):
    store = _store(tmp_path)
    now = datetime(2026, 10, 22, 7, 0, 30).timestamp()  # a Thursday
    alarm = store.insert(None, datetime(2026, 10, 22, 7, 0).timestamp(), "d",
                         kind="alarm", days=[1, 4])
    store.insert("пицца", now - 1, "d", kind="timer")
    later = store.insert("выключи свет", now + 100, "d", kind="command")

    popped = store.pop_due(now)
    assert [(r["kind"], r["text"], r["days"]) for r in popped] == [
        ("alarm", None, [1, 4]), ("timer", "пицца", None)]
    # The alarm keeps its id and moves to Monday 07:00; the timer is gone.
    assert [(r["id"], r["kind"], r["due_ts"]) for r in store.list_pending()] == [
        (later, "command", now + 100),
        (alarm, "alarm", datetime(2026, 10, 26, 7, 0).timestamp()),
    ]
    store.close()


def test_next_occurrence_is_the_first_matching_local_time_after():
    def at(*a):
        return datetime(*a).timestamp()

    thu_8 = at(2026, 10, 22, 8, 0)  # a Thursday
    weekdays = [1, 2, 3, 4, 5]
    assert next_occurrence(9, 0, None, thu_8) == at(2026, 10, 22, 9, 0)  # still ahead today
    assert next_occurrence(8, 0, None, thu_8) == at(2026, 10, 23, 8, 0)  # not now: tomorrow
    assert next_occurrence(7, 0, weekdays, thu_8) == at(2026, 10, 23, 7, 0)
    assert next_occurrence(7, 0, weekdays, at(2026, 10, 23, 8, 0)) == at(2026, 10, 26, 7, 0)
    assert next_occurrence(9, 0, [4], thu_8) == at(2026, 10, 22, 9, 0)
    assert next_occurrence(7, 0, [4], thu_8) == at(2026, 10, 29, 7, 0)  # a week on


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

    async def fake_deliver(row):
        captured.append(row)
        fired.set()

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = fake_deliver
        await sched.start()
        sched.add("молоко", time.time() + 0.05, "kitchen")
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        await sched.stop()

    asyncio.run(main())
    [row] = captured
    assert (row["kind"], row["text"], row["device"]) == ("reminder", "молоко", "kitchen")
    assert store.list_pending() == []
    store.close()


def test_scheduler_start_drops_overdue_rows_and_moves_a_repeating_alarm_on(tmp_path):
    store = _store(tmp_path)
    # Pre-insert a reminder and an everyday alarm already past due.
    store.insert("missed", time.time() - 60, "kitchen")
    due = time.time() - 3600
    alarm = store.insert(None, due, "kitchen", kind="alarm", days=[1, 2, 3, 4, 5, 6, 7])
    captured = []

    async def fake_deliver(row):
        captured.append(row)

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = fake_deliver
        await sched.start()
        # Give the loop a couple of ticks; the overdue row must never be delivered.
        await asyncio.sleep(0.1)
        await sched.stop()

    asyncio.run(main())
    assert captured == []
    [left] = store.list_pending()
    assert left["id"] == alarm and 0 < left["due_ts"] - time.time() <= 86400
    assert (datetime.fromtimestamp(left["due_ts"]).strftime("%H:%M")
            == datetime.fromtimestamp(due).strftime("%H:%M"))
    store.close()


def test_scheduler_cancel_prevents_fire(tmp_path):
    store = _store(tmp_path)
    captured = []

    async def fake_deliver(row):
        captured.append(row)

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


async def _wait_until(predicate, timeout=2.0, interval=0.005):
    """Bounded poll: await until predicate() is truthy or timeout elapses.

    Event-driven where possible; for branches with no deliver callback to signal on
    (deliver=None drops silently), we poll observable store state on short ticks with a
    hard upper bound — never a single fixed sleep that assumes timing.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def test_scheduler_deliver_none_drops_row_without_raising(tmp_path):
    # reminders.py ~236-240: with no deliver callback, a due reminder is popped from
    # the store (one-shot loss) and the loop logs+continues without raising. We drive
    # it with a bounded poll of the store, never a fixed sleep.
    store = _store(tmp_path)

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = None  # explicit: no delivery callback wired
        await sched.start()
        rid = sched.add("молоко", time.time() + 0.05, "kitchen")
        # Wait (bounded) until the loop pops the due row; one-shot => row gone.
        # Observe NON-destructively via list_pending so the probe never deletes the row.
        def ids():
            return {r["id"] for r in store.list_pending()}
        assert rid in ids()  # present before its due time
        popped = await _wait_until(lambda: rid not in ids())
        await sched.stop()
        return popped

    popped = asyncio.run(main())
    # The row was removed by the loop (no deliver callback => silent one-shot loss).
    assert popped is True
    assert store.list_pending() == []
    store.close()


def test_scheduler_survives_deliver_exception_and_delivers_next(tmp_path):
    # reminders.py ~243-250: a deliver that raises on its FIRST call must not kill the
    # loop. The first reminder is lost (one-shot, already popped), but a later reminder
    # is still delivered. Synchronization is an asyncio.Event set by the SECOND
    # successful deliver, bounded by wait_for(timeout=2.0) — no fixed sleeps.
    store = _store(tmp_path)
    calls = []
    second_delivered = asyncio.Event()

    async def flaky_deliver(row):
        calls.append(row)
        if len(calls) == 1:
            raise RuntimeError("boom on first delivery")
        second_delivered.set()

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = flaky_deliver
        await sched.start()
        sched.add("первое", time.time() + 0.05, "kitchen")
        sched.add("второе", time.time() + 0.12, "bedroom")
        await asyncio.wait_for(second_delivered.wait(), timeout=2.0)
        await sched.stop()

    asyncio.run(main())
    # Both reminders reached deliver: the first raised, the loop survived, the second
    # was delivered afterwards.
    assert [(r["text"], r["device"]) for r in calls] == [("первое", "kitchen"), ("второе", "bedroom")]
    # Both one-shot rows are gone from the store.
    assert store.list_pending() == []
    store.close()


def test_a_ringing_delivery_does_not_hold_back_the_next_row(tmp_path):
    # An alarm rings for minutes: the timer due meanwhile fires on time, and stop()
    # cancels the delivery still ringing.
    store = _store(tmp_path)
    timer_fired = asyncio.Event()
    ringing = asyncio.Event()

    async def deliver(row):
        if row["kind"] == "alarm":
            ringing.set()
            await asyncio.sleep(60)  # rings until stopped
        else:
            timer_fired.set()

    async def main():
        sched = ReminderScheduler(store)
        sched.deliver = deliver
        await sched.start()
        sched.add("", time.time() + 0.05, "bedroom", kind="alarm")
        sched.add("", time.time() + 0.1, "kitchen", kind="timer")
        await asyncio.wait_for(timer_fired.wait(), timeout=2.0)
        assert ringing.is_set()
        await asyncio.wait_for(sched.stop(), timeout=2.0)

    asyncio.run(main())
    store.close()


# --- Built-in reminders MCP server -------------------------------------------


class FakeScheduler:
    """Records scheduler calls so the tool layer can be tested in isolation."""

    def __init__(self, pending=None):
        self.added = []  # list of dict(text, due_ts, device, kind, days)
        self.cancelled = []  # list of reminder_id
        self._pending = pending or []
        self._next_id = 1
        self.cancel_result = True

    def add(self, text, due_ts, device, kind="reminder", days=None):
        rid = self._next_id
        self._next_id += 1
        self.added.append({"text": text, "due_ts": due_ts, "device": device, "kind": kind,
                           "days": days})
        return rid

    def cancel(self, reminder_id):
        self.cancelled.append(reminder_id)
        return self.cancel_result

    def pending(self):
        return self._pending


def _source(scheduler):
    return BuiltinMcpSource("reminders", build_reminders_server(scheduler))


def _call(scheduler, name, args):
    async def main():
        source = _source(scheduler)
        await source.start()
        return await source.call(name, args)

    return asyncio.run(main())


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


def test_list_reminders_shows_each_kind():
    now = time.time()
    pending = [
        {"id": 1, "kind": "reminder", "text": "молоко", "due_ts": now + 600, "days": None,
         "device": "kitchen", "created_ts": now},
        {"id": 2, "kind": "timer", "text": None, "due_ts": now + 90, "days": None,
         "device": "kitchen", "created_ts": now},
        {"id": 3, "kind": "alarm", "text": "на работу", "due_ts": now + 3600,
         "days": [1, 2, 3, 4, 5], "device": None, "created_ts": now},
        {"id": 4, "kind": "command", "text": "выключи свет", "due_ts": now + 2400, "days": None,
         "device": "kitchen", "created_ts": now},
    ]
    lines = _call(FakeScheduler(pending=pending), "list_reminders", {}).split("\n")
    assert lines[0].startswith("№1 напоминание: молоко — ")
    assert lines[0].endswith("(через 10 мин) [kitchen]")
    assert lines[1].startswith("№2 таймер — ")
    assert lines[1].endswith(", через 1 минуту 30 секунд [kitchen]")
    assert lines[2].startswith("№3 будильник: на работу — ")
    assert lines[2].endswith(", повтор: пн, вт, ср, чт, пт")
    assert lines[3].startswith("№4 отложенная команда: выключи свет — ")


def test_set_timer_stores_a_timer_and_says_for_how_long():
    sched = FakeScheduler()
    assert _call(sched, "set_timer", {"seconds": 300, "label": "пицца"}) == (
        "Таймер поставлен (№1) на 5 минут.")
    [added] = sched.added
    assert (added["kind"], added["text"], added["days"]) == ("timer", "пицца", None)
    assert abs(added["due_ts"] - (time.time() + 300)) < 5
    assert "положительное" in _call(sched, "set_timer", {"seconds": 0})
    assert len(sched.added) == 1


def test_set_alarm_repeats_on_its_days_or_rings_once():
    sched = FakeScheduler()
    assert _call(sched, "set_alarm", {"time": "07:30", "days": [5, 1, 1]}).endswith(
        ", повтор: пн, пт.")
    # Neither days nor date: once, today if 07:30 is still ahead, else tomorrow.
    _call(sched, "set_alarm", {"time": "07:30", "label": "на работу"})
    _call(sched, "set_alarm", {"time": "07:30", "date": "2099-01-02"})
    weekly, once, dated = sched.added
    assert (weekly["kind"], weekly["days"], weekly["text"]) == ("alarm", [1, 5], None)
    assert weekly["due_ts"] == next_occurrence(7, 30, [1, 5], time.time())
    assert (once["days"], once["text"]) == (None, "на работу")
    assert once["due_ts"] == next_occurrence(7, 30, None, time.time())
    assert (dated["days"], dated["due_ts"]) == (None, datetime(2099, 1, 2, 7, 30).timestamp())


def test_set_alarm_refuses_what_it_cannot_read():
    sched = FakeScheduler()
    for args, words in (
        ({"time": "7:3x"}, "ЧЧ:ММ"),
        ({"time": "07:30", "days": [1], "date": "2099-01-02"}, "не оба"),
        ({"time": "07:30", "days": [0, 8]}, "от 1"),
        ({"time": "07:30", "date": "02.01.2099"}, "ГГГГ-ММ-ДД"),
        ({"time": "07:30", "date": "2000-01-01"}, "уже прошло"),
    ):
        assert words in _call(sched, "set_alarm", args)
    assert sched.added == []


def test_schedule_command_defers_the_phrase_but_not_from_a_fired_command():
    sched = FakeScheduler()
    out = _call(sched, "schedule_command", {"command": "выключи свет в туалете", "in_minutes": 40})
    assert out.startswith("Команда отложена (№1)")
    [added] = sched.added
    assert (added["kind"], added["text"]) == ("command", "выключи свет в туалете")
    assert abs(added["due_ts"] - (time.time() + 40 * 60)) < 5
    assert "уже прошло" in _call(sched, "schedule_command",
                                 {"command": "x", "at": "2000-01-01T10:00"})
    token = in_command.set(True)
    try:
        out = _call(sched, "schedule_command", {"command": "выключи свет", "in_minutes": 40})
    finally:
        in_command.reset(token)
    assert "выполни её сейчас" in out
    assert len(sched.added) == 1


def test_list_reminders_empty():
    sched = FakeScheduler(pending=[])

    async def main():
        source = _source(sched)
        await source.start()
        return await source.call("list_reminders", {})

    out = asyncio.run(main())
    assert out == "Нет ни напоминаний, ни таймеров, ни будильников, ни отложенных команд."


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


# --- LlmStage device wiring ---------------------------------------------------


class _DeviceRecordingSource:
    """In-process ToolSource double whose call() records the ambient current_device.

    Driven through a REAL ToolHub so the test proves the whole chain:
    LlmRequest.device -> hub.call(..., device=...) -> ContextVar -> tool.
    """

    id = "rem"

    def __init__(self):
        self._tools = [{
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "set a reminder",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        self.seen_device = "UNSET"

    async def start(self):
        return None

    async def ensure(self):
        return None

    def raw_tools(self):
        return self._tools

    async def call(self, name, args):
        self.seen_device = current_device.get()
        return "ok"


class _ScriptedBackend:
    """Returns one tool call then a final answer."""

    def __init__(self):
        self._i = 0

    async def complete(self, messages, tools, *, has_next=False):
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


def test_llm_stage_publishes_device_via_hub_and_resets():
    source = _DeviceRecordingSource()
    backend = _ScriptedBackend()

    async def main():
        hub = ToolHub([source])
        await hub.start()
        stage = LlmStage(backend, hub, LlmConfig(max_tool_rounds=5))
        return await stage.respond(LlmRequest(
            system_prompt="PROMPT", history=[], user_text="напомни", device="kitchen",
        ))

    asyncio.run(main())
    # The tool saw the device via the ContextVar the hub published.
    assert source.seen_device == "kitchen"
    # After the call the ContextVar is back to its default.
    assert current_device.get() is None


# --- DeviceManager.fire routing ----------------------------------------------


class _FakeClient:
    def __init__(self, name, online):
        self.name = name
        self.online = online
        self.calls = []

    async def announce(self, text, reason):
        self.calls.append(("announce", text, reason))

    async def ring(self, text, reason, seconds):
        self.calls.append(("ring", text, reason, seconds))

    async def run_command(self, text):
        self.calls.append(("run_command", text))


def _row(kind="reminder", text="молоко", device="kitchen", created_ts=None):
    return {"id": 1, "kind": kind, "text": text, "due_ts": 1300.0, "days": None,
            "device": device, "created_ts": created_ts}


def _manager_with(clients):
    """A DeviceManager whose device list is `clients` (no runtime, no server)."""
    from src.device_server import DeviceManager

    class _Manager(DeviceManager):
        def __init__(self):
            pass

        @property
        def clients(self):
            return clients

    return _Manager()


def test_fire_hands_each_kind_to_its_device():
    kitchen = _FakeClient("kitchen", online=True)
    bedroom = _FakeClient("bedroom", online=True)
    mgr = _manager_with([kitchen, bedroom])

    async def main():
        await mgr.fire(_row(created_ts=time.time() - 60))
        await mgr.fire(_row("timer", "пицца", created_ts=1000.0))
        await mgr.fire(_row("alarm", None))
        await mgr.fire(_row("command", "включи свет"))

    asyncio.run(main())
    assert kitchen.calls == [
        ("announce", "минуту назад вы просили напомнить вам молоко", "reminder"),
        ("ring", "Таймер на 5 минут, пицца", "timer", TIMER_RING_S),
        ("ring", "Будильник", "alarm", ALARM_RING_S),
        ("run_command", "включи свет"),
    ]
    assert bedroom.calls == []


def test_fire_drops_an_offline_device_unless_the_row_is_a_command():
    kitchen = _FakeClient("kitchen", online=False)
    mgr = _manager_with([kitchen])

    async def main():
        for kind in ("reminder", "timer", "alarm", "command"):
            await mgr.fire(_row(kind, "x"))

    asyncio.run(main())
    assert kitchen.calls == [("run_command", "x")]


def test_fire_without_a_device_picks_the_first_online():
    offline = _FakeClient("a", online=False)
    online = _FakeClient("b", online=True)
    mgr = _manager_with([offline, online])

    asyncio.run(mgr.fire(_row(text="привет", device=None)))
    assert online.calls == [("announce", "привет", "reminder")]
    assert offline.calls == []


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
    out = format_reminder_speech("Вытащить стирку", created_ts=now - 60, now=now)
    assert out == "минуту назад вы просили напомнить вам вытащить стирку"
    # The reminder body is lower-cased on its first letter.
    assert "вытащить" in out


def test_format_reminder_speech_falls_back_when_created_ts_none():
    now = time.time()
    assert format_reminder_speech("что-то", created_ts=None, now=now) == "что-то"


def test_format_duration_and_the_timer_and_alarm_phrases():
    assert [format_duration(s) for s in (60, 300, 3600, 5400, 90, 7322)] == [
        "минуту", "5 минут", "час", "1 час 30 минут", "1 минуту 30 секунд",
        "2 часа 2 минуты 2 секунды"]
    assert format_timer_speech("пицца", 1300.4, 1000.0) == "Таймер на 5 минут, пицца"
    assert format_timer_speech(None, 1021.0, 1000.0) == "Таймер на 21 секунду"
    assert format_alarm_speech(None) == "Будильник"
    assert format_alarm_speech("на работу") == "Будильник, на работу"


# --- _format_due past-due / due-now ------------------------------------------


def test_format_due_past_and_now_omits_suffix():
    # When the due time is in the past or exactly now, minutes_left <= 0 so the
    # "(через N мин)" hint must be omitted: the output is the bare local timestamp.
    now = time.time()
    past = now - 120
    bare_past = datetime.fromtimestamp(past).strftime("%Y-%m-%d %H:%M")
    out_past = _format_due(past, now)
    assert out_past == bare_past
    assert "через" not in out_past

    bare_now = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M")
    out_now = _format_due(now, now)
    assert out_now == bare_now
    assert "через" not in out_now
