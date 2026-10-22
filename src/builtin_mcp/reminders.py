"""Built-in reminders MCP server (in-process FastMCP).

Exposes reminders, timers, alarms and deferred commands as tools the model calls when
the user says e.g. "напомни через полчаса купить молоко", "поставь таймер на 5 минут",
"разбуди по будням в семь", "включи свет через 40 минут". The tools are thin: they
resolve a due timestamp, read the originating speaker from the per-run ContextVar
(never an LLM-visible argument), and hand off to the ReminderScheduler. Every tool
catches its exceptions and returns a human-readable Russian string (it never raises),
mirroring the calendar/OpenWeatherMap tool contract.
"""

import time
from datetime import date as ddate
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from src.reminders import format_duration, next_occurrence
from src.run_context import current_device, in_command

# How list_reminders names each kind of row.
_KINDS = {"reminder": "напоминание", "timer": "таймер", "alarm": "будильник",
          "command": "отложенная команда"}
# ISO weekdays 1..7 as list_reminders and set_alarm show them.
_WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def _format_due(due_ts: float, now: float) -> str:
    """Render a due timestamp as Russian local time plus a "через N мин" hint."""
    when = datetime.fromtimestamp(due_ts).strftime("%Y-%m-%d %H:%M")
    minutes_left = int(round((due_ts - now) / 60))
    if minutes_left > 0:
        return f"{when} (через {minutes_left} мин)"
    return when


def _format_days(days: list[int]) -> str:
    return ", ".join(_WEEKDAYS[d - 1] for d in days)


def _resolve_due(in_minutes, at) -> tuple[float | None, str | None]:
    """The due timestamp of `in_minutes` from now or of `at` (ISO), or None and the
    message that says why there is none (set_reminder, schedule_command)."""
    if in_minutes is not None:
        minutes = int(in_minutes)
        if minutes <= 0:
            return None, "Укажи положительное число минут в in_minutes."
        due_ts = time.time() + minutes * 60
    elif at is not None:
        try:
            due_ts = datetime.fromisoformat(at).timestamp()
        except ValueError:
            return None, "Не понял время: используй ISO формат, например 2026-06-08T15:30."
    else:
        return None, "Укажи время: in_minutes или at."
    # Backstop for every resolution path (mainly a past `at`): never
    # schedule anything whose due time is not in the future.
    if due_ts <= time.time():
        return None, "Это время уже прошло — укажи время в будущем."
    return due_ts, None


def build_reminders_server(scheduler) -> FastMCP:
    """Build a FastMCP server exposing the reminder, timer, alarm and deferred command
    tools backed by `scheduler`."""
    mcp = FastMCP("reminders")

    @mcp.tool(
        name="set_reminder",
        description=(
            "Поставить напоминание, которое будет произнесено вслух на этой же "
            "колонке в нужное время. text — что напомнить: пиши в инфинитиве, со "
            "строчной буквы, без обращений и вводных слов (например «купить молоко», "
            "«позвонить маме»); не начинай с «напомни» или «напоминание» — это "
            "подразумевается само собой. Укажи ровно один из "
            "вариантов времени: in_minutes — через сколько минут напомнить "
            "(например, для «напомни через полчаса» используй in_minutes=30); "
            "at — конкретное дата-время в формате ISO (например 2026-06-08T15:30)."
        ),
    )
    async def set_reminder(
        text: str, in_minutes: int | None = None, at: str | None = None
    ) -> str:
        try:
            due_ts, error = _resolve_due(in_minutes, at)
            if error is not None:
                return error
            device = current_device.get()
            rid = scheduler.add(text=text, due_ts=due_ts, device=device)
            return f"Напоминание поставлено (№{rid})."
        except Exception as e:
            return f"Не удалось поставить напоминание: {e}"

    @mcp.tool(
        name="set_timer",
        description=(
            "Поставить таймер на этой же колонке: когда время выйдет, колонка скажет "
            "«Сработал таймер на …» и будет звенеть, пока его не остановят голосом или кнопкой "
            "(не дольше минуты). seconds — длительность в секундах (например, для "
            "«таймер на 5 минут» используй seconds=300). label — необязательная подпись, "
            "для чего таймер (например «пицца»), без слова «таймер»."
        ),
    )
    async def set_timer(seconds: int, label: str | None = None) -> str:
        try:
            seconds = int(seconds)
            if seconds <= 0:
                return "Укажи положительное число секунд в seconds."
            rid = scheduler.add(text=label, due_ts=time.time() + seconds,
                                device=current_device.get(), kind="timer")
            return f"Таймер поставлен (№{rid}) на {format_duration(seconds)}."
        except Exception as e:
            return f"Не удалось поставить таймер: {e}"

    @mcp.tool(
        name="set_alarm",
        description=(
            "Поставить будильник на этой же колонке: в нужное время колонка скажет "
            "«Будильник» и будет звенеть, пока его не остановят голосом или кнопкой (не "
            "дольше пяти минут). time — местное время в формате ЧЧ:ММ (например 07:30). "
            "days — дни недели, по которым будильник повторяется, числами от 1 "
            "(понедельник) до 7 (воскресенье): «по будням» — [1,2,3,4,5]. date — дата "
            "разового будильника в формате ГГГГ-ММ-ДД. Без days и date будильник звонит "
            "один раз: сегодня, если это время ещё не прошло, иначе завтра. Не указывай "
            "days и date вместе. label — необязательная подпись (например «на работу»)."
        ),
    )
    async def set_alarm(time: str, days: list[int] | None = None,
                        date: str | None = None, label: str | None = None) -> str:
        try:
            try:
                at = datetime.strptime(time, "%H:%M")
            except ValueError:
                return "Не понял время: укажи его как ЧЧ:ММ, например 07:30."
            if days and date is not None:
                return "Укажи либо days (повтор по дням), либо date (разовый на дату), не оба."
            if days and not all(1 <= d <= 7 for d in days):
                return "Дни повтора — числа от 1 (понедельник) до 7 (воскресенье)."
            days = sorted(set(days)) if days else None
            now = datetime.now().timestamp()
            if date is not None:
                try:
                    due_ts = datetime.combine(ddate.fromisoformat(date), at.time()).timestamp()
                except ValueError:
                    return "Не понял дату: используй формат ГГГГ-ММ-ДД, например 2026-06-08."
                if due_ts <= now:
                    return "Это время уже прошло — укажи время в будущем."
            else:
                due_ts = next_occurrence(at.hour, at.minute, days, now)
            rid = scheduler.add(text=label, due_ts=due_ts, device=current_device.get(),
                                kind="alarm", days=days)
            reply = (f"Будильник поставлен (№{rid}) на "
                     f"{datetime.fromtimestamp(due_ts).strftime('%Y-%m-%d %H:%M')}")
            if days:
                reply += f", повтор: {_format_days(days)}"
            return reply + "."
        except Exception as e:
            return f"Не удалось поставить будильник: {e}"

    @mcp.tool(
        name="schedule_command",
        description=(
            "Отложить команду: в нужное время она будет выполнена так, будто пользователь "
            "сказал её этой колонке. command — самостоятельная фраза, как пользователь "
            "сказал бы её ассистенту, без времени и без местоимений («включи свет на кухне "
            "и выключи в туалете», а не «включи его»). Если часть просьбы нужно сделать "
            "сейчас, а часть позже («включи на кухне, а через 40 минут выключи в "
            "туалете»), первую часть выполни сейчас — её инструментом, а если это вопрос "
            "(«скажи, сколько времени, а через час…»), ответь на него в своей реплике, — и "
            "отложи только вторую. Укажи ровно "
            "один из вариантов времени: in_minutes — через сколько минут выполнить "
            "(например, для «через 40 минут» используй in_minutes=40); at — конкретное "
            "дата-время в формате ISO (например 2026-06-08T15:30)."
        ),
    )
    async def schedule_command(
        command: str, in_minutes: int | None = None, at: str | None = None
    ) -> str:
        if in_command.get():
            return ("Это уже выполнение отложенной команды: не откладывай её снова, а "
                    "выполни её сейчас.")
        try:
            due_ts, error = _resolve_due(in_minutes, at)
            if error is not None:
                return error
            rid = scheduler.add(text=command, due_ts=due_ts, device=current_device.get(),
                                kind="command")
            return f"Команда отложена (№{rid}) до {_format_due(due_ts, time.time())}."
        except Exception as e:
            return f"Не удалось отложить команду: {e}"

    @mcp.tool(
        name="list_reminders",
        description=(
            "Список активных напоминаний, таймеров, будильников и отложенных команд: "
            "номер (id), вид, текст, время; для таймера — через сколько он сработает, для "
            "будильника — дни повтора; колонка."
        ),
    )
    async def list_reminders() -> str:
        try:
            pending = scheduler.pending()
        except Exception as e:
            return f"Не удалось получить список: {e}"
        if not pending:
            return "Нет ни напоминаний, ни таймеров, ни будильников, ни отложенных команд."
        now = time.time()
        lines = []
        for r in pending:
            line = f"№{r['id']} {_KINDS[r['kind']]}"
            if r["text"]:
                line += f": {r['text']}"
            if r["kind"] == "timer":
                when = datetime.fromtimestamp(r["due_ts"]).strftime("%Y-%m-%d %H:%M:%S")
                line += f" — {when}, через {format_duration(round(r['due_ts'] - now))}"
            else:
                line += f" — {_format_due(r['due_ts'], now)}"
            if r["days"]:
                line += f", повтор: {_format_days(r['days'])}"
            if r.get("device"):
                line += f" [{r['device']}]"
            lines.append(line)
        return "\n".join(lines)

    @mcp.tool(
        name="cancel_reminder",
        description=(
            "Отменить напоминание, таймер, будильник или отложенную команду по номеру "
            "(id) из list_reminders."
        ),
    )
    async def cancel_reminder(reminder_id: int) -> str:
        try:
            ok = scheduler.cancel(reminder_id)
        except Exception as e:
            return f"Не удалось отменить: {e}"
        if ok:
            return f"№{reminder_id} отменено."
        return f"№{reminder_id} не найдено."

    return mcp
