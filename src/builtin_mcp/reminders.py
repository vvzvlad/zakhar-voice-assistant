"""Built-in reminders MCP server (in-process FastMCP).

Exposes one-shot voice reminders as tools the model calls when the user says e.g.
"напомни через полчаса купить молоко". The tools are thin: they resolve a due
timestamp, read the originating speaker from the per-run ContextVar (never an
LLM-visible argument), and hand off to the ReminderScheduler. Every tool catches
its exceptions and returns a human-readable Russian string (it never raises),
mirroring the calendar/OpenWeatherMap tool contract.
"""

import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from src.run_context import current_device


def _format_due(due_ts: float, now: float) -> str:
    """Render a due timestamp as Russian local time plus a "через N мин" hint."""
    when = datetime.fromtimestamp(due_ts).strftime("%Y-%m-%d %H:%M")
    minutes_left = int(round((due_ts - now) / 60))
    if minutes_left > 0:
        return f"{when} (через {minutes_left} мин)"
    return when


def build_reminders_server(scheduler) -> FastMCP:
    """Build a FastMCP server exposing the reminder tools backed by `scheduler`."""
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
            if in_minutes is not None:
                minutes = int(in_minutes)
                if minutes <= 0:
                    return "Укажи положительное число минут в in_minutes."
                due_ts = time.time() + minutes * 60
            elif at is not None:
                try:
                    due_ts = datetime.fromisoformat(at).timestamp()
                except ValueError:
                    return (
                        "Не понял время: используй ISO формат, например "
                        "2026-06-08T15:30."
                    )
            else:
                return "Укажи время напоминания: in_minutes или at."

            # Backstop for every resolution path (mainly a past `at`): never
            # schedule a reminder whose due time is not in the future.
            if due_ts <= time.time():
                return "Это время уже прошло — укажи время в будущем."

            device = current_device.get()
            rid = scheduler.add(text=text, due_ts=due_ts, device=device)
            return f"Напоминание поставлено (№{rid})."
        except Exception as e:
            return f"Не удалось поставить напоминание: {e}"

    @mcp.tool(
        name="list_reminders",
        description="Список активных напоминаний (id, текст, время, колонка).",
    )
    async def list_reminders() -> str:
        try:
            pending = scheduler.pending()
        except Exception as e:
            return f"Не удалось получить напоминания: {e}"
        if not pending:
            return "Активных напоминаний нет."
        now = time.time()
        lines = []
        for r in pending:
            line = f"№{r['id']}: {r['text']} — {_format_due(r['due_ts'], now)}"
            if r.get("device"):
                line += f" [{r['device']}]"
            lines.append(line)
        return "\n".join(lines)

    @mcp.tool(
        name="cancel_reminder",
        description="Отменить напоминание по его номеру (id).",
    )
    async def cancel_reminder(reminder_id: int) -> str:
        try:
            ok = scheduler.cancel(reminder_id)
        except Exception as e:
            return f"Не удалось отменить напоминание: {e}"
        if ok:
            return f"Напоминание №{reminder_id} отменено."
        return f"Напоминание №{reminder_id} не найдено."

    return mcp
