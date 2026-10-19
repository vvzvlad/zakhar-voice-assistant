"""Built-in assistant-tasks MCP server (in-process FastMCP).

Lets the model hand a long job to the ZeroClaw assistant and list the jobs
handed off so far. The tools are thin: the originating speaker comes from the per-run
ContextVar (never an LLM-visible argument), the work is done by the AssistantTasks
service. Every tool catches its exceptions and returns a human-readable Russian string
(it never raises), mirroring the reminders/calendar tool contract.
"""

from datetime import datetime

from mcp.server.fastmcp import FastMCP

from src.run_context import current_device


def build_assistant_tasks_server(tasks) -> FastMCP:
    """Build a FastMCP server exposing the assistant-task tools backed by `tasks`."""
    mcp = FastMCP("assistant_tasks")

    @mcp.tool(
        name="delegate_task",
        description=(
            "Передать задачу ассистенту (агенту ZeroClaw): долгое исследование, поиск, "
            "вопрос, на который нужно время. Ответ придёт позже и прозвучит "
            "на этой же колонке. text — что нужно сделать или выяснить."
        ),
    )
    async def delegate_task(text: str) -> str:
        try:
            task_id = await tasks.delegate(text, current_device.get())
        except Exception as e:
            return f"Не удалось передать задачу ассистенту: {e}"
        return f"Задача №{task_id} передана ассистенту, ответ скажу, когда придёт."

    @mcp.tool(
        name="list_tasks",
        description=(
            "Последние задачи, переданные ассистенту: номер, время, колонка, текст "
            "и ответ или «в работе». limit — сколько последних задач показать."
        ),
    )
    async def list_tasks(limit: int = 5) -> str:
        try:
            rows = tasks.recent(limit)
        except Exception as e:
            return f"Не удалось получить задачи: {e}"
        if not rows:
            return "Задач ассистенту ещё не было."
        lines = []
        for r in rows:
            when = datetime.fromtimestamp(r["created_ts"]).strftime("%Y-%m-%d %H:%M")
            line = f"№{r['id']} {when}"
            if r["device"]:
                line += f" [{r['device']}]"
            line += f": {r['text']} — "
            if r["answer"] is None:
                line += "в работе"
            else:
                line += f"ответ: {r['answer']}"
                if not r["delivered"]:
                    line += " (не прозвучал)"
            lines.append(line)
        return "\n".join(lines)

    return mcp
