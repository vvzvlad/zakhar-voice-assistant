"""Built-in memory MCP server (in-process FastMCP).

Lets the model keep facts across conversations: what the user asks it to remember and
what he corrected it on (e.g. "хули ты ко мне в женском роде обращаешься" → the user is
a man). The facts live in a MemoryStore (src/memory.py, data/memory.json) and reach the
model through the <<<<<MEMORY>>>>> placeholder of the system prompt (src/prompt.py).
The tools catch their exceptions and return a human-readable Russian string (they
never raise), mirroring the reminders tool contract.
"""

from mcp.server.fastmcp import FastMCP


def build_memory_server(memory) -> FastMCP:
    """Build a FastMCP server exposing the memory tools (remember, forget) over
    `memory`, a MemoryStore."""
    mcp = FastMCP("memory")

    @mcp.tool(
        name="remember",
        description=(
            "Запомнить надолго факт о пользователе или доме: то, что пользователь просит "
            "запомнить, и то, в чём он тебя поправил и что не меняется от разговора к "
            "разговору (его пол и как к нему обращаться, имена домашних, привычки). fact — "
            "одна короткая фраза от третьего лица, например «Пользователь — мужчина, "
            "обращаться к нему в мужском роде». Запомненное ты видишь в разделе «Память» "
            "своего системного промпта в каждом разговоре."
        ),
    )
    async def remember(fact: str) -> str:
        try:
            added = memory.add(fact)
        except Exception as e:
            return f"Не удалось запомнить: {e}"
        return f"Запомнил: {fact.strip()}" if added else "Это уже запомнено."

    @mcp.tool(
        name="forget",
        description=(
            "Забыть запомненный факт, когда он устарел или неверен. fact — текст факта "
            "ровно так, как он стоит в разделе «Память». Чтобы исправить факт, забудь "
            "старый и запомни новый."
        ),
    )
    async def forget(fact: str) -> str:
        try:
            removed = memory.remove(fact)
        except Exception as e:
            return f"Не удалось забыть: {e}"
        return f"Забыл: {fact.strip()}" if removed else "Такого факта в памяти нет."

    return mcp
