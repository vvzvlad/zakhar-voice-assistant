"""Agentic tool-calling loop driving an injected LLM backend.

The loop is provider-agnostic: it drives a single-round-trip LlmBackend (built by a
provider, see src.plugins.llm.base) and never builds HTTP requests itself. The backend
returns the raw provider JSON and raises httpx.HTTPStatusError on non-2xx.
"""

import json

import httpx
from loguru import logger

from src.prompt import build_system_prompt
from src.text import processing_response

# Fallbacks so an empty/null model reply never reaches TTS as "".
EMPTY_REPLY_AFTER_TOOLS = "Готово."
EMPTY_REPLY_FALLBACK = "Я тебя не расслышала, повтори."


async def call_llm_api(
    llm_backend,
    hub,
    text: str,
    *,
    weather_client: httpx.AsyncClient,
    core,
    max_tool_rounds: int,
    history: list | None = None,
) -> str:
    """Drive the LLM backend with the given text and return a plain-text result.

    `history` is the recent prior messages spliced between the system prompt and the
    new user turn so the model remembers the last few exchanges.

    `weather_client` (proxied) and `core` are used to build the system prompt (which
    encapsulates weather + the configured prompt file). Smart-home control is performed
    by calling MCP tools (advertised to the model and executed via `hub`).

    Runs an agentic loop: model -> tool_calls -> execute via MCP -> feed results
    back -> final text. On success returns the assistant text. On error returns a
    human-readable string starting with "Ошибка: ".
    """
    # Self-heal a startup race: pick up tools if the MCP server was down at boot.
    await hub.ensure_tools()

    messages = [{"role": "system", "content": await build_system_prompt(weather_client, core)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})

    last_content = ""
    tool_executed = False
    for _ in range(max_tool_rounds):
        try:
            data = await llm_backend.complete(messages, hub.tools or None)
        except httpx.HTTPStatusError as e:
            # Log full status + body for diagnostics (the returned string is spoken via TTS).
            logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429:
                return "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии"
            try:
                reason = e.response.json().get("error", {}).get("message")
            except Exception:
                reason = None
            return f"Ошибка: {reason or e}"
        except httpx.HTTPError as e:
            logger.error(f"LLM request failed: {e}")
            return f"Ошибка: {e}"

        choices = data.get("choices")
        if not choices:
            logger.error("No choices found in LLM response")
            return "Ошибка: не найден ответ от модели"

        message = choices[0]["message"]
        usage = data.get("usage", {})
        logger.info(
            f"LLM response: model={data.get('model')} "
            f"total_tokens={usage.get('total_tokens')} "
            f"content={message.get('content')!r} "
            f"tool_calls={len(message.get('tool_calls') or [])}"
        )

        # Append the assistant message verbatim (carries content + tool_calls).
        messages.append(message)
        last_content = message.get("content") or ""

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            reply = processing_response(last_content)
            if reply:
                return reply
            return EMPTY_REPLY_AFTER_TOOLS if tool_executed else EMPTY_REPLY_FALLBACK

        # Execute each requested tool via MCP and feed results back.
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, json.JSONDecodeError):
                args = {}
            out = await hub.call(name, args)
            tool_executed = True
            logger.info(f"tool {name}({args}) -> {out!r}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": out,
            })
        # Loop again so the model can produce its final spoken reply.
        continue

    # Rounds exhausted: the model kept asking for tools without a final reply.
    logger.warning(f"Tool-calling loop exhausted after {max_tool_rounds} rounds")
    if last_content:
        return processing_response(last_content)
    return "Ошибка: слишком много вызовов инструментов"
