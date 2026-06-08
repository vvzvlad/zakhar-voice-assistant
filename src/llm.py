"""Groq chat API client with an agentic tool-calling loop."""

import json

import httpx
from loguru import logger

from src.prompt import build_system_prompt
from src.settings import settings
from src.text import processing_response

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Hard cap on model<->tool round-trips to avoid an unbounded loop if the model
# keeps requesting tools.
MAX_TOOL_ROUNDS = 5

# Fallbacks so an empty/null model reply never reaches TTS as "".
EMPTY_REPLY_AFTER_TOOLS = "Готово."
EMPTY_REPLY_FALLBACK = "Я тебя не расслышала, повтори."


async def call_groq_api(client_ext: httpx.AsyncClient, hub, text: str, history: list | None = None) -> str:
    """Call Groq API with the given text and return plain-text result.

    `history` is the recent prior messages spliced between the system prompt and the
    new user turn so the model remembers the last few exchanges.

    The external client (proxied) is used for the Groq request and for building the
    system prompt (weather). Smart-home control is performed by calling MCP tools
    (advertised to the model and executed via `hub`).

    Runs an agentic loop: model -> tool_calls -> execute via MCP -> feed results
    back -> final text. On success returns the assistant text. On error returns a
    human-readable string starting with "Ошибка: ".
    """
    url = GROQ_API_URL
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.groq_api_key}"}

    # Self-heal a startup race: pick up tools if the MCP server was down at boot.
    await hub.ensure_tools()

    messages = [{"role": "system", "content": await build_system_prompt(client_ext)}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": text})

    last_content = ""
    tool_executed = False
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {
            "messages": messages,
            "model": settings.groq_model,
            "temperature": 0.8,
            "max_completion_tokens": 4096,
            "top_p": 0.95,
            "stream": False,
            "reasoning_effort": "medium",
        }
        # Advertise smart-home tools only when the hub actually has any.
        if hub.tools:
            payload["tools"] = hub.tools
            payload["tool_choice"] = "auto"

        try:
            response = await client_ext.post(url, headers=headers, json=payload, timeout=300)
            logger.info(f"Groq API response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"Groq API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                # If rate limited, return fixed Russian message
                if response.status_code == 429:
                    return (
                        "У меня кончились ресурсы на вас, мясных мешков. Я занимаюсь своими делами, обратитесь позже, и может быть, я вас обслужу, раз вы сами не в состоянии"
                    )
                # Try to extract detailed error message
                try:
                    err_json = response.json()
                    reason_msg = err_json.get("error", {}).get("message")
                except (ValueError, json.JSONDecodeError):
                    reason_msg = None
                return f"Ошибка: {reason_msg if reason_msg else error_msg}"

            response_json = response.json()
            choices = response_json.get("choices")
            if not choices:
                logger.error("No choices found in Groq API response")
                return "Ошибка: не найден ответ от модели"

            message = choices[0]["message"]
            usage = response_json.get("usage", {})
            logger.info(
                f"Groq response: model={response_json.get('model')} "
                f"total_time={usage.get('total_time')}s "
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

        except httpx.HTTPError as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            return f"Ошибка: {str(e)}"

    # Rounds exhausted: the model kept asking for tools without a final reply.
    logger.warning(f"Tool-calling loop exhausted after {MAX_TOOL_ROUNDS} rounds")
    if last_content:
        return processing_response(last_content)
    return "Ошибка: слишком много вызовов инструментов"
