"""Agentic tool-calling loop driving an injected LLM backend.

The loop is provider-agnostic: it drives a single-round-trip LlmBackend (built by a
provider, see src.plugins.llm.base) and never builds HTTP requests itself. The backend
returns the raw provider JSON and raises httpx.HTTPStatusError on non-2xx.
"""

import json

import httpx
from loguru import logger

from src.prompt import build_system_prompt
from src.run_context import current_device
from src.text import processing_response


async def call_llm_api(
    llm_backend,
    hub,
    text: str,
    *,
    core,
    llm_cfg,
    history: list | None = None,
    trace: dict | None = None,
    device: str | None = None,
    on_filler=None,
) -> str:
    """Drive the LLM backend with the given text and return a plain-text result.

    `history` is the recent prior messages spliced between the system prompt and the
    new user turn so the model remembers the last few exchanges.

    `core` is used to build the system prompt (date/time prefix + the configured prompt
    file). Weather and smart-home control are performed by calling tools advertised to
    the model and executed via `hub` (the multi-source tool hub).

    `trace`, when given, is populated for observability as the loop runs:
    `trace["model"]` (last seen), `trace["tokens"]` (summed total_tokens across rounds)
    and `trace["rounds"]` (one entry per round with its note/tokens/tool calls). It
    never changes the return value or any behavior; error paths may leave it partial.

    `device`, when given, names the speaker this run belongs to. It is published to
    in-process tools via the current_device ContextVar for the whole tool loop (set on
    entry, reset on exit) so a tool like set_reminder can target the originating speaker
    without it being an LLM-visible argument.

    `on_filler`, when given, is an optional `async (content: str, tool_names: list[str])
    -> None` callback invoked for each tool-requesting round that also carries spoken
    content, so the caller can play an early "filler" line while the (slow) tool + final
    round run. It never affects the return value or control flow.

    Runs an agentic loop: model -> tool_calls -> execute via the hub -> feed results
    back -> final text. On success returns the assistant text. On error returns a
    human-readable string starting with "Ошибка: ".
    """
    token = current_device.set(device)
    try:
        # Self-heal a startup race: pick up tools if a source was down at boot.
        await hub.ensure_tools()

        messages = [{"role": "system", "content": build_system_prompt(core)}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})

        # Observability trace (optional). Accumulate model/tokens/rounds as we go.
        if trace is not None:
            trace.setdefault("model", None)
            trace.setdefault("tokens", None)
            trace.setdefault("rounds", [])

        last_content = ""
        tool_executed = False
        round_no = 0
        for _ in range(llm_cfg.max_tool_rounds):
            round_no += 1
            try:
                data = await llm_backend.complete(messages, hub.tools or None)
            except httpx.HTTPStatusError as e:
                # Log full status + body for diagnostics (the returned string is spoken via TTS).
                logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 429:
                    return llm_cfg.reply_rate_limit
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
            round_tokens = usage.get("total_tokens")
            logger.info(
                f"LLM response: model={data.get('model')} "
                f"total_tokens={round_tokens} "
                f"content={message.get('content')!r} "
                f"tool_calls={len(message.get('tool_calls') or [])}"
            )

            if trace is not None:
                if data.get("model") is not None:
                    trace["model"] = data.get("model")
                if round_tokens is not None:
                    trace["tokens"] = (trace["tokens"] or 0) + round_tokens

            # Append the assistant message verbatim (carries content + tool_calls).
            messages.append(message)
            last_content = message.get("content") or ""

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                if trace is not None:
                    trace["rounds"].append({
                        "round": round_no,
                        "note": "final answer",
                        "tokens": round_tokens,
                        "calls": [],
                    })
                reply = processing_response(last_content)
                if reply:
                    return reply
                return llm_cfg.reply_empty_after_tools if tool_executed else llm_cfg.reply_empty

            # A tool-requesting round may also carry a short spoken "filler" line in its
            # content (the persona's "I'll go check it" placeholder). Forward it to the
            # caller so an early reply can be spoken while the (slow) tool + final round
            # run. Policy (speak-or-not, at-most-once) lives entirely in the callback; we
            # stay policy-free and just forward content + the tool names. Isolated: a
            # callback failure must never break the tool loop.
            if on_filler is not None:
                filler_text = (message.get("content") or "").strip()
                if filler_text:
                    tool_names = [tc["function"]["name"] for tc in tool_calls]
                    try:
                        await on_filler(filler_text, tool_names)
                    except Exception as e:  # noqa: BLE001 - filler is best-effort
                        logger.warning(f"on_filler callback failed: {e}")

            # Execute each requested tool via MCP and feed results back.
            round_calls = []
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
                round_calls.append({"name": name, "args": args, "result": out})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": out,
                })
            if trace is not None:
                trace["rounds"].append({
                    "round": round_no,
                    "note": "tool call",
                    "tokens": round_tokens,
                    "calls": round_calls,
                })
            # Loop again so the model can produce its final spoken reply.
            continue

        # Rounds exhausted: the model kept asking for tools without a final reply.
        logger.warning(f"Tool-calling loop exhausted after {llm_cfg.max_tool_rounds} rounds")
        if last_content:
            return processing_response(last_content)
        return "Ошибка: слишком много вызовов инструментов"
    finally:
        current_device.reset(token)
