"""Agentic tool-calling LLM stage with an explicit input/output contract.

The stage is a brick: it consumes an LlmRequest (system prompt assembled by the
orchestrator, prior history, the new user text, the originating device) and returns
an LlmResult (cleaned reply + observability data). The loop is provider-agnostic: it
drives a single-round-trip LlmBackend (built by a provider, see src.plugins.llm.base)
and never builds HTTP requests itself. The backend returns the raw provider JSON and
raises httpx.HTTPStatusError on non-2xx.
"""

import json
from dataclasses import dataclass

import httpx
from loguru import logger

from src.llm_text import clean_llm_output
from src.stage_errors import StageError


@dataclass(frozen=True)
class LlmRequest:
    """Input of the LLM stage. Assembled entirely by the orchestrator."""

    system_prompt: str          # assembled by the ORCHESTRATOR (src.prompt)
    history: list               # prior context messages spliced before the user turn
    user_text: str
    device: str | None = None   # passed through to hub.call, not via ContextVar here


@dataclass
class LlmResult:
    """Output of the LLM stage: the spoken reply + per-run observability data."""

    reply: str            # cleaned final text (clean_llm_output applied)
    model: str | None     # last model name seen across rounds
    tokens: int | None    # total_tokens summed across rounds
    rounds: list          # per-round dicts: {round, note, tokens, content, calls}
    request_debug: dict   # full model input: {system_prompt, context, user_text, tools}
    tool_used: bool       # whether any tool actually executed this run


class LlmStage:
    """The agentic tool loop as a stage brick.

    Thin and stateless: construct per run from the live runtime refs so hot-swapped
    backends/config apply naturally.
    """

    def __init__(self, backend, hub, cfg):
        self._backend = backend
        self._hub = hub
        self._cfg = cfg

    async def respond(self, req: LlmRequest, *, on_filler=None) -> LlmResult:
        """Drive the LLM backend with the request and return an LlmResult.

        Runs an agentic loop: model -> tool_calls -> execute via the hub -> feed
        results back -> final text. `req.device` names the speaker this run belongs
        to; it travels to in-process tools via `hub.call(name, args, device=...)`
        (the hub owns the ambient-device ContextVar, not this stage).

        `on_filler`, when given, is an optional `async (content: str, tool_names:
        list[str]) -> None` callback invoked for each tool-requesting round that also
        carries spoken content, so the caller can play an early "filler" line while
        the (slow) tool + final round run. Best-effort: it never affects the result
        or control flow, and its failures are swallowed with a warning.

        On failure raises StageError("llm", ...) (kind="rate_limit" for HTTP 429);
        the orchestrator maps it to a configured spoken phrase. The observability
        data accumulated before the failure (model/tokens/rounds/request) is
        attached to the exception as `err.partial`, so the orchestrator can still
        record what happened up to the failure point.
        """
        hub = self._hub
        cfg = self._cfg

        # Self-heal a startup race: pick up tools if a source was down at boot.
        await hub.ensure_tools()

        messages = [{"role": "system", "content": req.system_prompt}]
        if req.history:
            messages.extend(req.history)
        messages.append({"role": "user", "content": req.user_text})

        # Capture the full model input once for the run-detail debug view.
        request_debug = {
            "system_prompt": req.system_prompt,
            "context": list(req.history) if req.history else [],
            "user_text": req.user_text,
            "tools": list(hub.tools or []),
        }

        # Observability accumulators: model = last seen, tokens summed across rounds.
        model: str | None = None
        tokens: int | None = None
        rounds: list = []
        last_content = ""
        tool_executed = False

        def _error(message: str, *, kind: str = "error") -> StageError:
            # Build a StageError carrying the partial observability data, so the
            # orchestrator's error path records the same fields it always did
            # (the pre-R7 trace dict stayed partially filled after a failure).
            err = StageError("llm", message, kind=kind)
            err.partial = {
                "model": model,
                "tokens": tokens,
                "rounds": rounds,
                "request": request_debug,
            }
            return err

        def _result(reply: str) -> LlmResult:
            return LlmResult(
                reply=reply,
                model=model,
                tokens=tokens,
                rounds=rounds,
                request_debug=request_debug,
                tool_used=tool_executed,
            )

        round_no = 0
        for _ in range(cfg.max_tool_rounds):
            round_no += 1
            try:
                data = await self._backend.complete(messages, hub.tools or None)
            except httpx.HTTPStatusError as e:
                # Log full status + body for diagnostics; the orchestrator decides
                # what (if anything) is spoken to the user.
                logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 429:
                    raise _error(
                        f"rate limited: {e.response.text}", kind="rate_limit"
                    ) from e
                try:
                    reason = e.response.json().get("error", {}).get("message")
                except Exception:
                    reason = None
                raise _error(str(reason or e)) from e
            except httpx.HTTPError as e:
                logger.error(f"LLM request failed: {e}")
                raise _error(str(e)) from e

            choices = data.get("choices")
            if not choices:
                logger.error("No choices found in LLM response")
                raise _error("no choices in LLM response")

            message = choices[0]["message"]
            usage = data.get("usage", {})
            round_tokens = usage.get("total_tokens")
            logger.info(
                f"LLM response: model={data.get('model')} "
                f"total_tokens={round_tokens} "
                f"content={message.get('content')!r} "
                f"tool_calls={len(message.get('tool_calls') or [])}"
            )

            if data.get("model") is not None:
                model = data.get("model")
            if round_tokens is not None:
                tokens = (tokens or 0) + round_tokens

            # Append the assistant message verbatim (carries content + tool_calls).
            messages.append(message)
            last_content = message.get("content") or ""

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                rounds.append({
                    "round": round_no,
                    "note": "final answer",
                    "tokens": round_tokens,
                    "content": last_content,
                    "calls": [],
                })
                reply = clean_llm_output(last_content)
                if reply:
                    return _result(reply)
                return _result(
                    cfg.reply_empty_after_tools if tool_executed else cfg.reply_empty
                )

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

            # Execute each requested tool via the hub and feed results back. The
            # originating device rides along so the hub can publish it to in-process
            # tools for the duration of each call.
            round_calls = []
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, json.JSONDecodeError):
                    args = {}
                out = await hub.call(name, args, device=req.device)
                tool_executed = True
                logger.info(f"tool {name}({args}) -> {out!r}")
                round_calls.append({"name": name, "args": args, "result": out})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": out,
                })
            rounds.append({
                "round": round_no,
                "note": "tool call",
                "tokens": round_tokens,
                "content": last_content,
                "calls": round_calls,
            })
            # Loop again so the model can produce its final spoken reply.
            continue

        # Rounds exhausted: the model kept asking for tools without a final reply.
        logger.warning(f"Tool-calling loop exhausted after {cfg.max_tool_rounds} rounds")
        if last_content:
            return _result(clean_llm_output(last_content))
        raise _error("tool-calling rounds exhausted", kind="tool_rounds")
