"""Jev decision-model handler: one OpenRouter Decisions API request per utterance.

A recognition-chain handler like any `llm` provider, but no chat model: the
utterance goes to a decision model (typesafe/jev-1.14 by default) in one request
with five typed questions — the request kind, the device, the action, whether it
asks for several things, whether it asks for later. Only a confident command to one
device or scene becomes a tool call; anything else, and any failure of the call, is
no command: it declines to the next chain, or the last chain speaks reply_unmatched.
No local phrase matching runs here.

The device options and the action verbs are not its own settings: every request
reads the aliases and action verbs of the simple-nlu instance (the Commands page)
through Deps.instance_config, so an edit there reaches the next request without a
rebuild. Round 2 (a tool ran) speaks this handler's reply_done, or reply_error when
the tool failed, the way simple-nlu does.
"""

import json

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import ADVANCED_FIELD_EXTRA, SECRET_FIELD_EXTRA, Deps, Provider, register
from src.plugins.llm.base import LlmBackend
from src.plugins.llm.simple_nlu import (
    SimpleNluProvider,
    envelope,
    extract_number,
    fill_numbers,
    fill_singletons,
    has_tool_result,
    index_tools,
    last_user_text,
    parse_actions,
    parse_aliases,
    resolve_alias,
    round2_reply,
)
from src.stage_errors import Declined

# OpenRouter Decisions API: typed answers (choice / noul) with probabilities, shared by
# every decision model OpenRouter serves.
DECISIONS_API_URL = "https://openrouter.ai/api/alpha/decisions"


class JevConfig(BaseModel):
    # --- Decision request (baked into the backend at create()) ---
    model: str = Field(
        "typesafe/jev-1.14",
        title="Model",
        description="OpenRouter decision model asked about every phrase this chain gets.",
    )
    api_key: str = Field("", title="API key", json_schema_extra=SECRET_FIELD_EXTRA)
    min_confidence: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        title="Confidence threshold",
        description=(
            "The decision model's answers about the request kind, the device and the "
            "action (when it sets a slot) must each reach this confidence for the "
            "command to run."
        ),
    )
    timeout: float = Field(
        2.0,
        title="Request timeout",
        description="Past this the request gives up and the phrase counts as unmatched.",
        json_schema_extra={**ADVANCED_FIELD_EXTRA, "unit": "s"},
    )
    reply_unmatched: str = Field(
        "Не понял команду, повтори.",
        title="Reply when nothing matched",
    )
    reply_done: str = Field(
        "Готово.",
        title="Reply after a tool ran",
    )

    # --- Fields the LLM stage / orchestrator read live (mirror LlmConfig; see
    # reconfig._LLM_LIVE_LEAVES). max_tool_rounds minimum is 2: the tool round + the
    # final reply round both run. ---
    max_tool_rounds: int = Field(
        2, ge=2,
        title="Max rounds",
        description="Rounds per request: the tool call and the confirmation.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_empty: str = Field(
        "Я тебя не расслышал, повтори.",
        title="Reply when empty", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_empty_after_tools: str = Field(
        "Готово.",
        title="Reply when empty after a tool", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_rate_limit: str = Field(
        "Лимит запросов исчерпан. Попробуй ещё раз чуть позже.",
        title="Reply on rate limit", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_error: str = Field(
        "Что-то сломалось, попробуй ещё раз попозже.",
        title="Reply on error",
        description="Spoken after a tool returned an error, and when the handler fails.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )
    reply_error_after_tools: str = Field(
        "Сделал, но ответить не смог.",
        title="Reply on error after a tool", json_schema_extra=ADVANCED_FIELD_EXTRA,
    )


# The decision model's questions that do not depend on the catalog. The device and
# action questions are built per request from the aliases and the action verbs.
_KIND_QUESTION = {
    "type": "choice",
    "instructions": "What kind of request is this utterance to a home voice assistant?",
    "criteria": {
        "device_command": "A command to switch or adjust one smart-home device or scene",
        "other": (
            "A question, conversation, or a request for something else "
            "(weather, reminders, search, several devices at once)"
        ),
        "noise": "Not addressed to the assistant: a fragment, thanks, or meaningless speech",
    },
}
_COMPOUND_QUESTION = {
    "type": "noul",
    "instructions": (
        "Does the utterance ask for more than one distinct action or more than one device?"
    ),
    "criteria": {"true": "", "false": ""},
}
# A delay or a set time without a number («попозже», «через полчаса») would be
# dropped by the immediate tool call; phrases with a number never get here.
_DELAYED_QUESTION = {
    "type": "noul",
    "instructions": (
        "Does the utterance ask to do it later, after a delay, at a set time, "
        "or only for a limited time?"
    ),
    "criteria": {"true": "", "false": ""},
}


def _choice_answer(answers: dict, key: str) -> tuple:
    """(choice, confidence) of the choice answer `key`; raises KeyError, TypeError or
    ValueError when the answer is missing or malformed."""
    answer = answers[key]
    choice = answer["choice"]
    if answer["type"] != "choice" or not isinstance(choice, str):
        raise ValueError(f"{key}: not a choice answer: {answer!r}")
    return choice, float(answer["confidence"])


def _noul_answer(answers: dict, key: str) -> float:
    """The probability of the noul answer `key`; raises like _choice_answer."""
    answer = answers[key]
    if answer["type"] != "noul":
        raise ValueError(f"{key}: not a noul answer: {answer!r}")
    return float(answer["noul"])


def _targets(aliases: list[dict], tools_index: dict) -> dict[str, dict]:
    """The device options: one per distinct (tool, slot, value) the aliases resolve
    to, keyed "tool.slot=value", with every alias phrase mapped to it. Only enum slots
    of advertised tools: an explicit alias may name a tool that is not in the catalog
    or a free-form slot, and the args are built from the index."""
    targets: dict[str, dict] = {}
    for alias in aliases:
        resolved = resolve_alias(alias, tools_index)
        if resolved is None:
            continue
        tool, slot = resolved
        tinfo = tools_index.get(tool)
        if tinfo is None or slot not in tinfo["enum_slots"]:
            continue
        key = f"{tool}.{slot}={alias['value']}"
        target = targets.setdefault(key, {
            "tool": tool, "slot": slot, "value": alias["value"], "phrases": [],
        })
        target["phrases"].extend(alias["phrases"])
    return targets


def _questions(targets: dict[str, dict], actions: dict[str, list[str]]) -> dict:
    """The five questions of one decision request."""
    target_criteria = {key: ", ".join(t["phrases"]) for key, t in targets.items()}
    target_criteria["none"] = "No device or scene is named"
    action_criteria = {name: ", ".join(verbs) for name, verbs in actions.items()}
    action_criteria["none"] = "No action verb"
    return {
        "kind": _KIND_QUESTION,
        "target": {
            "type": "choice",
            "instructions": "Which device does the utterance refer to?",
            "criteria": target_criteria,
        },
        "action": {
            "type": "choice",
            "instructions": "What should be done with the device?",
            "criteria": action_criteria,
        },
        "compound": _COMPOUND_QUESTION,
        "delayed": _DELAYED_QUESTION,
    }


class JevBackend(LlmBackend):
    """LlmBackend contract over one decision request per utterance. Stateless across
    rounds: round 2 is recognized from the `messages` list alone."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        min_confidence: float,
        timeout: float,
        reply_unmatched: str,
        reply_done: str,
        reply_error: str,
        client: httpx.AsyncClient,
        instance_config,
    ):
        self._model = model
        self._api_key = api_key
        self._min_confidence = min_confidence
        self._timeout = timeout
        self._reply_unmatched = reply_unmatched
        self._reply_done = reply_done
        self._reply_error = reply_error
        self._client = client
        # (category, provider id) -> that instance's live config (Deps.instance_config).
        self._instance_config = instance_config

    async def complete(self, messages: list, tools: list | None, *, has_next: bool = False) -> dict:
        # ROUND 2: tool results are present -> speak a confirmation (or error).
        if has_tool_result(messages):
            content = round2_reply(messages, self._reply_done, self._reply_error)
            return envelope(content, None, "jev/none")

        # ROUND 1: one decision request -> one tool call, or no command, which declines
        # when a later chain would take it.
        command, reason = await self._decide(last_user_text(messages), tools or [])
        if command is not None:
            tool, args = command
            tool_calls = [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": tool,
                    # arguments is a JSON STRING (LlmStage json.loads it).
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }]
            # Empty content: avoid double-speak (the on_filler path would speak it).
            return envelope("", tool_calls, f"jev/{tool}")
        if has_next:
            raise Declined(self._reply_unmatched, reason=reason)
        return envelope(self._reply_unmatched, None, "jev/none")

    async def _decide(self, user_text: str, tools: list) -> tuple[tuple[str, dict] | None, str]:
        """Ask the decision model about the utterance in one request. Returns
        ((tool, args), "") only when the model is confident the utterance commands one
        device or scene, else (None, why not)."""
        # A number may be a level, a delay or a step from the current level, and none
        # of the questions tells which: leave such phrases to the next chain.
        if extract_number(user_text) is not None:
            return None, "decision model: a number is left to the next chain"
        # The names and verbs of the Commands page, read on every request.
        names = self._instance_config("llm", SimpleNluProvider.id)
        actions = parse_actions(names.actions)
        tools_index = index_tools(tools, set(actions))
        targets = _targets(parse_aliases(names.aliases), tools_index)
        if not targets:
            return None, "decision model: no device options"
        payload = {
            "model": self._model,
            "state": {"utterance": user_text},
            "questions": _questions(targets, actions),
        }
        try:
            resp = await self._client.post(
                DECISIONS_API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            answers = resp.json()["answers"]
            kind, kind_conf = _choice_answer(answers, "kind")
            target, target_conf = _choice_answer(answers, "target")
            action, action_conf = _choice_answer(answers, "action")
            compound = _noul_answer(answers, "compound")
            delayed = _noul_answer(answers, "delayed")
            entry = None if target == "none" else targets[target]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"jev: decision model call failed: {type(e).__name__}: {e}")
            return None, f"decision model error: {type(e).__name__}: {e}"
        logger.info(
            f"jev decision: kind={kind} ({kind_conf:.2f}), "
            f"target={target} ({target_conf:.2f}), action={action} ({action_conf:.2f}), "
            f"compound={compound:.2f}, delayed={delayed:.2f}"
        )

        min_conf = self._min_confidence
        if kind != "device_command" or kind_conf < min_conf:
            return None, f"decision model: kind {kind} ({kind_conf:.2f})"
        if entry is None or target_conf < min_conf:
            return None, f"decision model: target {target} ({target_conf:.2f})"
        if compound >= 0.5:
            return None, f"decision model: compound ({compound:.2f})"
        if delayed >= 0.5:
            return None, f"decision model: delayed ({delayed:.2f})"

        tool = entry["tool"]
        tinfo = tools_index[tool]
        args = {entry["slot"]: entry["value"]}
        state_slot = tinfo["state_slot"]
        if state_slot is not None:
            if action not in tinfo["slots"][state_slot]["enum"]:
                return None, f"decision model: action {action} does not fit {tool}"
            args[state_slot] = action
        _, level_action = fill_numbers(
            tinfo, user_text, {action}, args, level_on=names.level_on, off_token=names.off_token,
        )
        # The action decides on vs off, lock vs unlock: when it set a slot it must be
        # as confident as the kind and the device.
        if (state_slot is not None or level_action is not None) and action_conf < min_conf:
            return None, f"decision model: action {action} ({action_conf:.2f})"
        fill_singletons(tinfo, args)
        missing = tinfo["required"] - set(args)
        if missing:
            return None, (
                f"decision model: missing required slot for {tool}: {', '.join(sorted(missing))}"
            )
        return (tool, args), ""


@register
class JevProvider(Provider):
    category = "llm"
    id = "jev"
    label = "Jev (decision model)"
    ConfigModel = JevConfig
    uses_http_cloud = True

    def create(self, cfg: JevConfig, deps: Deps):
        return JevBackend(
            model=cfg.model,
            api_key=cfg.api_key,
            min_confidence=cfg.min_confidence,
            timeout=cfg.timeout,
            reply_unmatched=cfg.reply_unmatched,
            reply_done=cfg.reply_done,
            reply_error=cfg.reply_error,
            client=deps.http_cloud,
            instance_config=deps.instance_config,
        )
