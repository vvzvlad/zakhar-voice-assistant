"""Jev decision-model handler: one OpenRouter Decisions API request per utterance.

A recognition-chain handler like any `llm` provider, but no chat model: the
utterance goes to a decision model (typesafe/jev-1.14 by default) in one request
with five typed questions — the request kind, the device, the action, whether it
asks for several things, whether it asks for later. Only a confident command to one
device or scene becomes a tool call; anything else, and any failure of the call, is
no command: it declines to the next chain, or the last chain speaks reply_unmatched.
The only local phrase matching: a phrase that names two devices or scenes by their
aliases declines without a request (a dropped «и» reads as one place to the model).

The device options and the action verbs are not its own settings: every request
reads the aliases, the action verbs and the levels an on- or off-verb sets from the
command catalog (core.commands, the Commands page) through Deps.commands, so an edit
there reaches the next request without a rebuild. Round 2 (a tool ran) speaks this
handler's reply_done, or reply_error when the tool failed, the way simple-nlu does.
"""

import json

import httpx
from loguru import logger
from pydantic import Field

from src.commands import (
    build_lemmatizer,
    envelope,
    extract_number,
    fill_numbers,
    fill_singletons,
    has_tool_result,
    index_tools,
    last_user_text,
    lemmas,
    parse_actions,
    parse_aliases,
    resolve_alias,
    round2_reply,
)
from src.plugins.base import (
    ADVANCED_FIELD_EXTRA,
    PERCENT_FIELD_EXTRA,
    PHRASE_FIELD_EXTRA,
    SECRET_FIELD_EXTRA,
    Deps,
    Provider,
    register,
)
from src.plugins.llm._decisions import DECISIONS_API_URL, noul_answer
from src.plugins.llm.base import LlmBackend, ReplyConfig
from src.stage_errors import Declined


class JevConfig(ReplyConfig):
    # --- Decision request (baked into the backend at create()) ---
    model: str = Field(
        "typesafe/jev-1.14",
        title="Model",
        description="OpenRouter decision model asked about every phrase this chain gets.",
    )
    api_key: str = Field("", title="API key", json_schema_extra=SECRET_FIELD_EXTRA)
    # The answers about the request kind, the device and the action (when it sets a slot)
    # must each reach this confidence for the command to run.
    min_confidence: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        title="Confidence threshold",
        description="How sure the decision model must be of the request, device and action to run it.",
        json_schema_extra=PERCENT_FIELD_EXTRA,
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
        json_schema_extra=PHRASE_FIELD_EXTRA,
    )
    reply_done: str = Field(
        "Готово.",
        title="Reply after a tool ran",
        json_schema_extra=PHRASE_FIELD_EXTRA,
    )
    # The ReplyConfig field with this handler's own default and minimum: the tool round
    # and the final reply round both run.
    max_tool_rounds: int = Field(
        2, ge=2,
        title="Max rounds",
        description="Rounds per request: the tool call and the confirmation.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )


# The decision model's questions that do not depend on the catalog. The device and
# action questions are built per request from the aliases and the action verbs.
_KIND_QUESTION = {
    "type": "choice",
    "instructions": "What kind of request is this utterance to a home voice assistant?",
    "criteria": {
        "device_command": "A command to switch or adjust one smart-home device or scene",
        # One option for everything that is not a command: _decide declines all of it
        # alike, so splitting it only spreads the probability over two labels.
        "other": (
            "Anything else: a question, conversation, a request for something else "
            "(weather, reminders, search, several devices at once), a fragment, "
            "thanks, or meaningless speech"
        ),
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


def _percents(probabilities: dict) -> dict[str, int]:
    """The options' probabilities in whole percents summing to exactly 100, highest
    first. The API rounds each probability, so their sum drifts off 1: they are
    divided by their sum, and the percents lost to rounding go to the largest
    remainders."""
    total = sum(float(p) for p in probabilities.values())
    if not total > 0:
        raise ValueError(f"probabilities sum to {total}")
    exact = {k: float(p) * 100 / total for k, p in probabilities.items()}
    pct = {k: int(v) for k, v in exact.items()}
    short = 100 - sum(pct.values())
    for k in sorted(exact, key=lambda k: exact[k] - pct[k], reverse=True)[:short]:
        pct[k] += 1
    return dict(sorted(pct.items(), key=lambda kv: kv[1], reverse=True))


def _choice_answer(answers: dict, key: str) -> tuple[str, float, str]:
    """(choice, confidence, summary) of the choice answer `key`. The summary reads
    «other (other 74%, device_command 26%; confidence 48%)»: every option's normalized
    probability, options at 0% left out, then the confidence, which measures how
    concentrated the distribution is and is not the chosen option's probability.
    Raises KeyError, TypeError or ValueError when the answer is missing or malformed."""
    answer = answers[key]
    choice = answer["choice"]
    if answer["type"] != "choice" or not isinstance(choice, str):
        raise ValueError(f"{key}: not a choice answer: {answer!r}")
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, dict):
        raise ValueError(f"{key}: no probabilities: {answer!r}")
    confidence = float(answer["confidence"])
    shown = ", ".join(f"{k} {v}%" for k, v in _percents(probabilities).items() if v)
    return choice, confidence, f"{choice} ({shown}; confidence {confidence:.0%})"


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


def _named_targets(targets: dict[str, dict], text: str, lemmatize) -> list[str]:
    """Keys of the targets the utterance names: every word of one of their phrases is
    in it, by lemma, and that phrase has a word no other target's phrases use. A shared
    word alone names nothing: «свет» of the room light in «свет в прихожей».

    Words are matched without their role, so a place or time word that is another
    target's own alias counts as that target: «тусклый свет в зале» names the dim and
    the bright light, «закрой дверь на ночь» the lock and the night scene. Such a
    command declines to the next chain, which still runs it, only slower. An alias
    that reuses such a word («тусклый свет в зале») makes it shared, so it stops
    naming the other target anywhere."""
    phrases = {
        key: [frozenset(lemmas(ph, lemmatize)) for ph in t["phrases"]]
        for key, t in targets.items()
    }
    words = frozenset(lemmas(text, lemmatize))
    named = []
    for key, own in phrases.items():
        others = frozenset().union(*(p for k, ps in phrases.items() if k != key for p in ps))
        if any(p <= words and p - others for p in own):
            named.append(key)
    return named


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
        commands,
    ):
        self._model = model
        self._api_key = api_key
        self._min_confidence = min_confidence
        self._timeout = timeout
        self._reply_unmatched = reply_unmatched
        self._reply_done = reply_done
        self._reply_error = reply_error
        self._client = client
        # () -> the live command catalog, core.commands (Deps.commands).
        self._commands = commands
        self._lemmatize = build_lemmatizer()

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
        catalog = self._commands()
        actions = parse_actions(catalog.actions)
        tools_index = index_tools(tools, set(actions))
        targets = _targets(parse_aliases(catalog.aliases), tools_index)
        if not targets:
            return None, "decision model: no device options"
        named = _named_targets(targets, user_text, self._lemmatize)
        if len(named) > 1:
            return None, f"decision model: several devices named: {', '.join(named)}"
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
            kind, kind_conf, kind_said = _choice_answer(answers, "kind")
            target, target_conf, target_said = _choice_answer(answers, "target")
            action, action_conf, action_said = _choice_answer(answers, "action")
            compound = noul_answer(answers, "compound")
            delayed = noul_answer(answers, "delayed")
            entry = None if target == "none" else targets[target]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"jev: decision model call failed: {type(e).__name__}: {e}")
            return None, f"decision model error: {type(e).__name__}: {e}"
        logger.info(
            f"jev decision: kind={kind_said}, target={target_said}, action={action_said}, "
            f"compound={compound:.0%}, delayed={delayed:.0%}"
        )

        min_conf = self._min_confidence
        if kind != "device_command" or kind_conf < min_conf:
            return None, f"decision model: kind {kind_said}"
        if entry is None or target_conf < min_conf:
            return None, f"decision model: target {target_said}"
        if compound >= 0.5:
            return None, f"decision model: compound ({compound:.0%})"
        if delayed >= 0.5:
            return None, f"decision model: delayed ({delayed:.0%})"

        tool = entry["tool"]
        tinfo = tools_index[tool]
        args = {entry["slot"]: entry["value"]}
        state_slot = tinfo["state_slot"]
        if state_slot is not None:
            if action not in tinfo["slots"][state_slot]["enum"]:
                return None, f"decision model: action {action} does not fit {tool}"
            args[state_slot] = action
        _, level_action = fill_numbers(
            tinfo, user_text, {action}, args, level_on=catalog.level_on, off_token=catalog.off_token,
        )
        # The action decides on vs off, lock vs unlock: when it set a slot it must be
        # as confident as the kind and the device.
        if (state_slot is not None or level_action is not None) and action_conf < min_conf:
            return None, f"decision model: action {action_said}"
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
            commands=deps.commands,
        )
