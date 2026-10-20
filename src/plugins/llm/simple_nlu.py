"""Offline, LLM-free NLU 'LLM' provider — deterministic matching + a command classifier.

It slots into the pipeline's `llm` stage like any other provider, but uses NO
LLM and NO network: pymorphy3 morphology, a scikit-learn classifier and rapidfuzz.
A Russian utterance becomes a smart-home tool call in two steps:
- the deterministic match: an alias phrase of a device, by lemma, plus an action
  verb in an imperative or infinitive form; with a later chain behind this one
  (strict mode) every word of the utterance must be explained by the command; in
  both modes the linguistic gate below must pass it;
- when it finds no command, a classifier over concrete commands (one class per
  device and action, plus a "not a command" class of ordinary assistant requests).
  Its prediction acts only at or above a probability threshold, when every word
  was heard and no number was said, past a linguistic gate that stops questions,
  negations, past tense, an infinitive that asks («мне открыть дверь»), other verbs,
  several actions, quantifiers, requests for later, a second request after «и» and
  several devices, when no indicative verb or predicative word states or wishes
  rather than commands, the action verbs agree with the class, a word names the
  device, every other word names that device or is a filler word, and the action
  is not one left to the direct match (unlock).
The OpenAI/Groq-shaped envelope and the LlmStage agentic loop are exactly the same
as for the cloud LLM providers, so the backend emits `tool_calls` to drive the
tools and, on the next round (tool results present), speaks a short confirmation.

Design notes:
- The names, verbs, filler words and the levels an on/off verb sets are the command
  catalog (core.commands), read through Deps.commands when the backend is built; a
  catalog change rebuilds it.
  Its parsing, morphology and number helpers are shared with the jev handler in
  src/commands.py.
- Heavy deps (pymorphy3, scikit-learn, rapidfuzz) are imported LAZILY, never at
  module top level: importing this module for the registry stays light and must
  not fail if those libs are absent. pymorphy3 is needed to build the model (without
  it nothing matches); without scikit-learn there is no classifier, without
  rapidfuzz no tolerance for a misheard first filler word.
- The backend is STATELESS across rounds: nothing per-request is stashed on self
  (it is shared across concurrent speakers). Round 2 is reconstructed purely from
  the `messages` list (presence of a role=="tool" message).
- The model (indices + classifier) is built once per distinct tool catalog and
  cached, keyed by a fingerprint of the advertised tools.
"""

import asyncio
import functools
import hashlib
import json
import random
from pathlib import Path

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
    is_number_word,
    last_user_text,
    lemmas,
    morphology,
    nlu_vocabulary,
    parse_actions,
    parse_aliases,
    resolve_alias,
    round2_reply,
    tokenize,
    warn_missing,
)
from src.plugins.base import ADVANCED_FIELD_EXTRA, PERCENT_FIELD_EXTRA, PHRASE_FIELD_EXTRA, Deps, Provider, register
from src.plugins.llm.base import LlmBackend, ReplyConfig
from src.stage_errors import Declined


class SimpleNluConfig(ReplyConfig):
    # --- NLU configuration (baked into the backend at create(); changing any of
    # these triggers a backend rebuild via reconfig, which is correct) ---
    # When the direct match finds no command, the classifier's command runs only at or
    # above this probability.
    classifier_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        title="Classifier confidence threshold",
        description="How sure the classifier must be to run a command the direct match missed.",
        json_schema_extra=PERCENT_FIELD_EXTRA,
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
    # The ReplyConfig field with this engine's own default and minimum: the tool round
    # and the final reply round both run.
    max_tool_rounds: int = Field(
        2, ge=2,
        title="Max rounds",
        description="Rounds per request: the tool call and the confirmation.",
        json_schema_extra=ADVANCED_FIELD_EXTRA,
    )


# --------------------------------------------------------------------------- #
# Backend.
# --------------------------------------------------------------------------- #

def _one_edit(a: str, b: str) -> bool:
    """`a` and `b` are at most one edit apart; False without rapidfuzz."""
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        warn_missing("rapidfuzz")
        return False
    return Levenshtein.distance(a, b) <= 1


def _no_match(reason: str) -> dict:
    """The detection result for an utterance that is no command; `reason` says why."""
    return {"is_command": False, "tool": None, "args": {}, "name": "none", "reason": reason}


# --------------------------------------------------------------------------- #
# The classifier's training data and the linguistic gate.
# --------------------------------------------------------------------------- #

# The "not a command" class: ordinary assistant requests, one per line ('#' lines
# are comments).
_NEGATIVES_PATH = Path(__file__).with_name("simple_nlu_negatives.txt")
_NONE_CLASS = "NONE"
# The level a class of a level-slot tool sets: "on", "off" or "number" (_realize).
_LEVEL_KEY = "_level"
# Numbers the synthetic phrases of a "number" class carry.
_SYNTH_NUMBERS = ("10", "двадцать", "30", "пятьдесят", "70", "сто", "5", "восемьдесят")

_QUESTION_WORDS = frozenset({
    "кто", "что", "чего", "почему", "зачем", "когда", "где", "куда", "откуда", "как",
    "какой", "какая", "какое", "какие", "каком", "сколько", "ли", "разве", "неужели", "чей",
})
# Word forms with ё folded to е, as tokenize gives them.
_QUANTIFIERS = frozenset({
    "весь", "все", "всех", "всем", "всей", "всю", "вся", "всего", "везде", "всюду", "каждый",
})
# With one of these in the phrase an infinitive asks rather than commands («мне
# открыть дверь?»); an imperative with «мне» («включи мне свет») still commands.
_DATIVE_PRONOUNS = frozenset({"мне", "нам", "тебе", "вам", "ему", "ей", "им"})
# Actions the classifier never runs: only an exact direct match unlocks the door.
_DIRECT_ONLY_ACTIONS = frozenset({"unlock"})
# A request for later, which the immediate call would drop: these word forms (ё
# folded to е; by form, since pymorphy3 gives «ночью» the lemma «ночь», a scene
# alias) and any form of these lemmas.
_DELAY_WORDS = frozenset({
    "через", "завтра", "послезавтра", "потом", "позже", "попозже", "утром", "днем",
    "вечером", "ночью",
})
_DELAY_LEMMAS = frozenset({"минута", "час", "секунда"})
# What a narrowed decoder emits for a word outside its vocabulary.
_UNHEARD = "[unk]"
# Imperatives the gate lets through besides the action verbs and the filler words.
_GATE_IMPERATIVES = frozenset({"поставь", "сделай", "спасибо"})


@functools.cache
def _negative_lemmas() -> tuple[str, ...]:
    """The lemmatized "not a command" phrases, read once per process. Every second
    one starts with the wake word, as a real utterance often does."""
    lemmatize = build_lemmatizer()
    lines = (line.strip() for line in _NEGATIVES_PATH.read_text(encoding="utf-8").splitlines())
    phrases = [line for line in lines if line and not line.startswith("#")]
    return tuple(
        " ".join(lemmas(f"захар {p}" if i % 2 else p, lemmatize)) for i, p in enumerate(phrases)
    )


def _command_classes(
    aliases: list[dict], tools_index: dict, actions: dict[str, list[str]], lemmatize
) -> tuple[dict[str, tuple[str, dict]], list[str], list[str]]:
    """The classifier's command classes and their synthetic phrases, from the aliases
    and the action verbs. A class is one concrete call: an alias's device with one
    value of its tool's state slot; for a tool with a level slot instead, the device
    switched on, off, or set to a number; for a tool with neither (a scene), the
    alias alone. Returns ({class: (tool, args)}, phrases, their classes)."""
    rng = random.Random(0)
    classes: dict[str, tuple[str, dict]] = {}
    texts: list[str] = []
    labels: list[str] = []
    for alias in aliases:
        resolved = resolve_alias(alias, tools_index)
        if resolved is None or resolved[0] not in tools_index:
            continue
        tool, slot = resolved
        tinfo = tools_index[tool]
        state_slot = tinfo["state_slot"]
        base = {slot: alias["value"]}
        if state_slot is not None:
            variants = [({**base, state_slot: action}, actions.get(action, []), "verb")
                        for action in tinfo["slots"][state_slot]["enum"]]
        elif tinfo["level_slot"] is not None:
            variants = [({**base, _LEVEL_KEY: "on"}, actions.get("on", []), "verb"),
                        ({**base, _LEVEL_KEY: "off"}, actions.get("off", []), "verb"),
                        ({**base, _LEVEL_KEY: "number"}, actions.get("on", []), "number")]
        else:
            variants = [(base, actions.get("on", []), "scene")]
        for args, verbs, mode in variants:
            name = f"{tool}{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            classes[name] = (tool, args)
            # The verbs and their infinitives («включи», «включить»).
            verbs = sorted(set(verbs) | {" ".join(lemmas(v, lemmatize)) for v in verbs})
            for p in alias["phrases"]:
                if mode == "number":
                    phrase_texts = [f"{p} {n}" for n in _SYNTH_NUMBERS]
                    phrase_texts += [f"{p} на {n}" for n in _SYNTH_NUMBERS]
                    if verbs:
                        phrase_texts += [f"захар {rng.choice(verbs)} {p} на {n}" for n in _SYNTH_NUMBERS]
                else:
                    phrase_texts = [
                        t for v in verbs for t in (
                            f"{v} {p}", f"{p} {v}", f"захар {v} {p}", f"{v} {p} пожалуйста",
                            f"захар {v} {p} спасибо",
                        )
                    ]
                    if mode == "scene":
                        phrase_texts += [p, f"захар {p}", f"{p} пожалуйста", f"режим {p}"]
                texts += phrase_texts
                labels += [name] * len(phrase_texts)
    return classes, texts, labels


def _fit_classifier(texts: list[str], labels: list[str]):
    """A char n-gram TF-IDF + logistic SGD classifier fitted on the lemmatized
    phrases, or None without scikit-learn or when the fit fails. SGD rather than
    LogisticRegression for the retrain time: the catalog retrains on every change."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
    except ImportError:
        warn_missing("scikit-learn")
        return None
    try:
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=None, random_state=0)),
        ]).fit(texts, labels)
    except Exception as e:  # noqa: BLE001 - never let training crash the request
        logger.warning(f"simple-nlu: classifier training failed: {e}")
        return None


class SimpleNluBackend(LlmBackend):
    """LlmBackend contract WITHOUT any LLM: a deterministic match, then a classifier
    of concrete commands behind a linguistic gate.

    Emits OpenAI-shaped tool_calls to drive the smart-home tools. The model
    (tool index, deterministic phrase index, gate vocabulary, classifier) is built
    once per distinct tool catalog (keyed by a fingerprint) and cached.
    """

    def __init__(
        self,
        *,
        aliases_text: str,
        actions_text: str,
        filler_text: str,
        classifier_threshold: float,
        level_on: int,
        off_token: str,
        reply_unmatched: str,
        reply_done: str,
        reply_error: str,
    ):
        self._aliases = parse_aliases(aliases_text)
        self._actions = parse_actions(actions_text)
        # Filler words as exact normalized word forms.
        self._fillers = frozenset(tokenize(filler_text))
        # The ones a misheard first word may be one letter off («лахар» for «захар»).
        self._long_fillers = tuple(f for f in self._fillers if len(f) >= 4)
        self._classifier_threshold = classifier_threshold
        self._level_on = level_on
        self._off_token = off_token
        self._reply_unmatched = reply_unmatched
        self._reply_done = reply_done
        self._reply_error = reply_error
        # Model cache, rebuilt when the advertised tools change.
        self._lock = asyncio.Lock()
        self._fingerprint: str | None = None
        self._model: dict | None = None

    # --- public contract -------------------------------------------------- #

    async def warm_up(self, tools: list) -> dict | None:
        """Build the model for this tool catalog unless it is already built, and
        return it. The app calls it ahead of the first request (at boot and after
        every reconfiguration), so that request does not pay for the training."""
        fingerprint = self._fingerprint_tools(tools)
        # Rebuild the model only when the tool catalog changed. Training is
        # blocking (pymorphy3 + sklearn fit) so it runs in a worker thread.
        async with self._lock:
            if fingerprint != self._fingerprint:
                self._model = await asyncio.to_thread(self._train, tools)
                self._fingerprint = fingerprint
            return self._model

    async def complete(self, messages: list, tools: list | None, *, has_next: bool = False) -> dict:
        model = await self.warm_up(tools or [])

        # ROUND 2: tool results are present -> speak a confirmation (or error).
        if has_tool_result(messages):
            content = round2_reply(messages, self._reply_done, self._reply_error)
            return self._envelope(content, None, "none")

        # ROUND 1: classify the user utterance into a tool call (or no-match). With a
        # later chain behind this one, detection is strict and a no-match declines,
        # so the next chain answers instead.
        user_text = last_user_text(messages)
        intent = self._detect(user_text, model, strict=has_next)
        if intent["is_command"] and intent["tool"]:
            tool_calls = [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": intent["tool"],
                    # arguments is a JSON STRING (LlmStage json.loads it).
                    "arguments": json.dumps(intent["args"], ensure_ascii=False),
                },
            }]
            # Empty content: avoid double-speak (the on_filler path would speak it).
            return self._envelope("", tool_calls, intent["name"])
        if has_next:
            raise Declined(self._reply_unmatched, reason=intent["reason"])
        return self._envelope(self._reply_unmatched, None, "none")

    @staticmethod
    def _envelope(content: str, tool_calls, intent_name: str) -> dict:
        return envelope(content, tool_calls, f"simple-nlu/{intent_name or 'none'}")

    # --- fingerprint ----------------------------------------------------- #

    @staticmethod
    def _fingerprint_tools(tools: list) -> str:
        items = []
        for t in tools:
            fn = t.get("function", {})
            items.append((fn.get("name", ""), json.dumps(fn.get("parameters", {}), sort_keys=True)))
        items.sort()
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest()

    # --- training (runs in a worker thread) ------------------------------ #

    def _train(self, tools: list) -> dict | None:
        """Build the immutable model dict from the advertised tools, or None without
        pymorphy3.

        Indexes each tool's params, resolves aliases to (tool, slot, value), builds
        the deterministic phrase and verb indices, the gate's alias index and (with
        scikit-learn and at least one command class) the classifier.
        """
        morph = morphology()
        if morph is None:
            return None
        lemmatize = morph.lemma
        tools_index = index_tools(tools, set(self._actions))

        # Resolve aliases to (tool, slot, value). Auto form locates the tool+param
        # whose enum contains the value; unroutable aliases are skipped.
        aliases_resolved: list[dict] = []
        for alias in self._aliases:
            resolved = resolve_alias(alias, tools_index)
            if resolved is None:
                logger.warning(
                    f"simple-nlu: alias value {alias['value']!r} not found in any tool enum; skipped"
                )
                continue
            phrase_lemmas = [frozenset(lemmas(p, lemmatize)) for p in alias["phrases"]]
            aliases_resolved.append({
                "tool": resolved[0],
                "slot": resolved[1],
                "value": alias["value"],
                "phrase_lemmas": [pl for pl in phrase_lemmas if pl],
            })
        per_tool_alias_phrases: dict[str, list[frozenset]] = {}
        for entry in aliases_resolved:
            per_tool_alias_phrases.setdefault(entry["tool"], []).extend(entry["phrase_lemmas"])

        # Verb lemma -> action, from the verb words of the action verbs («зажги» of
        # «зажги свет»). Which forms of the utterance count is _verb_action's call.
        action_of: dict[str, str] = {}
        for action, verbs in self._actions.items():
            for verb in verbs:
                for tok in tokenize(verb):
                    parse = morph.parses(tok)[0]
                    if parse.tag.POS in ("VERB", "INFN"):
                        action_of[parse.normal_form] = action

        classes, texts, labels = _command_classes(self._aliases, tools_index, self._actions, lemmatize)
        classifier = None
        if texts:
            negatives = _negative_lemmas()
            classifier = _fit_classifier(
                [" ".join(lemmas(t, lemmatize)) for t in texts] + list(negatives),
                labels + [_NONE_CLASS] * len(negatives),
            )

        return {
            "morph": morph,
            "tools_index": tools_index,
            "aliases_resolved": aliases_resolved,
            "per_tool_alias_phrases": per_tool_alias_phrases,
            "action_of": action_of,
            # Every alias phrase as a lemma set with its value, for the gate.
            "alias_sets": [
                (frozenset(lemmas(p, lemmatize)), a["value"])
                for a in self._aliases for p in a["phrases"]
            ],
            "classes": classes,
            "classifier": classifier,
        }

    # --- detection ------------------------------------------------------- #

    def _detect(self, user_text: str, model: dict | None, *, strict: bool = False) -> dict:
        """Classify `user_text` into {is_command, tool, args, name, reason}; `reason`
        says why the utterance is no command ("" for a command).

        The deterministic match runs first (_match; `strict` keeps its command only
        when every word is explained) and its command must pass the gate; when it
        finds no command, the classifier gets the utterance (_classify)."""
        if not model or not user_text:
            return _no_match("")
        found = self._match(user_text, model, strict=strict)
        if found["is_command"]:
            why = self._gate(user_text, model)
            if why is not None:
                return _no_match(f"gate: {why}")
        if found["is_command"] or model["classifier"] is None:
            return found
        return self._classify(user_text, model, found["reason"])

    def _match(self, user_text: str, model: dict, *, strict: bool) -> dict:
        """The deterministic match: the tool whose alias phrase is fully in the
        utterance by lemma (the longest phrase wins) and, when the tool has a state
        slot, an action verb of that slot. In `strict` mode the command stands only
        when every word is explained (see _unexplained)."""
        tools_index = model["tools_index"]
        tokens = tokenize(user_text)
        user_lemmas = lemmas(user_text, model["morph"].lemma)
        user_lemma_set = set(user_lemmas)
        verb_actions = [self._verb_action(tok, model) for tok in tokens]
        present_actions = {a for a in verb_actions if a}

        best_tool = None
        best_len = -1
        verbless = []  # tools whose alias phrase matched but no applicable verb is present
        for tname, tinfo in tools_index.items():
            matched_phrase_len = max(
                (len(pl) for pl in model["per_tool_alias_phrases"].get(tname, []) if pl <= user_lemma_set),
                default=-1,
            )
            if matched_phrase_len < 0:
                continue
            state_slot = tinfo["state_slot"]
            if state_slot is not None and not present_actions & set(tinfo["slots"][state_slot]["enum"]):
                verbless.append(tname)
                continue
            if matched_phrase_len > best_len:
                best_len = matched_phrase_len
                best_tool = tname
        if best_tool is None:
            if verbless:
                return _no_match(f"no applicable verb for {', '.join(verbless)}")
            return _no_match("no alias phrase matched")

        args, evidence = self._fill_slots(best_tool, user_text, user_lemma_set, verb_actions, model)
        # Required-slot validation is the main false-positive guard.
        missing = tools_index[best_tool]["required"] - set(args)
        if missing:
            return _no_match(f"missing required slot for {best_tool}: {', '.join(sorted(missing))}")
        if strict:
            unknown = self._unexplained(tokens, user_lemmas, verb_actions, evidence)
            if unknown:
                return _no_match(f"unknown words: {', '.join(unknown)}")
        return {"is_command": True, "tool": best_tool, "args": args, "name": best_tool, "reason": ""}

    @staticmethod
    def _verb_action(token: str, model: dict) -> str | None:
        """The action `token` commands: it has an imperative or infinitive parse of an
        action verb («включи», «включить»). A past form or a participle («закрыл»,
        «выключен») shares the verb's lemma but commands nothing. Every parse counts,
        not only the most likely one."""
        action_of = model["action_of"]
        for parse in model["morph"].parses(token):
            if parse.normal_form in action_of and ("impr" in parse.tag or parse.tag.POS == "INFN"):
                return action_of[parse.normal_form]
        return None

    def _unexplained(
        self, tokens: list[str], user_lemmas: list[str], verb_actions: list, evidence: dict
    ) -> list[str]:
        """The words of the utterance nothing in the command accounts for. A word is
        explained when it is an action verb of an action the command carries, its
        lemma is in an alias phrase that filled a slot, a number filled a slot and
        it is a number, «на», «до» or a form of «процент», it is a filler word, or it
        is the first word and one letter off a filler word of four letters or more
        (a misheard «захар»). The three lists are parallel."""
        phrase_lemmas = set().union(*evidence["phrases"])
        return [
            tok for i, (tok, lem, action) in enumerate(zip(tokens, user_lemmas, verb_actions))
            if action not in evidence["actions"]
            and lem not in phrase_lemmas
            and not (evidence["number"] and is_number_word(tok, lem))
            and not self._is_filler(i, tok)
        ]

    def _is_filler(self, index: int, token: str) -> bool:
        """`token`, the word at `index`, is a filler word, or the first word and one
        edit off a filler word of four letters or more."""
        return token in self._fillers or (index == 0 and self._near_filler(token))

    def _near_filler(self, token: str) -> bool:
        """`token` is one edit off a filler word of four letters or more."""
        return any(_one_edit(token, f) for f in self._long_fillers)

    def _fill_slots(
        self, tool: str, user_text: str, user_lemma_set: set, verb_actions: list, model: dict
    ) -> tuple[dict, dict]:
        """Fill the chosen tool's args from the utterance. `verb_actions` holds the
        action of each word (None for a word that is no action verb). Returns (args,
        evidence): evidence holds the actions whose verbs the command explains
        ('actions'), the lemma sets of the alias phrases that filled a slot
        ('phrases'), and whether a number filled a slot ('number')."""
        tinfo = model["tools_index"][tool]
        args: dict = {}
        evidence: dict = {"actions": set(), "phrases": [], "number": False}

        # State slot <- the first action verb of the utterance that is one of its
        # values (action name == enum value by convention; "on"/"off" below mirror
        # that same convention deliberately).
        state_slot = tinfo["state_slot"]
        if state_slot is not None:
            enum = tinfo["slots"][state_slot]["enum"] or []
            action = next((a for a in verb_actions if a in enum), None)
            if action is not None:
                args[state_slot] = action
                evidence["actions"].add(action)

        # Enum slots <- value of the longest alias phrase of that slot in the utterance.
        for slot in tinfo["enum_slots"]:
            best_value = None
            best_phrase = None
            for entry in model["aliases_resolved"]:
                if entry["tool"] != tool or entry["slot"] != slot:
                    continue
                for pl in entry["phrase_lemmas"]:
                    if pl <= user_lemma_set and (best_phrase is None or len(pl) > len(best_phrase)):
                        best_value = entry["value"]
                        best_phrase = pl
            if best_value is not None:
                args[slot] = best_value
                evidence["phrases"].append(best_phrase)

        level_slot = tinfo["level_slot"]
        level_open = level_slot is not None and level_slot not in args
        number_used, level_action = fill_numbers(
            tinfo, user_text, {a for a in verb_actions if a}, args,
            level_on=self._level_on, off_token=self._off_token,
        )
        evidence["number"] = number_used
        if level_action is not None:
            evidence["actions"].add(level_action)
        # An on-verb says nothing the command does not: when a number set the level
        # («включи ночник на пятьдесят») and for a tool with nothing to switch, a
        # scene («включи ночной режим»). An off-verb there contradicts the command.
        if level_open and level_slot in args and level_action is None:
            evidence["actions"].add("on")
        if state_slot is None and level_slot is None:
            evidence["actions"].add("on")

        # Required single-value enum slots left unfilled <- their only value (a real
        # alias-phrase match already proved intent).
        fill_singletons(tinfo, args)
        return args, evidence

    # --- classifier ------------------------------------------------------ #

    def _classify(self, user_text: str, model: dict, reason: str) -> dict:
        """The classifier's command for an utterance the deterministic match found
        no command in (`reason` says why). It acts only on a command class at or
        above the threshold, with every word heard, no number, past the gate, with
        no indicative verb or predicative word, the action verbs agreeing with the
        class, a word left to name the device, no word outside the class
        (_stray_words), an action it may run, and every required slot filled; the
        no-match reason then names the classifier's class and probability."""
        clf = model["classifier"]
        morph = model["morph"]
        tokens = tokenize(user_text)
        user_lemmas = lemmas(user_text, morph.lemma)
        try:
            proba = clf.predict_proba([" ".join(user_lemmas)])[0]
        except Exception as e:  # noqa: BLE001 - prediction must never crash the request
            logger.warning(f"simple-nlu: classifier prediction failed: {e}")
            return _no_match(reason)
        idx = int(proba.argmax())
        name = str(clf.classes_[idx])
        said = f"{reason}; classifier: {name} {float(proba[idx]):.2f}"
        if name == _NONE_CLASS:
            return _no_match(said)
        if float(proba[idx]) < self._classifier_threshold:
            return _no_match(f"{said}, below {self._classifier_threshold}")
        # The unheard word may be «не» or a second request.
        if _UNHEARD in user_text.lower():
            return _no_match(f"{said}, unheard word")
        # A number may be a level, a time or a step, and the class does not tell
        # which: levels are the direct match's (the jev handler leaves them alike).
        if extract_number(user_text) is not None:
            return _no_match(f"{said}, has a number")
        why = self._gate(user_text, model)
        if why is not None:
            return _no_match(f"{said}, gate: {why}")
        # A statement or a wish, not a command: «мы откроем дверь», «можно открыть
        # дверь», «ночной режим работает». The most likely parse decides.
        stated = next((t for t in tokens if "indc" in morph.parses(t)[0].tag
                       or morph.parses(t)[0].tag.POS == "PRED"), None)
        if stated is not None:
            return _no_match(f"{said}, indicative or predicative: {stated}")
        verb_actions = [self._verb_action(tok, model) for tok in tokens]
        if not self._verbs_fit(name, verb_actions, model):
            return _no_match(f"{said}, verbs do not fit the class")
        # The class's device must come from some word, not from the verb alone.
        if all(
            action is not None or self._is_filler(i, tok) or is_number_word(tok, lem)
            for i, (tok, lem, action) in enumerate(zip(tokens, user_lemmas, verb_actions))
        ):
            return _no_match(f"{said}, no device word")
        tool, template = model["classes"][name]
        stray = self._stray_words(template, tokens, user_lemmas, verb_actions, model)
        if stray:
            return _no_match(f"{said}, words outside the class: {', '.join(stray)}")
        state_slot = model["tools_index"][tool]["state_slot"]
        if state_slot is not None and template.get(state_slot) in _DIRECT_ONLY_ACTIONS:
            return _no_match(f"{said}, {template[state_slot]} is left to the direct match")
        tool, args = self._realize(name, user_text, model)
        missing = model["tools_index"][tool]["required"] - set(args)
        if missing:
            return _no_match(f"{said}, missing required slot for {tool}: {', '.join(sorted(missing))}")
        return {"is_command": True, "tool": tool, "args": args, "name": f"classifier/{tool}", "reason": ""}

    def _stray_words(
        self, template: dict, tokens: list[str], user_lemmas: list[str], verb_actions: list, model: dict
    ) -> list[str]:
        """The words of the utterance the class `template` does not account for. Each
        word must be an action verb, a filler word (a misheard first one too), an
        imperative the gate allows, or name the class's device or scene: its lemma
        is a word of that device's alias phrases or close to one, a declined or
        misheard form («ванне» for «ванной», «мониторчики» for «мониторы»). Anything
        else may be a time, a place or another device the class drops («в полночь»,
        «дверь гаража»). The three lists are parallel."""
        values = set(template.values())
        own = set().union(*(s for s, v in model["alias_sets"] if v in values))

        def close(lem: str) -> bool:
            return lem in own or any(
                (len(w) >= 4 and lem[:4] == w[:4]) or (len(w) >= 3 and _one_edit(lem, w))
                for w in own
            )

        return [
            tok for i, (tok, lem, action) in enumerate(zip(tokens, user_lemmas, verb_actions))
            if action is None and not self._is_filler(i, tok) and tok not in _GATE_IMPERATIVES
            and not close(lem)
        ]

    @staticmethod
    def _verbs_fit(name: str, verb_actions: list, model: dict) -> bool:
        """The action verbs of the utterance agree with the class `name`: a class of
        a state value needs a verb of exactly that action, a level class that
        switches on or off needs its verb, and a scene or a level set by a number
        takes an on-verb or no action verb («выключи ночной режим» is no scene)."""
        tool, args = model["classes"][name]
        present = {a for a in verb_actions if a}
        state_slot = model["tools_index"][tool]["state_slot"]
        if state_slot is not None:
            return present == {args[state_slot]}
        level = args.get(_LEVEL_KEY)
        if level in ("on", "off"):
            return present == {level}
        return present <= {"on"}

    def _realize(self, name: str, user_text: str, model: dict) -> tuple[str, dict]:
        """The (tool, args) of the class `name` for this utterance: a level class sets
        its level slot from a number in the utterance, else by its on/off action."""
        tool, template = model["classes"][name]
        args = dict(template)
        level = args.pop(_LEVEL_KEY, None)
        fill_numbers(
            model["tools_index"][tool], user_text, {level} & {"on", "off"}, args,
            level_on=self._level_on, off_token=self._off_token,
        )
        return tool, args

    def _gate(self, user_text: str, model: dict) -> str | None:
        """Why the utterance cannot be one command to one device, or None. Reads the
        surface form, the most likely parse of every word."""
        if "?" in user_text:
            return "question"
        tokens = tokenize(user_text)
        if any(t in _QUESTION_WORDS for t in tokens):
            return "question"
        if "не" in tokens:
            return "negation"
        action_of = model["action_of"]
        parses = [model["morph"].parses(t)[0] for t in tokens]
        verbs = [p for p in parses if p.normal_form in action_of]
        if any("past" in p.tag or p.tag.POS in ("PRTS", "PRTF") for p in verbs):
            return "past/participle"
        dative = any(t in _DATIVE_PRONOUNS for t in tokens)
        if any(p.tag.POS == "INFN" and p.normal_form in action_of
               and (dative or (i > 0 and tokens[i - 1] == "а"))
               for i, p in enumerate(parses)):
            return "infinitive question"
        allowed = self._fillers | _GATE_IMPERATIVES
        if any("impr" in p.tag and p.normal_form not in action_of and t not in allowed
               for t, p in zip(tokens, parses)):
            return "other verb"
        if len({action_of[p.normal_form] for p in verbs}) > 1:
            return "two actions"
        if any(t in _QUANTIFIERS for t in tokens):
            return "quantifier"
        if any(t in _DELAY_WORDS for t in tokens) or any(p.normal_form in _DELAY_LEMMAS for p in parses):
            return "delayed"
        lemma_set = {p.normal_form.replace("ё", "е") for p in parses}
        hits = [(s, v) for s, v in model["alias_sets"] if s <= lemma_set]
        # «и» outside an alias phrase joins a second request («свет и чайник»).
        if "и" in tokens and not any("и" in s for s, _ in hits):
            return "second request"
        # A phrase inside a longer phrase of another device names nothing of its own
        # («свет» in «свет в зале»).
        hits = [(s, v) for s, v in hits if not any(s < s2 and v != v2 for s2, v2 in hits)]
        if len({v for _, v in hits}) > 1:
            return "two devices"
        return None


@register
class SimpleNluProvider(Provider):
    category = "llm"
    id = "simple-nlu"
    label = "Commands (no LLM)"
    ConfigModel = SimpleNluConfig
    uses_http_cloud = False

    def create(self, cfg: SimpleNluConfig, deps: Deps):
        # The catalog as it is now: a catalog change rebuilds this backend.
        commands = deps.commands()
        return SimpleNluBackend(
            aliases_text=commands.aliases,
            actions_text=commands.actions,
            filler_text=commands.filler_words,
            classifier_threshold=cfg.classifier_threshold,
            level_on=commands.level_on,
            off_token=commands.off_token,
            reply_unmatched=cfg.reply_unmatched,
            reply_done=cfg.reply_done,
            reply_error=cfg.reply_error,
        )

    def describe(self, cfg: SimpleNluConfig) -> str:
        return "simple-nlu"

    def vocabulary(self, cfg: SimpleNluConfig, commands) -> list[str]:
        # The closed command vocabulary an offline STT can restrict its grammar to.
        return nlu_vocabulary(commands.aliases, commands.actions, commands.filler_words)
