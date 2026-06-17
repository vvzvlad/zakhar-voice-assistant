"""Offline, LLM-free NLU 'LLM' provider — intent classification + slot filling.

It slots into the pipeline's `llm` stage like any other provider, but uses NO
LLM and NO network: an intent+slot NLU built on pymorphy3 + scikit-learn +
rapidfuzz. A hybrid of deterministic phrase/verb matching and a small TF-IDF +
LogisticRegression intent classifier turns a Russian utterance into a smart-home
tool call. The OpenAI/Groq-shaped envelope and the LlmStage agentic loop are
exactly the same as for the cloud LLM providers, so the backend emits
`tool_calls` to drive the tools and, on the next round (tool results present),
speaks a short confirmation.

Design notes:
- Heavy deps (pymorphy3, scikit-learn, rapidfuzz) are imported LAZILY inside the
  training thread / fuzzy helper, never at module top level: importing this
  module for the registry stays light and must not fail if those libs are absent.
  Each missing dep degrades gracefully (see _train / _fuzzy_ratio).
- The backend is STATELESS across rounds: nothing per-request is stashed on self
  (it is shared across concurrent speakers). Round 2 is reconstructed purely from
  the `messages` list (presence of a role=="tool" message).
- The model (lemmatizer + indices + classifier) is built once per distinct tool
  catalog and cached, keyed by a fingerprint of the advertised tools.
"""

import asyncio
import hashlib
import json
import re

from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import Deps, Provider, register
from src.plugins.llm.base import LlmBackend

# Default verbs are real (not commented) so the provider works out of the box.
_DEFAULT_ACTIONS = (
    "on = включи, включай, зажги, зажги свет, вруби, врубай, запусти\n"
    "off = выключи, выключай, погаси, выруби, отключи, останови"
)

# A self-documenting, all-commented example so the aliases field matches nothing
# until the operator configures it for their actual entity/scene ids.
_DEFAULT_ALIASES = (
    "# One mapping per line. LHS: comma-separated Russian synonyms; RHS: the exact\n"
    "# ARG VALUE the tool expects (an entity/scene id), auto-routed to the tool+param\n"
    "# whose enum contains it. Use 'tool.slot:value' to target explicitly when the\n"
    "# MCP schema does not enumerate the value. Lines starting with # are ignored.\n"
    "# свет в зале, люстра = bright_room_light\n"
    "# ночь, спать = night\n"
    "# яркость, свет.brightness:80 = set_light.brightness:80"
)


class SimpleNluConfig(BaseModel):
    # --- NLU configuration (baked into the backend at create(); changing any of
    # these triggers a backend rebuild via reconfig, which is correct) ---
    aliases: str = Field(
        _DEFAULT_ALIASES,
        title="Aliases (Russian phrase = value)",
        description=(
            "Map Russian phrases to the exact tool arg value. One per line: "
            "'phrase1, phrase2 = value' (auto-route by enum) or "
            "'phrase = tool.slot:value' (explicit target). '#' lines are ignored."
        ),
        json_schema_extra={"widget": "textarea"},
    )
    actions: str = Field(
        _DEFAULT_ACTIONS,
        title="Action verbs",
        description=(
            "Map a canonical action name to its Russian trigger verbs, one per "
            "line: 'action = verb1, verb2, ...'. When the action name equals an "
            "enum value of a tool's state-like param, it fills that slot."
        ),
        json_schema_extra={"widget": "textarea"},
    )
    min_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        title="Classifier confidence threshold",
        description="Below this max class probability the classifier result is ignored.",
    )
    fuzzy_threshold: int = Field(
        82,
        ge=0,
        le=100,
        title="Fuzzy match threshold",
        description="RapidFuzz token_set_ratio score (0-100) for fuzzy phrase matching.",
    )
    level_on: int = Field(
        100,
        ge=0,
        title="On level",
        description=(
            "Value an on-verb assigns to a numeric level slot with no on/off enum "
            "(e.g. dimmer brightness). An off-verb assigns 0."
        ),
    )
    off_token: str = Field(
        "off",
        title="Off token (string level slot)",
        description=(
            "Value an off-verb assigns to a required string level slot with no enum "
            "(e.g. climate temperature)."
        ),
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
    # reconfig._LLM_LIVE_LEAVES). Kept here so this provider is a complete drop-in.
    # max_tool_rounds minimum is 2: the tool round + the final reply round both run. ---
    max_tool_rounds: int = Field(2, ge=2)
    reply_empty: str = "Я тебя не расслышал, повтори."
    reply_empty_after_tools: str = "Готово."
    reply_rate_limit: str = "Лимит запросов исчерпан. Попробуй ещё раз чуть позже."
    reply_error: str = "Что-то сломалось, попробуй ещё раз попозже."

    # --- Panel UI state (no runtime effect): ids of MCP sources the operator hid in
    # the NLU catalog editor's Sources bar. Persisted so the selection survives a
    # reload; the NLU itself always uses every source's aliases/actions. Marked
    # `hidden` so the generic SchemaForm never renders it, and whitelisted as a
    # "live" leaf in reconfig so toggling it never rebuilds the NLU model. ---
    hidden_sources: list[str] = Field(
        default_factory=list,
        title="Hidden MCP sources (panel UI state)",
        description="Source ids hidden in the catalog editor's Sources bar. Panel-only; ignored at runtime.",
        json_schema_extra={"hidden": True},
    )


# --------------------------------------------------------------------------- #
# Pure helper functions (module-level, unit-testable, NO heavy imports).
# --------------------------------------------------------------------------- #

def parse_aliases(text: str) -> list[dict]:
    """Parse the alias mini-format into a list of routing entries.

    Each non-blank, non-'#' line is `phrase1, phrase2, ... = RHS` where RHS is
    either a bare `value` (auto-route: locate the tool+param whose enum contains
    it) or `tool.slot:value` (explicit target). Returns a list of dicts:
    {"phrases": [str, ...], "value": str, "tool": str|None, "slot": str|None}.
    Phrases are lowercased; lines without a usable LHS/RHS are skipped.
    """
    entries: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        phrases = [p.strip().lower() for p in lhs.split(",") if p.strip()]
        rhs = rhs.strip()
        if not phrases or not rhs:
            continue
        tool: str | None = None
        slot: str | None = None
        value = rhs
        # Explicit form: tool.slot:value. The ':' separates target from value;
        # the '.' before it separates tool from slot. Be lenient about spaces.
        if ":" in rhs:
            target, val = rhs.split(":", 1)
            target = target.strip()
            val = val.strip()
            if "." in target and val:
                tool, slot = target.split(".", 1)
                tool = tool.strip()
                slot = slot.strip()
                value = val
        entries.append({"phrases": phrases, "value": value, "tool": tool, "slot": slot})
    return entries


def parse_actions(text: str) -> dict[str, list[str]]:
    """Parse the action mini-format `action = verb1, verb2, ...` per line.

    Returns {action_name: [verb, ...]} with action names and verbs lowercased.
    Blank/'#' lines and lines without a usable action or verbs are skipped.
    """
    actions: dict[str, list[str]] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, verbs_str = line.split("=", 1)
        name = name.strip().lower()
        verbs = [v.strip().lower() for v in verbs_str.split(",") if v.strip()]
        if not name or not verbs:
            continue
        actions[name] = verbs
    return actions


# Russian number words 0..100 for spoken-number slots. Units/teens and tens are
# composed (e.g. "двадцать два" -> 22). Kept compact and bounded.
_RU_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19,
}
_RU_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100,
}


def nlu_vocabulary(aliases_text: str, actions_text: str) -> list[str]:
    """Collect the distinct spoken Russian words the NLU can match, for use as a
    Vosk closed-vocabulary grammar. Union of: every word in every alias phrase,
    every action-verb word, and the Russian number words 0..100. Phrases are
    split into individual word tokens (so the decoder can recognise any sequence
    of them); tokens are lowercased. Returns a sorted, de-duplicated list. The
    caller appends the Vosk "[unk]" sentinel."""
    words: set[str] = set()
    for entry in parse_aliases(aliases_text):
        for phrase in entry["phrases"]:
            words.update(re.findall(r"[a-zа-яё0-9]+", phrase.lower()))
    for verbs in parse_actions(actions_text).values():
        for verb in verbs:
            words.update(re.findall(r"[a-zа-яё0-9]+", verb.lower()))
    words.update(_RU_UNITS.keys())
    words.update(_RU_TENS.keys())
    words.discard("")
    return sorted(words)


def extract_number(text: str) -> int | None:
    """Return the first number in `text`, digits first, else Russian words 0..100.

    Digits are matched by regex. Failing that, Russian number words are scanned in
    order; adjacent tens+units compose (e.g. "двадцать два" -> 22). Returns the
    first number found, or None.
    """
    if not text:
        return None
    m = re.search(r"\d+", text)
    if m:
        return int(m.group())
    tokens = re.findall(r"[а-яё]+", text.lower())
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _RU_TENS:
            base = _RU_TENS[tok]
            # Compose tens + a trailing unit only for proper tens (20..90).
            if base < 100 and i + 1 < len(tokens) and tokens[i + 1] in _RU_UNITS:
                unit = _RU_UNITS[tokens[i + 1]]
                if 1 <= unit <= 9:
                    return base + unit
            return base
        if tok in _RU_UNITS:
            return _RU_UNITS[tok]
        i += 1
    return None


# --------------------------------------------------------------------------- #
# Backend.
# --------------------------------------------------------------------------- #

# Logged-once flags for missing heavy deps (module-level so the warning fires once
# per process rather than per backend instance).
_warned_missing: set[str] = set()


def _warn_missing(dep: str) -> None:
    if dep not in _warned_missing:
        _warned_missing.add(dep)
        logger.warning(f"simple-nlu: optional dependency '{dep}' unavailable; degrading gracefully")


class SimpleNluBackend(LlmBackend):
    """LlmBackend contract WITHOUT any LLM: deterministic + TF-IDF intent NLU.

    Emits OpenAI-shaped tool_calls to drive the smart-home tools. The model
    (lemmatizer, tool index, deterministic phrase index, classifier) is built
    once per distinct tool catalog (keyed by a fingerprint) and cached.
    """

    def __init__(
        self,
        *,
        aliases_text: str,
        actions_text: str,
        min_confidence: float,
        fuzzy_threshold: int,
        level_on: int,
        off_token: str,
        reply_unmatched: str,
        reply_done: str,
        reply_error: str,
        lemmatizer=None,
    ):
        self._aliases = parse_aliases(aliases_text)
        self._actions = parse_actions(actions_text)
        self._min_confidence = min_confidence
        self._fuzzy_threshold = fuzzy_threshold
        self._level_on = level_on
        self._off_token = off_token
        self._reply_unmatched = reply_unmatched
        self._reply_done = reply_done
        self._reply_error = reply_error
        # Injectable lemmatizer (str -> str, lemmatize ONE lowercased token). When
        # None, a pymorphy3-based one is built lazily on first train.
        self._lemmatizer = lemmatizer
        # Model cache, rebuilt when the advertised tools change.
        self._lock = asyncio.Lock()
        self._fingerprint: str | None = None
        self._model: dict | None = None

    # --- public contract -------------------------------------------------- #

    async def complete(self, messages: list, tools: list | None) -> dict:
        fingerprint = self._fingerprint_tools(tools or [])
        # Rebuild the model only when the tool catalog changed. Training is
        # blocking (pymorphy3 + sklearn fit) so it runs in a worker thread.
        async with self._lock:
            if fingerprint != self._fingerprint:
                self._model = await asyncio.to_thread(self._train, tools or [])
                self._fingerprint = fingerprint
            model = self._model

        # ROUND 2: tool results are present -> speak a confirmation (or error).
        if self._has_tool_result(messages):
            content = self._round2_reply(messages)
            return self._envelope(content, None, "none")

        # ROUND 1: classify the user utterance into a tool call (or no-match).
        user_text = self._last_user_text(messages)
        intent = self._detect(user_text, model)
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
        return self._envelope(self._reply_unmatched, None, "none")

    def _envelope(self, content: str, tool_calls, intent_name: str) -> dict:
        # OpenAI/Groq-shaped envelope the LlmStage loop expects (see src/llm.py).
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
            ],
            "usage": {"total_tokens": 0},
            "model": f"simple-nlu/{intent_name or 'none'}",
        }

    # --- round-2 + message helpers --------------------------------------- #

    @staticmethod
    def _last_user_text(messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    @staticmethod
    def _has_tool_result(messages: list) -> bool:
        return any(msg.get("role") == "tool" for msg in messages)

    def _round2_reply(self, messages: list) -> str:
        """After a tool ran, speak a confirmation — or the error fallback if the
        last tool result looks like an error string (ToolHub returns errors as
        text, e.g. 'error calling set_light: ...')."""
        last_tool = None
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                last_tool = msg
                break
        content = (last_tool or {}).get("content") or ""
        low = content.strip().lower()
        if low.startswith("error") or "error calling" in low:
            return self._reply_error
        return self._reply_done

    # --- tokenization / lemmatization ------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> list[str]:
        """Lowercase, fold ё->е, and tokenize on non-letters (Cyrillic + Latin +
        digits)."""
        low = (text or "").lower().replace("ё", "е")
        return re.findall(r"[a-zа-я0-9]+", low)

    def _lemmas(self, text: str, lemmatizer=None) -> list[str]:
        lem = lemmatizer or self._lemmatizer or (lambda w: w.lower())
        out = []
        for tok in self._normalize(text):
            out.append(lem(tok).lower().replace("ё", "е"))
        return out

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

    def _build_lemmatizer(self):
        """Resolve a lemmatizer: the injected one, else a lazy pymorphy3 one, else
        an identity-lowercase fallback if pymorphy3 is unavailable."""
        if self._lemmatizer is not None:
            return self._lemmatizer
        try:
            import pymorphy3

            morph = pymorphy3.MorphAnalyzer()
            cache: dict[str, str] = {}

            def lemmatize(word: str) -> str:
                w = word.lower()
                hit = cache.get(w)
                if hit is not None:
                    return hit
                normal = morph.parse(w)[0].normal_form
                cache[w] = normal
                return normal

            return lemmatize
        except ImportError:
            _warn_missing("pymorphy3")
            return lambda w: w.lower()

    def _train(self, tools: list) -> dict:
        """Build the immutable model dict from the advertised tools.

        Indexes each tool's params, resolves aliases to (tool, slot, value),
        builds a deterministic phrase/verb index and (when sklearn + >=2 trainable
        tools are available) a TF-IDF + LogisticRegression classifier.
        """
        lemmatizer = self._build_lemmatizer()
        action_names = set(self._actions.keys())

        # 1) Index tools: per-tool params with {type, enum, required}.
        tools_index: dict[str, dict] = {}
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            required = set(params.get("required") or [])
            slots: dict[str, dict] = {}
            for pname, pschema in props.items():
                pschema = pschema or {}
                enum = pschema.get("enum")
                slots[pname] = {
                    "type": pschema.get("type"),
                    "enum": list(enum) if enum else None,
                    "required": pname in required,
                }
            # State slot: enum non-empty AND a subset of the action-name set.
            state_slot = None
            for pname, info in slots.items():
                enum = info["enum"]
                if enum and set(enum) <= action_names:
                    state_slot = pname
                    break
            enum_slots = [p for p, i in slots.items() if i["enum"] and p != state_slot]
            number_slots = [p for p, i in slots.items() if i["type"] in ("integer", "number")]
            # Level slot: a required, NON-state, NON-enum param a verb can set to a
            # numeric/off value (e.g. dimmer brightness, climate temperature). Prefer
            # a numeric one; else accept a lone required string slot.
            level_candidates = [
                p for p, i in slots.items()
                if i["required"] and p != state_slot and not i["enum"]
            ]
            level_slot = None
            level_type = None
            numeric_candidates = [
                p for p in level_candidates if slots[p]["type"] in ("integer", "number")
            ]
            if numeric_candidates:
                level_slot = numeric_candidates[0]
                level_type = "number"
            elif len(level_candidates) == 1 and slots[level_candidates[0]]["type"] == "string":
                level_slot = level_candidates[0]
                level_type = "string"
            tools_index[name] = {
                "slots": slots,
                "required": required,
                "state_slot": state_slot,
                "enum_slots": enum_slots,
                "number_slots": number_slots,
                "level_slot": level_slot,
                "level_type": level_type,
            }

        # 2) Resolve aliases to (tool, slot, value). Auto form locates the
        # tool+param whose enum contains the value; unroutable aliases are skipped.
        aliases_resolved: list[dict] = []
        for alias in self._aliases:
            tool = alias["tool"]
            slot = alias["slot"]
            value = alias["value"]
            if tool and slot:
                resolved_tool, resolved_slot = tool, slot
            else:
                resolved_tool, resolved_slot = None, None
                for tname, tinfo in tools_index.items():
                    for pname, pinfo in tinfo["slots"].items():
                        if pinfo["enum"] and value in pinfo["enum"]:
                            resolved_tool, resolved_slot = tname, pname
                            break
                    if resolved_tool:
                        break
                if resolved_tool is None:
                    logger.warning(
                        f"simple-nlu: alias value {value!r} not found in any tool enum; skipped"
                    )
                    continue
            phrase_lemmas = [frozenset(self._lemmas(p, lemmatizer)) for p in alias["phrases"]]
            aliases_resolved.append({
                "tool": resolved_tool,
                "slot": resolved_slot,
                "value": value,
                "phrases": alias["phrases"],
                "phrase_lemmas": [pl for pl in phrase_lemmas if pl],
            })

        # 3) Action verb lemma -> action name (for state-slot filling).
        action_verb_lemmas: dict[frozenset, str] = {}
        for action, verbs in self._actions.items():
            for verb in verbs:
                lemset = frozenset(self._lemmas(verb, lemmatizer))
                if lemset:
                    action_verb_lemmas[lemset] = action

        # 4) Per-tool applicable verbs (those whose action is in the tool's state
        # enum; if a tool has no state slot, ALL verbs are bare triggers).
        per_tool_verbs: dict[str, list[frozenset]] = {}
        for tname, tinfo in tools_index.items():
            state_slot = tinfo["state_slot"]
            applicable = []
            for lemset, action in action_verb_lemmas.items():
                if state_slot is None:
                    applicable.append(lemset)
                else:
                    enum = tinfo["slots"][state_slot]["enum"] or []
                    if action in enum:
                        applicable.append(lemset)
            per_tool_verbs[tname] = applicable

        # 5) Per-tool alias phrase lemma-sets (for the deterministic match).
        per_tool_alias_phrases: dict[str, list[frozenset]] = {}
        for entry in aliases_resolved:
            per_tool_alias_phrases.setdefault(entry["tool"], []).extend(entry["phrase_lemmas"])

        # 6) Classifier (optional). Generate synthetic utterances per tool from its
        # applicable verbs x alias phrases, then fit TF-IDF + LogisticRegression.
        classifier = None
        train_texts: list[str] = []
        train_labels: list[str] = []
        for tname in tools_index:
            verbs = per_tool_verbs.get(tname, [])
            phrases = [e for e in aliases_resolved if e["tool"] == tname]
            verb_strs = [" ".join(sorted(v)) for v in verbs] or [""]
            phrase_strs = []
            for e in phrases:
                for pl in e["phrase_lemmas"]:
                    phrase_strs.append(" ".join(sorted(pl)))
            for p in phrase_strs:
                for v in verb_strs:
                    if v:
                        train_texts.append(f"{v} {p}".strip())
                        train_labels.append(tname)
                        train_texts.append(f"{p} {v}".strip())
                        train_labels.append(tname)
                    train_texts.append(p)
                    train_labels.append(tname)
            # Bare-verb utterances when a tool has no alias phrases (verb-only tools).
            if not phrase_strs:
                for v in verb_strs:
                    if v:
                        train_texts.append(v)
                        train_labels.append(tname)

        if len(set(train_labels)) >= 2 and train_texts:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline

                classifier = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                    ("clf", LogisticRegression(max_iter=1000)),
                ])
                classifier.fit(train_texts, train_labels)
            except ImportError:
                _warn_missing("scikit-learn")
                classifier = None
            except Exception as e:  # noqa: BLE001 - never let training crash the request
                logger.warning(f"simple-nlu: classifier training failed: {e}")
                classifier = None

        return {
            "lemmatizer": lemmatizer,
            "tools_index": tools_index,
            "aliases_resolved": aliases_resolved,
            "per_tool_verbs": per_tool_verbs,
            "per_tool_alias_phrases": per_tool_alias_phrases,
            "action_verb_lemmas": action_verb_lemmas,
            "classifier": classifier,
        }

    # --- fuzzy helper ---------------------------------------------------- #

    def _fuzzy_ratio(self, a: str, b: str) -> float:
        """token_set_ratio via rapidfuzz; 0.0 when rapidfuzz is unavailable (the
        caller then relies on exact lemma-subset containment only)."""
        try:
            from rapidfuzz import fuzz

            return fuzz.token_set_ratio(a, b)
        except ImportError:
            _warn_missing("rapidfuzz")
            return 0.0
        except Exception as e:  # noqa: BLE001 - a fuzzy runtime error must never crash inference
            logger.warning(f"simple-nlu: fuzzy match failed: {e}")
            return 0.0

    # --- detection ------------------------------------------------------- #

    def _detect(self, user_text: str, model: dict | None) -> dict:
        """Classify `user_text` into {is_command, tool, args, name}."""
        no_match = {"is_command": False, "tool": None, "args": {}, "name": "none"}
        if not model or not user_text:
            return no_match
        lemmatizer = model["lemmatizer"]
        tools_index = model["tools_index"]
        user_lemmas = self._lemmas(user_text, lemmatizer)
        user_lemma_set = set(user_lemmas)
        user_lemma_str = " ".join(user_lemmas)

        # 1) DETERMINISTIC: a tool whose alias phrase is fully present (subset or
        # fuzzy) and (if it has a state slot) an applicable verb is present. Prefer
        # the longest matched alias phrase (most specific).
        best_tool = None
        best_len = -1
        for tname, tinfo in tools_index.items():
            phrase_sets = model["per_tool_alias_phrases"].get(tname, [])
            verbs = model["per_tool_verbs"].get(tname, [])
            has_verb = any(v <= user_lemma_set for v in verbs) if verbs else False
            matched_phrase_len = -1
            for pl in phrase_sets:
                if pl <= user_lemma_set:
                    matched_phrase_len = max(matched_phrase_len, len(pl))
                elif self._fuzzy_threshold and self._fuzzy_ratio(
                    " ".join(sorted(pl)), user_lemma_str
                ) >= self._fuzzy_threshold:
                    matched_phrase_len = max(matched_phrase_len, len(pl))
            if matched_phrase_len < 0:
                continue
            # If the tool has a state slot, require an applicable verb.
            if tinfo["state_slot"] is not None and not has_verb:
                continue
            if matched_phrase_len > best_len:
                best_len = matched_phrase_len
                best_tool = tname

        chosen = best_tool
        from_classifier = False

        # 2) CLASSIFIER fallback.
        if chosen is None and model.get("classifier") is not None:
            try:
                clf = model["classifier"]
                proba = clf.predict_proba([user_lemma_str])[0]
                classes = clf.classes_
                idx = int(proba.argmax())
                if float(proba[idx]) >= self._min_confidence:
                    chosen = str(classes[idx])
                    from_classifier = True
            except Exception as e:  # noqa: BLE001 - prediction must never crash the request
                logger.warning(f"simple-nlu: classifier prediction failed: {e}")

        if chosen is None:
            return no_match

        # 3) FILL SLOTS for the chosen tool. The deterministic path already proved a
        # real alias-phrase match, so it may auto-fill required single-value enums;
        # the classifier path is evidence-free, so we DON'T auto-fill and we require
        # below that the predicted tool's enum slots were actually grounded in text.
        tinfo = tools_index[chosen]
        args = self._fill_slots(
            chosen, user_text, user_lemma_set, model, auto_fill_singletons=not from_classifier
        )
        # Classifier guard: a predicted tool with enum slots must have at least one
        # filled from the utterance, else it is an unsupported guess -> no command.
        if from_classifier and tinfo["enum_slots"]:
            if not any(slot in args for slot in tinfo["enum_slots"]):
                return no_match
        # Required-slot validation is the main false-positive guard.
        if not tinfo["required"] <= set(args.keys()):
            return no_match
        return {"is_command": True, "tool": chosen, "args": args, "name": chosen}

    def _fill_slots(
        self,
        tool: str,
        user_text: str,
        user_lemma_set: set,
        model: dict,
        *,
        auto_fill_singletons: bool = True,
    ) -> dict:
        tinfo = model["tools_index"][tool]
        args: dict = {}

        # The action names present in the utterance (action name == enum value by
        # convention; "on"/"off" below mirror that same convention deliberately).
        present_actions = {
            action for lemset, action in model["action_verb_lemmas"].items()
            if lemset <= user_lemma_set
        }

        # State slot <- the action name of a verb present in the utterance.
        state_slot = tinfo["state_slot"]
        if state_slot is not None:
            enum = tinfo["slots"][state_slot]["enum"] or []
            for lemset, action in model["action_verb_lemmas"].items():
                if lemset <= user_lemma_set and action in enum:
                    args[state_slot] = action
                    break

        # Enum slots <- value of the best-matching alias phrase for that slot.
        for slot in tinfo["enum_slots"]:
            best_value = None
            best_len = -1
            for entry in model["aliases_resolved"]:
                if entry["tool"] != tool or entry["slot"] != slot:
                    continue
                for pl in entry["phrase_lemmas"]:
                    matched = pl <= user_lemma_set
                    if not matched and self._fuzzy_threshold:
                        matched = self._fuzzy_ratio(
                            " ".join(sorted(pl)), " ".join(sorted(user_lemma_set))
                        ) >= self._fuzzy_threshold
                    if matched and len(pl) > best_len:
                        best_len = len(pl)
                        best_value = entry["value"]
            if best_value is not None:
                args[slot] = best_value

        # Level slot <- an explicit number, else an on/off verb maps to a level.
        # Done BEFORE number_slots so the level slot wins (the loop below skips
        # slots already in args). Hardcoded "on"/"off" mirror the existing
        # convention (action name == enum value).
        level_slot = tinfo["level_slot"]
        level_type = tinfo["level_type"]
        if level_slot is not None and level_slot not in args:
            num = extract_number(user_text)
            if num is not None:
                args[level_slot] = num if level_type == "number" else str(num)
            elif "off" in present_actions:
                args[level_slot] = 0 if level_type == "number" else self._off_token
            elif "on" in present_actions and level_type == "number":
                args[level_slot] = self._level_on
            # string level slot + on-verb + no number -> leave UNFILLED, so the
            # required-slot validation correctly no-matches it.

        # Number slots <- first number in the utterance.
        for slot in tinfo["number_slots"]:
            if slot in args:
                continue
            num = extract_number(user_text)
            if num is not None:
                args[slot] = num

        # Required single-value enum slots left unfilled <- their only value. Only on
        # the deterministic path (a real match already proved intent); skipped on the
        # classifier path so an evidence-free prediction can't satisfy required slots.
        if auto_fill_singletons:
            for slot, info in tinfo["slots"].items():
                if slot in args or not info["required"]:
                    continue
                enum = info["enum"]
                if enum and len(enum) == 1:
                    args[slot] = enum[0]

        return args


@register
class SimpleNluProvider(Provider):
    category = "llm"
    id = "simple-nlu"
    label = "Simple NLU (no LLM)"
    ConfigModel = SimpleNluConfig
    uses_http_cloud = False

    def create(self, cfg: SimpleNluConfig, deps: Deps):
        return SimpleNluBackend(
            aliases_text=cfg.aliases,
            actions_text=cfg.actions,
            min_confidence=cfg.min_confidence,
            fuzzy_threshold=cfg.fuzzy_threshold,
            level_on=cfg.level_on,
            off_token=cfg.off_token,
            reply_unmatched=cfg.reply_unmatched,
            reply_done=cfg.reply_done,
            reply_error=cfg.reply_error,
        )

    def describe(self, cfg: SimpleNluConfig) -> str:
        return "simple-nlu"

    def vocabulary(self, cfg: SimpleNluConfig) -> list[str]:
        # The closed command vocabulary an offline STT can restrict its grammar to.
        return nlu_vocabulary(cfg.aliases, cfg.actions)
