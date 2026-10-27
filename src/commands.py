"""Helpers over the command catalog (core.commands) shared by the command handlers.

The catalog — device aliases, action verbs, filler words — is household data, not a
setting of one handler: the offline simple-nlu engine and the jev decision model both
read it, route its names to the advertised tools' params and answer in the same
OpenAI-shaped envelope. These helpers live outside src/plugins/ so that neither
handler imports the other.

Heavy deps are imported LAZILY: pymorphy3 only when the morphology is first needed,
so importing this module stays light and must not fail without it.
"""

import functools
import re

from loguru import logger

# --------------------------------------------------------------------------- #
# The catalog's mini-formats.
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


# --------------------------------------------------------------------------- #
# Numbers.
# --------------------------------------------------------------------------- #

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

# Words a number brings along once it set a value («ночник на 50 процентов»): strict
# mode explains them, so a narrowed decoder must be able to emit them.
_NUMBER_PREPOSITIONS = frozenset({"на", "до"})
_PERCENT_LEMMA = "процент"
_PERCENT_FORMS = ("процент", "процента", "процентов")


def nlu_vocabulary(aliases_text: str, actions_text: str, filler_text: str) -> list[str]:
    """Collect the distinct spoken Russian words the NLU can match, for use as a
    Vosk closed-vocabulary grammar. Union of: every word in every alias phrase,
    every action-verb word and its lemma (the infinitive the matcher also takes),
    every filler word, the Russian number words 0..100 and the words a number
    brings along. Phrases are split into individual word tokens (so the decoder can
    recognise any sequence of them); tokens are lowercased. Returns a sorted,
    de-duplicated list. The caller appends the Vosk "[unk]" sentinel."""
    words: set[str] = set()
    for entry in parse_aliases(aliases_text):
        for phrase in entry["phrases"]:
            words.update(re.findall(r"[a-zа-яё0-9]+", phrase.lower()))
    lemmatize = build_lemmatizer()
    for verbs in parse_actions(actions_text).values():
        for verb in verbs:
            words.update(re.findall(r"[a-zа-яё0-9]+", verb.lower()))
            words.update(lemmas(verb, lemmatize))
    # Filler words are in the grammar so a narrowed decoder can emit them instead of
    # "[unk]", which strict mode treats as an unexplained word.
    words.update(re.findall(r"[a-zа-яё0-9]+", (filler_text or "").lower()))
    words.update(_RU_UNITS.keys())
    words.update(_RU_TENS.keys())
    words.update(_NUMBER_PREPOSITIONS)
    words.update(_PERCENT_FORMS)
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


def _is_number_token(token: str) -> bool:
    """A digit run or a Russian number word 0..100 (one normalized token)."""
    return token.isdigit() or token in _RU_UNITS or token in _RU_TENS


def is_number_word(token: str, lemma: str) -> bool:
    """A number or a word a number brings along: «на», «до», a form of «процент»."""
    return _is_number_token(token) or token in _NUMBER_PREPOSITIONS or lemma == _PERCENT_LEMMA


# --------------------------------------------------------------------------- #
# The advertised tools and the arguments a command fills (NO heavy imports).
# --------------------------------------------------------------------------- #

def index_tools(tools: list, action_names: set) -> dict[str, dict]:
    """Index each advertised tool's params: {tool: {slots: {param: {type, enum,
    required}}, required, state_slot, enum_slots, number_slots, level_slot,
    level_type}}. `action_names` are the configured action names; a param whose enum
    is a subset of them is the tool's state slot."""
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
    return tools_index


def resolve_alias(alias: dict, tools_index: dict) -> tuple[str, str] | None:
    """The (tool, slot) a parsed alias routes to: its explicit target, else the first
    tool+param whose enum contains its value; None when no enum contains it."""
    if alias["tool"] and alias["slot"]:
        return alias["tool"], alias["slot"]
    for tname, tinfo in tools_index.items():
        for pname, pinfo in tinfo["slots"].items():
            if pinfo["enum"] and alias["value"] in pinfo["enum"]:
                return tname, pname
    return None


def fill_numbers(
    tinfo: dict, user_text: str, actions: set, args: dict, *, level_on: int, off_token: str
) -> tuple[bool, str | None]:
    """Fill the tool's level slot and number slots into `args` from the utterance.
    `actions` are the action names the utterance carries; an on-verb sets a numeric
    level slot to `level_on`, an off-verb sets it to 0 (a string one to `off_token`).
    Returns whether a number filled a slot, and the action ("on"/"off") that filled
    the level slot or None."""
    number_used = False
    level_action = None
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
            number_used = True
        elif "off" in actions:
            args[level_slot] = 0 if level_type == "number" else off_token
            level_action = "off"
        elif "on" in actions and level_type == "number":
            args[level_slot] = level_on
            level_action = "on"
        # string level slot + on-verb + no number -> leave UNFILLED, so the
        # required-slot validation correctly no-matches it.

    # Number slots <- first number in the utterance.
    for slot in tinfo["number_slots"]:
        if slot in args:
            continue
        num = extract_number(user_text)
        if num is not None:
            args[slot] = num
            number_used = True
    return number_used, level_action


def fill_singletons(tinfo: dict, args: dict) -> None:
    """Set each required enum slot that is still unset and has one value to it."""
    for slot, info in tinfo["slots"].items():
        if slot in args or not info["required"]:
            continue
        enum = info["enum"]
        if enum and len(enum) == 1:
            args[slot] = enum[0]


# --------------------------------------------------------------------------- #
# The reply envelope and the rounds of the LlmStage loop.
# --------------------------------------------------------------------------- #

def envelope(content: str, tool_calls, model: str) -> dict:
    """The OpenAI/Groq-shaped reply the LlmStage loop expects (see src/llm.py)."""
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
        ],
        "usage": {"total_tokens": 0},
        "model": model,
    }


def last_user_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


def has_tool_result(messages: list) -> bool:
    return any(msg.get("role") == "tool" for msg in messages)


def round2_reply(messages: list, reply_done: str, reply_error: str) -> str:
    """After a tool ran, the confirmation — or `reply_error` if the last tool result
    looks like an error string (ToolHub returns errors as text, e.g. 'error calling
    set_light: ...')."""
    last_tool = None
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            last_tool = msg
            break
    content = (last_tool or {}).get("content") or ""
    low = content.strip().lower()
    if low.startswith("error") or "error calling" in low:
        return reply_error
    return reply_done


# --------------------------------------------------------------------------- #
# Morphology.
# --------------------------------------------------------------------------- #

# Logged-once flags for missing heavy deps (module-level so the warning fires once
# per process rather than per backend instance).
_warned_missing: set[str] = set()


def warn_missing(dep: str) -> None:
    if dep not in _warned_missing:
        _warned_missing.add(dep)
        logger.warning(f"commands: optional dependency '{dep}' unavailable; degrading gracefully")


def tokenize(text: str) -> list[str]:
    """Lowercase, fold ё->е, and tokenize on non-letters (Cyrillic + Latin +
    digits)."""
    low = (text or "").lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", low)


def lemmas(text: str, lemmatize) -> list[str]:
    """The lemma of every token of `text`, lowercased with ё folded to е."""
    return [lemmatize(tok).lower().replace("ё", "е") for tok in tokenize(text)]


class _Morph:
    """pymorphy3 analysis of one lowercased token, memoized: every parse, and the
    lemma (the first parse's normal form)."""

    def __init__(self, analyzer):
        self._analyzer = analyzer
        self._cache: dict[str, list] = {}

    def parses(self, token: str) -> list:
        hit = self._cache.get(token)
        if hit is None:
            hit = self._cache[token] = self._analyzer.parse(token)
        return hit

    def lemma(self, word: str) -> str:
        return self.parses(word.lower())[0].normal_form


@functools.cache
def morphology() -> _Morph | None:
    """The process-wide morphology, or None when pymorphy3 is unavailable."""
    try:
        import pymorphy3
    except ImportError:
        warn_missing("pymorphy3")
        return None
    return _Morph(pymorphy3.MorphAnalyzer())


def build_lemmatizer():
    """A memoized pymorphy3 lemmatizer of ONE lowercased token, or an
    identity-lowercase fallback if pymorphy3 is unavailable."""
    morph = morphology()
    return morph.lemma if morph is not None else (lambda w: w.lower())


def _word_set(text: str, lemmatize) -> frozenset:
    """The lowercased words of `text`, lemmatized when `lemmatize` is given."""
    words = [w.lower() for w in re.findall(r"[а-яёa-zA-ZА-Я0-9]+", text)]
    if lemmatize is not None:
        words = [lemmatize(w) for w in words]
    return frozenset(words)


def aliases_from_facts(facts: list[str], catalog_entries: list[dict], lemmatize=None) -> list[dict]:
    """Alias entries derived from the assistant's saved memory facts.

    A fact of the form «X — это Y» records household names: X and Y are spoken
    names of the same thing. The names route to a device when the fact anchors
    them to one: exactly one latin id token in the fact (night_lamp —
    resolve_alias auto-routes it), or a name matching a catalog alias phrase by
    word sets (lemmatized when `lemmatize` is given). Facts naming nothing or
    several distinct devices, or no new names contribute nothing. The entries
    parse like parse_aliases output (auto-route form: tool and slot are None)."""
    catalog_phrases = {
        phrase.casefold() for entry in catalog_entries for phrase in entry["phrases"]
    }
    entries: list[dict] = []
    # Phrases already emitted by earlier facts, so two facts about one device do
    # not repeat a name in the criteria text.
    seen_phrases: set[str] = set()
    for fact in facts:
        # Candidate names, in order, catalog duplicates included: a name equal to
        # a catalog phrase adds no criteria text but still anchors the fact.
        names: list[str] = []
        for segment in re.split(r"[,;]", fact or ""):
            segment = segment.strip()
            if segment.endswith("."):
                segment = segment[:-1]
            segment = segment.strip()
            if not segment:
                continue
            m = re.match(r"^(.+?)\s+[—–-]+\s+это\s+(.+)$", segment, re.IGNORECASE)
            if m:
                sides = (m.group(1), m.group(2))
            else:
                m = re.match(r"^(.+?)\s+[—–-]+\s+(.+)$", segment)
                if m and re.fullmatch(r"[a-z][a-z0-9_]{2,}", m.group(2).strip()):
                    sides = (m.group(1),)
                else:
                    sides = ()
            for side in sides:
                name = side.strip().lower()
                if name and name not in names:
                    names.append(name)
        if not names:
            logger.debug(f"memory fact yields no names, skipped: {fact}")
            continue
        # The fact's latin id tokens: the device values it may anchor to.
        values = set(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", fact or ""))
        if len(values) > 1:
            logger.debug(f"memory fact names several devices, skipped: {fact}")
            continue
        if values:
            value = values.pop()
        else:
            # No id in the fact: anchor it by a name matching one catalog alias
            # phrase by word sets (either a subset of the other).
            matched = []
            for entry in catalog_entries:
                if any(
                    _word_set(phrase, lemmatize) <= _word_set(name, lemmatize)
                    or _word_set(name, lemmatize) <= _word_set(phrase, lemmatize)
                    for phrase in entry["phrases"] for name in names
                ):
                    matched.append(entry)
            if len(matched) != 1:
                logger.debug(f"memory fact names no device, skipped: {fact}")
                continue
            value = matched[0]["value"]
        # Names the catalog or an earlier fact already speaks add no criteria
        # text; a fact whose names are all known contributes nothing.
        phrases = []
        for name in names:
            folded = name.casefold()
            if folded not in catalog_phrases and folded not in seen_phrases:
                seen_phrases.add(folded)
                phrases.append(name)
        if phrases:
            entries.append({"phrases": phrases, "value": value, "tool": None, "slot": None})
    return entries
