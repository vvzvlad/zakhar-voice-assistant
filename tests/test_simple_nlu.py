"""Tests for the offline Simple NLU 'LLM' provider.

The provider uses NO LLM and NO network. These tests run it with real pymorphy3 and
scikit-learn on a small fake catalog and on the production one; each backend
trains once per module. They cover registration, the pure helpers, the
OpenAI-shaped envelope across both rounds, one end-to-end pass through the real
LlmStage loop, and strict mode (a later chain exists). The envelope's model names
the path that acted: "simple-nlu/<tool>" for the deterministic match,
"simple-nlu/classifier/<tool>" for the classifier.
"""

import functools
import json
from pathlib import Path

import httpx
import pytest

import src.plugins  # noqa: F401  triggers @register on all providers
from src.llm import LlmRequest, LlmStage
from src.plugins.base import Deps, get_provider
from src.plugins.llm.simple_nlu import (
    SimpleNluBackend,
    SimpleNluConfig,
    SimpleNluProvider,
    extract_number,
    nlu_vocabulary,
    parse_actions,
    parse_aliases,
)
from src.stage_errors import Declined


def _deps():
    return Deps(
        http_cloud=httpx.AsyncClient(),
        http_local=httpx.AsyncClient(),
        tts_timeout=30,
    )


# A fake tool catalog mirroring the real smart-home tools. set_light has an entity
# enum and a state enum (a subset of the action names on/off); set_scene a scene
# enum. set_dimmer has a device enum + a required integer 'brightness' (a numeric
# level slot); set_climate a device enum + a required string 'temperature' (a
# string level slot).
FAKE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_light",
            "description": "Turn a light on/off",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "enum": ["bright_room_light"]},
                    "state": {"type": "string", "enum": ["on", "off"]},
                },
                "required": ["entity", "state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scene",
            "description": "Activate a scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string", "enum": ["night"]},
                },
                "required": ["scene"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_dimmer",
            "description": "Set a dimmer brightness",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "enum": ["night_light"]},
                    "brightness": {"type": "integer"},
                },
                "required": ["device_id", "brightness"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_climate",
            "description": "Set a climate temperature",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "enum": ["room_ac"]},
                    "temperature": {"type": "string"},
                },
                "required": ["device_id", "temperature"],
            },
        },
    },
]

_ALIASES = (
    "свет в зале, люстра = bright_room_light\n"
    "ночь, спать = night\n"
    "ночник = night_light\n"
    "кондиционер = room_ac"
)


def _new_backend(filler_text=None, aliases=_ALIASES):
    """A fresh, untrained backend over the `aliases` names."""
    return SimpleNluBackend(
        aliases_text=aliases,
        actions_text=SimpleNluConfig().actions,
        filler_text=SimpleNluConfig().filler_words if filler_text is None else filler_text,
        classifier_threshold=SimpleNluConfig().classifier_threshold,
        level_on=SimpleNluConfig().level_on,
        off_token=SimpleNluConfig().off_token,
        reply_unmatched=SimpleNluConfig().reply_unmatched,
        reply_done=SimpleNluConfig().reply_done,
        reply_error=SimpleNluConfig().reply_error,
    )


# One backend per filler list for the module: it trains on its first request.
_backend = functools.cache(_new_backend)


# --- registration / provider ------------------------------------------------ #

def test_provider_is_registered():
    prov = get_provider("llm", "simple-nlu")
    assert prov.category == "llm"
    assert prov.id == "simple-nlu"
    assert prov.label == "Simple NLU (no LLM)"
    assert prov.uses_http_cloud is False


def test_describe():
    prov = get_provider("llm", "simple-nlu")
    assert prov.describe(SimpleNluConfig()) == "simple-nlu"


def test_create_returns_backend():
    prov = get_provider("llm", "simple-nlu")
    backend = prov.create(SimpleNluConfig(), _deps())
    assert isinstance(backend, SimpleNluBackend)


# --- pure helpers ----------------------------------------------------------- #

def test_parse_aliases_auto_and_explicit():
    entries = parse_aliases(
        "Свет в зале, Люстра = bright_room_light\n"
        "# a comment\n"
        "\n"
        "яркость = set_light.brightness:80\n"
        "broken line without eq\n"
    )
    assert entries[0]["phrases"] == ["свет в зале", "люстра"]
    assert entries[0]["value"] == "bright_room_light"
    assert entries[0]["tool"] is None and entries[0]["slot"] is None
    assert entries[1]["tool"] == "set_light"
    assert entries[1]["slot"] == "brightness"
    assert entries[1]["value"] == "80"
    assert len(entries) == 2  # comment, blank and the malformed line are skipped


def test_parse_actions():
    actions = parse_actions("on = Включи, Зажги\n# c\noff = выключи\n")
    assert actions == {"on": ["включи", "зажги"], "off": ["выключи"]}


def test_extract_number_digits_words_none():
    assert extract_number("поставь 42 процента") == 42
    assert extract_number("двадцать два") == 22
    assert extract_number("сто") == 100
    assert extract_number("пять") == 5
    assert extract_number("без чисел вообще") is None


# --- round 1: command -> tool_call ------------------------------------------ #

async def test_round1_turn_on():
    backend = _backend()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "включи свет в зале"},
    ]
    data = await backend.complete(messages, FAKE_TOOLS)
    msg = data["choices"][0]["message"]
    assert msg["content"] == ""
    calls = msg["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "set_light"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"entity": "bright_room_light", "state": "on"}
    assert data["usage"]["total_tokens"] == 0
    assert data["model"].startswith("simple-nlu")


async def test_round1_turn_off():
    backend = _backend()
    messages = [{"role": "user", "content": "выключи свет в зале"}]
    data = await backend.complete(messages, FAKE_TOOLS)
    args = json.loads(data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"entity": "bright_room_light", "state": "off"}


async def test_round1_scene_no_state_slot():
    backend = _backend()
    messages = [{"role": "user", "content": "включи ночь"}]
    data = await backend.complete(messages, FAKE_TOOLS)
    calls = data["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "set_scene"
    assert json.loads(calls[0]["function"]["arguments"]) == {"scene": "night"}


# --- level slot: numeric (dimmer) + string (climate) ------------------------ #

async def _call_args(backend, text):
    data = await backend.complete([{"role": "user", "content": text}], FAKE_TOOLS)
    call = data["choices"][0]["message"]["tool_calls"][0]["function"]
    return call["name"], json.loads(call["arguments"])


async def test_dimmer_on_sets_full_brightness():
    name, args = await _call_args(_backend(), "включи ночник")
    assert name == "set_dimmer"
    assert args == {"device_id": "night_light", "brightness": 100}


async def test_dimmer_off_sets_zero_brightness():
    name, args = await _call_args(_backend(), "выключи ночник")
    assert name == "set_dimmer"
    assert args == {"device_id": "night_light", "brightness": 0}


async def test_dimmer_explicit_number():
    name, args = await _call_args(_backend(), "поставь ночник на 30")
    assert name == "set_dimmer"
    assert args == {"device_id": "night_light", "brightness": 30}


async def test_climate_explicit_number_is_string():
    name, args = await _call_args(_backend(), "поставь кондиционер на 22")
    assert name == "set_climate"
    assert args == {"device_id": "room_ac", "temperature": "22"}  # string!


async def test_climate_off_sets_off_token():
    name, args = await _call_args(_backend(), "выключи кондиционер")
    assert name == "set_climate"
    assert args == {"device_id": "room_ac", "temperature": "off"}


# --- round 2: confirmation / error ------------------------------------------ #

async def test_round2_done():
    backend = _backend()
    messages = [
        {"role": "user", "content": "включи свет в зале"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0"}]},
        {"role": "tool", "tool_call_id": "call_0", "name": "set_light", "content": "ok"},
    ]
    data = await backend.complete(messages, FAKE_TOOLS)
    msg = data["choices"][0]["message"]
    assert msg["content"] == SimpleNluConfig().reply_done
    assert msg["tool_calls"] is None


async def test_round2_error():
    backend = _backend()
    messages = [
        {"role": "user", "content": "включи свет в зале"},
        {"role": "tool", "tool_call_id": "call_0", "name": "set_light",
         "content": "error calling set_light: boom"},
    ]
    data = await backend.complete(messages, FAKE_TOOLS)
    assert data["choices"][0]["message"]["content"] == SimpleNluConfig().reply_error


# --- no-match --------------------------------------------------------------- #

async def test_no_match():
    backend = _backend()
    messages = [{"role": "user", "content": "расскажи анекдот"}]
    data = await backend.complete(messages, FAKE_TOOLS)
    msg = data["choices"][0]["message"]
    assert msg["content"] == SimpleNluConfig().reply_unmatched
    assert msg["tool_calls"] is None


# --- end-to-end through the real LlmStage ----------------------------------- #

class FakeHub:
    """Tool-hub double: advertises the fake tools and records hub.call(...)."""

    def __init__(self, tools):
        self.tools = tools
        self.calls = []

    async def ensure_tools(self):
        return None

    async def call(self, name, args, *, device=None):
        self.calls.append((name, args))
        return "ok"


async def test_end_to_end_through_llm_stage():
    backend = _backend()
    hub = FakeHub(FAKE_TOOLS)
    stage = LlmStage(backend, hub, SimpleNluConfig(max_tool_rounds=2))

    result = await stage.respond(
        LlmRequest(system_prompt="s", history=[], user_text="включи свет в зале")
    )

    assert result.tool_used is True
    assert hub.calls == [("set_light", {"entity": "bright_room_light", "state": "on"})]
    assert result.reply == SimpleNluConfig().reply_done


# --- fingerprint / caching -------------------------------------------------- #

async def test_training_is_cached(monkeypatch):
    backend = _new_backend()
    counter = {"n": 0}
    original = backend._train

    def counting_train(tools):
        counter["n"] += 1
        return original(tools)

    monkeypatch.setattr(backend, "_train", counting_train)

    msgs = [{"role": "user", "content": "включи свет в зале"}]
    await backend.warm_up(FAKE_TOOLS)
    await backend.complete(msgs, FAKE_TOOLS)
    await backend.complete(msgs, FAKE_TOOLS)
    assert counter["n"] == 1  # same tools -> trained once, by the warm-up
    assert backend._fingerprint is not None


# --- nlu_vocabulary (Vosk closed-grammar vocabulary) ------------------------ #

def test_nlu_vocabulary_collects_phrase_and_verb_words_and_numbers():
    aliases = "свет в зале, люстра = bright_room_light\nночь = night"
    actions = "on = включи, вруби\noff = выключи"
    vocab = nlu_vocabulary(aliases, actions)

    # Every word of every alias phrase is present (phrases are split into tokens).
    for w in ("свет", "в", "зале", "люстра", "ночь"):
        assert w in vocab
    # Every action verb word is present, and its infinitive, which also commands.
    for w in ("включи", "вруби", "выключи", "включить", "врубить", "выключить"):
        assert w in vocab
    # So are the words a number brings along («ночник на 50 процентов»).
    for w in ("на", "до", "процентов"):
        assert w in vocab
    # Russian number words 0..100 are always included.
    for w in ("ноль", "один", "сто", "двадцать"):
        assert w in vocab

    # The RHS values (entity/scene ids) are NOT spoken words -> never in the vocab.
    assert "bright_room_light" not in vocab
    assert "night" not in vocab

    # Sorted and de-duplicated.
    assert vocab == sorted(set(vocab))


def test_nlu_vocabulary_empty_inputs_still_have_number_words():
    # Empty / comment-only inputs still yield a non-empty, sorted list (the numbers).
    vocab = nlu_vocabulary("", "# nothing here\n  ")
    assert vocab  # non-empty
    assert vocab == sorted(set(vocab))
    assert "один" in vocab and "сто" in vocab


def test_simple_nlu_provider_vocabulary_matches_pure_function():
    aliases = "свет в зале, люстра = bright_room_light"
    actions = "on = включи\noff = выключи"
    fillers = "захар, ну-ка"
    cfg = SimpleNluConfig(aliases=aliases, actions=actions, filler_words=fillers)
    vocab = SimpleNluProvider().vocabulary(cfg)
    assert vocab == nlu_vocabulary(aliases, actions, fillers)
    # Filler words are in the grammar, split into words like phrases are, so a
    # narrowed STT emits them rather than "[unk]".
    assert {"захар", "ну", "ка"} <= set(vocab)


# --- strict mode (a later chain takes what this handler declines) ------------ #

async def _strict_call(backend, text, tools=FAKE_TOOLS):
    """(envelope model, tool name, args) of the call the backend makes for `text`
    with a later chain behind it. Declined propagates."""
    data = await backend.complete([{"role": "user", "content": text}], tools, has_next=True)
    call = data["choices"][0]["message"]["tool_calls"][0]["function"]
    return data["model"], call["name"], json.loads(call["arguments"])


async def _strict(backend, text, tools=FAKE_TOOLS):
    """(tool name, args) the backend calls for `text` with a later chain behind it.
    Declined propagates."""
    _, name, args = await _strict_call(backend, text, tools)
    return name, args


@pytest.mark.parametrize("text, name, args", [
    # Filler words, the verb and the alias phrase explain every word.
    ("Захар, включи-ка свет в зале, пожалуйста", "set_light",
     {"entity": "bright_room_light", "state": "on"}),
    # The off-verb is explained by the level slot it filled.
    ("выключи ночник", "set_dimmer", {"device_id": "night_light", "brightness": 0}),
    # Number words and digits are explained by the number they filled.
    ("ночник двадцать пять", "set_dimmer", {"device_id": "night_light", "brightness": 25}),
    ("ночник 30", "set_dimmer", {"device_id": "night_light", "brightness": 30}),
    # A scene alias alone needs no verb.
    ("ночь", "set_scene", {"scene": "night"}),
])
async def test_strict_runs_a_command_whose_every_word_is_explained(text, name, args):
    assert await _strict_call(_backend(), text) == (f"simple-nlu/{name}", name, args)


@pytest.mark.parametrize("text, reason", [
    ("расскажи анекдот", "no alias phrase matched; classifier: NONE"),
    ("свет в зале", "no applicable verb for set_light; classifier: "),
    ("включи кондиционер", "missing required slot for set_climate: temperature; classifier: "),
    ("включи свет в зале и расскажи анекдот", "unknown words: и, расскажи, анекдот; classifier: NONE"),
    ("включи свет в зале и выключи ночник", "unknown words: и, выключи, ночник; classifier: "),
])
async def test_strict_decline_says_why(text, reason):
    # The reason names why the deterministic match declined, then what the
    # classifier made of the phrase.
    with pytest.raises(Declined) as ei:
        await _strict(_backend(), text)
    assert ei.value.reply == SimpleNluConfig().reply_unmatched
    assert ei.value.reason.startswith(reason)


async def test_last_chain_runs_a_direct_match_with_words_outside_the_command():
    # Without a later chain not every word needs explaining, only the gate must pass.
    backend = _backend()
    on = ("set_light", {"entity": "bright_room_light", "state": "on"})
    assert await _call_args(backend, "включи свет в зале уже") == on
    assert await _call_args(backend, "включи [unk] свет в зале") == on
    assert await _call_args(backend, "включи ночь") == ("set_scene", {"scene": "night"})


async def test_strict_matches_filler_words_by_word_form():
    # pymorphy3 lemmatizes the filler «мне» to «я»: fillers match by word form, so
    # «мне» is explained by itself and not by a filler «я», for the direct match and
    # the classifier alike.
    text = "включи мне свет в зале"
    assert (await _strict_call(_backend(), text))[0] == "simple-nlu/set_light"
    with pytest.raises(Declined) as ei:
        await _strict(_backend(filler_text="я"), text)
    assert ei.value.reason.endswith(", words outside the class: мне")


async def test_a_phrase_inside_a_longer_alias_names_no_device_of_its_own():
    # «ночь» (the night scene) inside «свет на ночь» (the night light) is no second
    # device for the gate.
    backend = _backend(aliases=_ALIASES.replace("ночник = ", "ночник, свет на ночь = "))
    assert await _strict_call(backend, "включи свет на ночь") == (
        "simple-nlu/set_dimmer", "set_dimmer", {"device_id": "night_light", "brightness": 100})


async def test_strict_round2_still_confirms():
    # A tool already ran: round 2 confirms it and never declines.
    messages = [
        {"role": "user", "content": "включи свет в зале"},
        {"role": "tool", "tool_call_id": "call_0", "name": "set_light", "content": "ok"},
    ]
    data = await _backend().complete(messages, FAKE_TOOLS, has_next=True)
    assert data["choices"][0]["message"]["content"] == SimpleNluConfig().reply_done


# --- production catalog regression ------------------------------------------- #

# The production tool catalog (every MCP source, as the panel's /api/tools lists it)
# and the Simple NLU settings it runs with.
_PROD = json.loads(
    (Path(__file__).parent / "fixtures" / "simple_nlu_prod_catalog.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def prod_backend():
    return get_provider("llm", "simple-nlu").create(SimpleNluConfig(**_PROD["nlu"]), _deps())


@pytest.mark.parametrize("text", [
    "Захар, какая погода будет утром?",
    "Захар, кто-то открыл дверь?",
    "Захар, я закрыл дверь?",
    "Захар, почему свет в зале мигает, когда я его включаю?",
    "Захар, включи свет в зале и расскажи анекдот",
    "Захар, включи свет на кухне и выключи в коридоре",
    "Захар, напомни через десять минут выключить ночник",
    "Захар, доброе утро, какие у меня встречи сегодня?",
    "Захар, сколько стоит новый монитор?",
    "Захар, что лучше: кондиционер или вентилятор?",
    "Захар, какая погода в Москве?",
    "Захар, выключи всё",
    "Захар выключен ночник",
])
async def test_prod_catalog_strict_declines(prod_backend, text):
    with pytest.raises(Declined):
        await _strict(prod_backend, text, _PROD["tools"])


@pytest.mark.parametrize("text, name, args", [
    ("Захар, спокойной ночи", "set_scene", {"scene": "night"}),
    ("Захар, включи настольный свет", "set_light", {"device_id": "table_light", "state": "on"}),
    ("Захар  Ночник 50.", "set_dimmer", {"device_id": "night_light", "brightness": 50}),
    # An infinitive commands like the imperative.
    ("Захар, включить настольный свет", "set_light", {"device_id": "table_light", "state": "on"}),
    # A number that set the level explains «на», «процентов» and the on-verb.
    ("Захар, включи ночник на пятьдесят процентов", "set_dimmer",
     {"device_id": "night_light", "brightness": 50}),
    # An on-verb says nothing more than the scene does.
    ("Захар, включи ночной режим", "set_scene", {"scene": "night"}),
    # A first word one letter off the wake word is a misheard wake word.
    ("Лахар, включи настольный свет", "set_light", {"device_id": "table_light", "state": "on"}),
    # «спасибо» is a filler word.
    ("Захар, включи настольный свет, спасибо", "set_light", {"device_id": "table_light", "state": "on"}),
    # An imperative with «мне» commands; only an infinitive with it asks.
    ("Захар, включи мне настольный свет", "set_light", {"device_id": "table_light", "state": "on"}),
    # Only the direct match unlocks.
    ("Захар, открой дверь", "set_lock", {"device_id": "main_lock", "action": "unlock"}),
])
async def test_prod_catalog_strict_runs_plain_commands(prod_backend, text, name, args):
    # The deterministic match runs them, not the classifier.
    assert await _strict_call(prod_backend, text, _PROD["tools"]) == (f"simple-nlu/{name}", name, args)


async def test_prod_catalog_past_tense_is_no_verb(prod_backend):
    # «закрыл» shares the lemma of «закрой» but commands nothing.
    with pytest.raises(Declined) as ei:
        await _strict(prod_backend, "Захар, закрыл дверь", _PROD["tools"])
    assert ei.value.reason.startswith("no applicable verb for set_lock;")


@pytest.mark.parametrize("text, name, args", [
    # «ванне» is no alias («ванная» is): the deterministic match finds no device.
    ("Захар, включи свет в ванне", "set_light", {"device_id": "bathroom_light", "state": "on"}),
    # A word names the class's device when its first four letters are an alias
    # word's («мониторчики», «монитор»)...
    ("Захар, вруби мониторчики", "set_switch", {"device_id": "monitors", "state": "on"}),
    # ...or it is one letter off one («зало», «зал»).
    ("Захар, включи свет в зало", "set_light", {"device_id": "bright_room_light", "state": "on"}),
    # An imperative the gate allows is no word outside the class.
    ("Захар, поставь ночной режим", "set_scene", {"scene": "night"}),
])
async def test_prod_catalog_classifier_runs_what_the_direct_match_declines(prod_backend, text, name, args):
    assert await _strict_call(prod_backend, text, _PROD["tools"]) == (
        f"simple-nlu/classifier/{name}", name, args)


async def test_prod_catalog_gate_stops_a_question_the_classifier_would_run(prod_backend):
    # The classifier's class clears the threshold (no "below"), the gate stops it.
    with pytest.raises(Declined) as ei:
        await _strict(prod_backend, "Захар, включи свет в ванне?", _PROD["tools"])
    reason = ei.value.reason
    assert 'classifier: set_light{"device_id": "bathroom_light", "state": "on"}' in reason
    assert reason.endswith(", gate: question")


@pytest.mark.parametrize("text, why", [
    # The direct match's unlock and light off, stopped as questions.
    ("Захар, открыть дверь?", "question"),
    ("Захар, мне выключить свет в зале?", "question"),
    ("Захар, зачем включать настольный свет", "question"),
    ("Захар, не включай настольный свет", "negation"),
    ("Захар, закрыл дверь", "past/participle"),
    # An infinitive after a dative pronoun or «а» asks; the direct match found it.
    ("Захар, мне открыть дверь", "infinitive question"),
    ("Захар, а открыть дверь", "infinitive question"),
    ("Захар, скажи включи настольный свет", "other verb"),
    ("Захар, включи выключи настольный свет", "two actions"),
    ("Захар, включи весь свет в зале", "quantifier"),
    ("Захар, выключи всю кухню", "quantifier"),
    ("Захар, включи свет в зале завтра", "delayed"),
    ("Захар, включи настольный свет ночью", "delayed"),  # by form: «ночью» is no «ночь» scene
    ("Захар, включи настольный свет на час", "delayed"),  # any form of «час», «минута», «секунда»
    ("Захар, включи свет в прихожей и чайник", "second request"),
    ("Захар, выключи мониторы и свет в ванне", "second request"),
    ("Захар, включи свет в зале на кухне", "two devices"),
])
async def test_prod_catalog_gate_says_why(prod_backend, text, why):
    with pytest.raises(Declined) as ei:
        await _strict(prod_backend, text, _PROD["tools"])
    assert ei.value.reason.endswith(f"gate: {why}")


@pytest.mark.parametrize("text, why", [
    # A misheard device word leaves the classifier unsure.
    ("Захар, включи настольный свят", "below 0.7"),
    # The unheard word may be «не» or a second request.
    ("Захар, включи [unk] настольный свет", "unheard word"),
    # A number may be a time as well as a level: the direct match takes levels.
    ("Захар, открой дверь в 9", "has a number"),
    ("Захар, выключи свет в зале в 22:00", "has a number"),
    ("Захар, ночной режим в 23", "has a number"),
    ("Захар, выключи ночник в 23", "has a number"),
    # A statement or a wish, not a command.
    ("Захар, можно закрыть дверь", "indicative or predicative: можно"),
    ("Захар, стоит закрыть дверь", "indicative or predicative: стоит"),
    ("Захар, ночной режим работает", "indicative or predicative: работает"),
    ("Захар, дверь", "verbs do not fit the class"),  # no verb: lock or unlock would be a guess
    ("Захар, дверь закрыта", "verbs do not fit the class"),  # a state, not a command
    ("Захар, выключи ночной режим", "verbs do not fit the class"),  # an off-verb is no scene
    ("Захар, заблокируй", "no device word"),  # however sure the classifier is of the lock
    # A word that names nothing of the class: a time, a place, another device.
    ("Захар, выключи свет в зале в полночь", "words outside the class: полночь"),
    ("Захар, выключи свет в ванне, в спальне", "words outside the class: спальне"),
    ("Захар, ночной режим это круто", "words outside the class: это, круто"),
    ("Захар, закрой дверь гаража", "words outside the class: гаража"),
    ("Захар, открой заднюю дверь", "words outside the class: заднюю"),
    ("Захар, разблокируй замочек", "unlock is left to the direct match"),
])
async def test_prod_catalog_classifier_says_why_it_does_not_act(prod_backend, text, why):
    with pytest.raises(Declined) as ei:
        await _strict(prod_backend, text, _PROD["tools"])
    assert ei.value.reason.endswith(f", {why}")


async def test_prod_catalog_last_chain_gate_stops_the_direct_match(prod_backend):
    # The deterministic match finds the light and «включай»; the negation stops it.
    data = await prod_backend.complete(
        [{"role": "user", "content": "Захар, не включай настольный свет"}], _PROD["tools"])
    msg = data["choices"][0]["message"]
    assert msg["tool_calls"] is None
    assert msg["content"] == SimpleNluConfig().reply_unmatched
