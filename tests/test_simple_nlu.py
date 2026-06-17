"""Tests for the offline Simple NLU 'LLM' provider.

The provider uses NO LLM and NO network. These tests exercise primarily the
DETERMINISTIC path with an INJECTED identity lemmatizer (`lambda w: w.lower()`),
so they do not depend on pymorphy3/sklearn being installed or on classifier
quality. They cover registration, the pure helpers, the OpenAI-shaped envelope
across both rounds, and one end-to-end pass through the real LlmStage loop.
"""

import json

import httpx

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


def _backend():
    return SimpleNluBackend(
        aliases_text=_ALIASES,
        actions_text=SimpleNluConfig().actions,
        min_confidence=SimpleNluConfig().min_confidence,
        fuzzy_threshold=SimpleNluConfig().fuzzy_threshold,
        level_on=SimpleNluConfig().level_on,
        off_token=SimpleNluConfig().off_token,
        reply_unmatched=SimpleNluConfig().reply_unmatched,
        reply_done=SimpleNluConfig().reply_done,
        reply_error=SimpleNluConfig().reply_error,
        lemmatizer=lambda w: w.lower(),
    )


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
    backend = _backend()
    counter = {"n": 0}
    original = backend._train

    def counting_train(tools):
        counter["n"] += 1
        return original(tools)

    monkeypatch.setattr(backend, "_train", counting_train)

    msgs = [{"role": "user", "content": "включи свет в зале"}]
    await backend.complete(msgs, FAKE_TOOLS)
    await backend.complete(msgs, FAKE_TOOLS)
    assert counter["n"] == 1  # same tools -> trained once
    assert backend._fingerprint is not None


# --- nlu_vocabulary (Vosk closed-grammar vocabulary) ------------------------ #

def test_nlu_vocabulary_collects_phrase_and_verb_words_and_numbers():
    aliases = "свет в зале, люстра = bright_room_light\nночь = night"
    actions = "on = включи, вруби\noff = выключи"
    vocab = nlu_vocabulary(aliases, actions)

    # Every word of every alias phrase is present (phrases are split into tokens).
    for w in ("свет", "в", "зале", "люстра", "ночь"):
        assert w in vocab
    # Every action verb word is present.
    for w in ("включи", "вруби", "выключи"):
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
    cfg = SimpleNluConfig(aliases=aliases, actions=actions)
    assert SimpleNluProvider().vocabulary(cfg) == nlu_vocabulary(aliases, actions)
