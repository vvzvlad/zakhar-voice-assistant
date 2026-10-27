"""Tests for the Jev decision-model handler against a fake Decisions endpoint.

The device options and verbs come from the command catalog (core.commands, the
Commands page), so the fixtures reuse the Simple NLU tests' tool catalog and aliases.
"""

import json

import httpx
import pytest

import src.plugins  # noqa: F401  triggers @register on all providers
from src.commands import (
    aliases_from_facts,
    build_lemmatizer,
    index_tools,
    parse_actions,
    parse_aliases,
)
from src.config_service import ConfigService
from src.core_config import CommandsConfig
from src.llm import LlmRequest, LlmStage
from src.plugins.base import Deps, get_provider
from src.plugins.llm._decisions import DECISIONS_API_URL
from src.plugins.llm.jev import JevConfig, _named_targets, _targets
from src.stage_errors import Declined
from tests.test_simple_nlu import _ALIASES, _PROD, FAKE_TOOLS, FakeHub, _strict


def _client(respond, requests):
    """A client whose calls go to `respond(request) -> httpx.Response`; `requests`
    collects every call."""
    def handler(request):
        requests.append(request)
        return respond(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _jev(respond, memory=None, **catalog):
    """(backend, requests): a jev backend reading the Commands names `_ALIASES` and
    the other `catalog` fields; `memory` is the Deps.memory callable."""
    requests = []
    client = _client(respond, requests)
    fields = {"aliases": _ALIASES, **catalog}
    deps = Deps(http_cloud=client, http_local=client,
                commands=lambda: CommandsConfig(**fields),
                memory=memory)
    return get_provider("llm", "jev").create(JevConfig(api_key="key"), deps), requests


def _choice(choice, confidence, rest, probabilities=None):
    """A choice answer; by default `choice` at 0.9 and `rest` at 0.1, so the shown
    probabilities differ from the confidence the gates read."""
    return {"type": "choice", "choice": choice, "confidence": confidence,
            "probabilities": {choice: 0.9, rest: 0.1} if probabilities is None else probabilities}


def _answers(kind="device_command", target="set_light.entity=bright_room_light", action="on",
             compound=0.05, kind_conf=0.95, target_conf=0.9, action_conf=0.9, delayed=0.05,
             kind_probs=None, target_probs=None):
    """A Decisions response body answering the five questions."""
    def reply(request):
        return httpx.Response(200, json={"answers": {
            "kind": _choice(kind, kind_conf, "other" if kind != "other" else "device_command",
                            kind_probs),
            "target": _choice(target, target_conf, "none" if target != "none" else
                              "set_scene.scene=night", target_probs),
            "action": _choice(action, action_conf, "off" if action != "off" else "on"),
            "compound": {"type": "noul", "noul": compound},
            "delayed": {"type": "noul", "noul": delayed},
        }})
    return reply


_NIGHT_LAMP_FACT = "Маленькая комната студии — это подсобка, её светильник — night_lamp."


def test_aliases_from_facts_anchor_by_the_latin_id():
    catalog = parse_aliases("ночник = night_lamp\nкладовка, свет в кладовке = storage_room_light")
    assert aliases_from_facts([_NIGHT_LAMP_FACT], catalog) == [
        {"phrases": ["маленькая комната студии", "подсобка", "её светильник"],
         "value": "night_lamp", "tool": None, "slot": None},
    ]


def test_aliases_from_facts_anchor_by_a_catalog_alias():
    catalog = parse_aliases("ночник = night_lamp\nкладовка, свет в кладовке = storage_room_light")
    assert aliases_from_facts(["Подсобка — это кладовка."], catalog) == [
        {"phrases": ["подсобка"], "value": "storage_room_light", "tool": None, "slot": None},
    ]


def test_aliases_from_facts_skip_unanchored_facts():
    catalog = parse_aliases("ночник = night_lamp\nкладовка, свет в кладовке = storage_room_light")
    # No «X — это Y» clause, no latin id: nothing to anchor.
    assert aliases_from_facts(["Пользователь находится в Москве."], catalog) == []
    # Two distinct device values in one fact.
    assert aliases_from_facts(["подсобка — night_lamp, а лампа — wled."], catalog) == []
    # A name the catalog already has adds no duplicate criteria text.
    assert aliases_from_facts(["ночник — это night_lamp."], catalog) == [
        {"phrases": ["night_lamp"], "value": "night_lamp", "tool": None, "slot": None},
    ]
    # A bare id without a dash clause yields no names: an empty criteria option
    # would only confuse the model.
    assert aliases_from_facts(["night_lamp."], catalog) == []
    # Two facts about one device: the second adds no repeated name.
    assert aliases_from_facts(["Подсобка — night_lamp.", "Подсобка — это ночник."], catalog) == [
        {"phrases": ["подсобка"], "value": "night_lamp", "tool": None, "slot": None},
    ]


@pytest.mark.parametrize("text, respond, name, args", [
    ("Захар, зажги люстру", _answers(), "set_light",
     {"entity": "bright_room_light", "state": "on"}),
    ("ночничок погаси", _answers(target="set_dimmer.device_id=night_light", action="off"),
     "set_dimmer", {"device_id": "night_light", "brightness": 0}),
])
async def test_runs_a_confident_command(text, respond, name, args):
    backend, _ = _jev(respond)
    data = await backend.complete([{"role": "user", "content": text}], FAKE_TOOLS, has_next=True)
    msg = data["choices"][0]["message"]
    assert msg["content"] == ""
    assert msg["tool_calls"][0]["function"]["name"] == name
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == args
    assert data["model"] == f"jev/{name}"


async def test_an_on_verb_sets_the_on_level_of_the_catalog():
    backend, _ = _jev(_answers(target="set_dimmer.device_id=night_light", action="on"), level_on=60)
    assert await _strict(backend, "включи ночник") == (
        "set_dimmer", {"device_id": "night_light", "brightness": 60})


async def test_end_to_end_through_llm_stage():
    # Round 1 asks the model once and calls the tool; round 2 only confirms.
    backend, requests = _jev(_answers())
    hub = FakeHub(FAKE_TOOLS)
    result = await LlmStage(backend, hub, JevConfig()).respond(
        LlmRequest(system_prompt="s", history=[], user_text="Захар, зажги люстру"))
    assert len(requests) == 1
    assert hub.calls == [("set_light", {"entity": "bright_room_light", "state": "on"})]
    assert result.tool_used is True
    assert result.reply == JevConfig().reply_done


async def test_last_chain_speaks_reply_unmatched_instead_of_declining():
    backend, _ = _jev(_answers(kind="other"))
    data = await backend.complete([{"role": "user", "content": "который час"}], FAKE_TOOLS)
    msg = data["choices"][0]["message"]
    assert msg["content"] == JevConfig().reply_unmatched
    assert msg["tool_calls"] is None


async def test_request_carries_the_alias_options():
    backend, requests = _jev(_answers())
    await _strict(backend, "Захар, зажги люстру")
    request = requests[0]
    assert str(request.url) == DECISIONS_API_URL
    assert request.headers["authorization"] == "Bearer key"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-1.14"
    assert body["state"] == {"utterance": "Захар, зажги люстру"}
    questions = body["questions"]
    assert set(questions["kind"]["criteria"]) == {"device_command", "other"}
    assert questions["target"]["criteria"] == {
        "set_light.entity=bright_room_light": "свет в зале, люстра",
        "set_scene.scene=night": "ночь, спать",
        "set_dimmer.device_id=night_light": "ночник",
        "set_climate.device_id=room_ac": "кондиционер",
        "none": "No device or scene is named",
    }
    assert questions["action"]["criteria"] == {
        "on": "включи, включай, зажги, зажги свет, вруби, врубай, запусти",
        "off": "выключи, выключай, погаси, выруби, отключи, останови",
        "none": "No action verb",
    }
    assert questions["compound"]["type"] == "noul"
    assert questions["delayed"]["type"] == "noul"


_LIGHT_TOOLS = [{"type": "function", "function": {
    "name": "set_light", "description": "Light",
    "parameters": {"type": "object", "properties": {
        "entity": {"type": "string", "enum": ["night_lamp"]},
        "state": {"type": "string", "enum": ["on", "off"]}},
        "required": ["entity", "state"]}}}]


async def test_memory_facts_name_the_device():
    backend, requests = _jev(_answers(target="set_light.entity=night_lamp"),
                             memory=lambda: [_NIGHT_LAMP_FACT],
                             aliases="ночник = night_lamp")
    assert await _strict(backend, "включи свет в подсобке", _LIGHT_TOOLS) == (
        "set_light", {"entity": "night_lamp", "state": "on"})
    criteria = json.loads(requests[0].content)["questions"]["target"]["criteria"]
    assert criteria["set_light.entity=night_lamp"] == (
        "ночник, маленькая комната студии, подсобка, её светильник")

    # Without the memory the fact's names are not among the criteria.
    backend, requests = _jev(_answers())
    await _strict(backend, "включи свет в подсобке")
    criteria = json.loads(requests[0].content)["questions"]["target"]["criteria"]
    assert "set_light.entity=night_lamp" not in criteria


async def test_an_edited_commands_alias_reaches_the_next_request_without_a_rebuild(tmp_path):
    # The real wiring: Deps.commands bound to the ConfigService, one backend built
    # before the Commands page's aliases change.
    requests = []
    client = _client(_answers(), requests)
    deps = Deps(http_cloud=client, http_local=client)
    svc = ConfigService({
        "stt": {"instances": {}},
        "llm": {"instances": {}},
        "tts": {"variants": [{"provider": "yandex"}]},
        "core": {"chains": [{"name": "jev", "stt": "groq", "handler": "jev"}],
                 "commands": {"aliases": _ALIASES}},
    }, deps, path=str(tmp_path / "config.json"))
    deps.commands = lambda: svc.core.commands
    backend = svc.create_instance("llm", "jev")

    await _strict(backend, "Захар, зажги люстру")
    edited = "большой свет = bright_room_light"
    svc.apply({"core": {"commands": {"aliases": edited}}})
    await _strict(backend, "Захар, зажги люстру")

    assert [json.loads(r.content)["questions"]["target"]["criteria"] for r in requests] == [
        {"set_light.entity=bright_room_light": "свет в зале, люстра",
         "set_scene.scene=night": "ночь, спать",
         "set_dimmer.device_id=night_light": "ночник",
         "set_climate.device_id=room_ac": "кондиционер",
         "none": "No device or scene is named"},
        {"set_light.entity=bright_room_light": "большой свет",
         "none": "No device or scene is named"},
    ]


@pytest.mark.parametrize("respond, reason", [
    (_answers(kind="other"),
     "decision model: kind other (other 90%, device_command 10%; confidence 95%)"),
    (_answers(kind_conf=0.6),
     "decision model: kind device_command (device_command 90%, other 10%; confidence 60%)"),
    (_answers(target="none"),
     "decision model: target none (none 90%, set_scene.scene=night 10%; confidence 90%)"),
    (_answers(target_conf=0.5),
     "decision model: target set_light.entity=bright_room_light "
     "(set_light.entity=bright_room_light 90%, none 10%; confidence 50%)"),
    (_answers(compound=0.7), "decision model: compound (70%)"),
    (_answers(action="none"), "decision model: action none does not fit set_light"),
    (_answers(action_conf=0.4), "decision model: action on (on 90%, off 10%; confidence 40%)"),
    (_answers(delayed=0.8), "decision model: delayed (80%)"),
    # The API rounds each probability: the shown percents are divided by their sum
    # (0.99 here) and the lost percent goes to the largest remainder; options at 0%
    # are left out.
    (_answers(kind="other", kind_conf=0.48, kind_probs={"device_command": 0.26, "other": 0.73}),
     "decision model: kind other (other 74%, device_command 26%; confidence 48%)"),
    (_answers(target_conf=0.1, target_probs={
        "set_light.entity=bright_room_light": 0.33, "set_scene.scene=night": 0.33,
        "set_dimmer.device_id=night_light": 0, "set_climate.device_id=room_ac": 0, "none": 0.33}),
     "decision model: target set_light.entity=bright_room_light "
     "(set_light.entity=bright_room_light 34%, set_scene.scene=night 33%, none 33%; "
     "confidence 10%)"),
    # 50/26/25 sums to 101: the extra percent comes off the smallest remainder (0.50),
    # not off the most likely option.
    (_answers(target_conf=0.1, target_probs={
        "set_light.entity=bright_room_light": 0.5, "set_scene.scene=night": 0.26, "none": 0.25}),
     "decision model: target set_light.entity=bright_room_light "
     "(set_light.entity=bright_room_light 49%, set_scene.scene=night 26%, none 25%; "
     "confidence 10%)"),
])
async def test_declines_what_is_not_one_confident_command(respond, reason):
    backend, _ = _jev(respond)
    with pytest.raises(Declined) as ei:
        await _strict(backend, "Захар, зажги люстру")
    assert ei.value.reason == reason


async def test_an_action_that_sets_a_level_needs_confidence():
    backend, _ = _jev(
        _answers(target="set_dimmer.device_id=night_light", action="off", action_conf=0.4))
    with pytest.raises(Declined) as ei:
        await _strict(backend, "ночничок погаси")
    assert ei.value.reason == "decision model: action off (off 90%, on 10%; confidence 40%)"


@pytest.mark.parametrize("text, tools, reason", [
    # «на пять» may be a level, a delay or a step: no request, the next chain decides.
    ("прибавь ночничок на пять", FAKE_TOOLS, "decision model: a number is left to the next chain"),
    ("зажги лампу", [], "decision model: no device options"),
    # Two devices by their aliases: the next chain does both, even when «и» is lost.
    ("включи ночник люстру", FAKE_TOOLS, "decision model: several devices named: "
     "set_light.entity=bright_room_light, set_dimmer.device_id=night_light"),
])
async def test_is_not_asked_what_it_cannot_answer(text, tools, reason):
    backend, requests = _jev(_answers())
    with pytest.raises(Declined) as ei:
        await _strict(backend, text, tools)
    assert ei.value.reason == reason
    assert requests == []


@pytest.mark.parametrize("text, named", [
    ("захар включи свет в туалете на кухне", ["kitchen_light", "toilet_light"]),
    ("включи свет в ванной и в туалете", ["toilet_light", "bathroom_light"]),
    ("выключи свет в прихожей", ["corridor_light"]),
    ("включи свет на кухне", ["kitchen_light"]),
    ("открой дверь", ["main_lock"]),
    ("спокойной ночи", ["night"]),
])
def test_names_the_devices_of_the_production_list(text, named):
    pytest.importorskip("pymorphy3")
    tools_index = index_tools(_PROD["tools"], set(parse_actions(_PROD["commands"]["actions"])))
    targets = _targets(parse_aliases(_PROD["commands"]["aliases"]), tools_index)
    keys = _named_targets(targets, text, build_lemmatizer())
    assert [key.split("=")[1] for key in keys] == named


def test_a_word_other_devices_share_names_no_device():
    pytest.importorskip("pymorphy3")
    targets = {
        "bright": {"phrases": ["свет", "люстра"]},
        "kitchen": {"phrases": ["свет на кухне", "кухня"]},
        "toilet": {"phrases": ["свет в туалете", "туалет"]},
        "corridor": {"phrases": ["прихожая"]},
    }
    lemmatize = build_lemmatizer()
    assert _named_targets(targets, "захар включи свет в туалете на кухне", lemmatize) == [
        "kitchen", "toilet"]
    assert _named_targets(targets, "Включи свет на кухне", lemmatize) == ["kitchen"]
    assert _named_targets(targets, "выключи свет в прихожей", lemmatize) == ["corridor"]
    assert _named_targets(targets, "включи люстру", lemmatize) == ["bright"]
    assert _named_targets(targets, "включи свет", lemmatize) == []


def _timeout(request):
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize("respond, error", [
    (lambda request: httpx.Response(500, json={"error": "boom"}), "HTTPStatusError"),
    (_timeout, "ReadTimeout"),
    (lambda request: httpx.Response(200, json={"answers": {}}), "KeyError"),
    (_answers(kind_probs={}), "ValueError"),
    (_answers(kind_probs={"device_command": 0, "other": 0}), "ValueError"),
    (_answers(kind_probs=[0.9, 0.1]), "ValueError"),
])
async def test_failure_declines_without_retry(respond, error):
    backend, requests = _jev(respond)
    with pytest.raises(Declined) as ei:
        await _strict(backend, "Захар, зажги люстру")
    assert ei.value.reason.startswith(f"decision model error: {error}")
    assert len(requests) == 1
