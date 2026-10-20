"""The jev follow-up check (src/plugins/followup/jev.py) against a fake Decisions endpoint,
and the followup stage slot of the config."""

import json

import httpx
import pytest

import src.plugins  # noqa: F401  triggers @register on all providers
from src.config_service import ConfigService
from src.followup import FollowupVerdict, NoFollowup
from src.plugins.base import Deps, get_provider
from src.plugins.followup.jev import JevFollowupConfig
from src.plugins.llm._decisions import DECISIONS_API_URL
from src.plugins.llm.jev import JevConfig


def _gate(respond, **cfg):
    """(gate, requests): an enabled jev follow-up check whose jev handler instance has
    its own model and key; `requests` collects every call."""
    requests = []

    def handler(request):
        requests.append(request)
        return respond(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    jev = JevConfig(model="typesafe/jev-test", api_key="jev-key")
    deps = Deps(http_cloud=client, http_local=client,
                instance_config=lambda *key: {("llm", "jev"): jev}[key])
    cfg = JevFollowupConfig(enabled=True, **cfg)
    return get_provider("followup", "jev").create(cfg, deps), requests


def _continuation(p):
    return lambda request: httpx.Response(
        200, json={"answers": {"continuation": {"type": "noul", "noul": p}}})


async def test_one_request_with_the_turn_and_the_jev_handlers_model_and_key():
    gate, requests = _gate(_continuation(0.9), window_ms=4000)
    verdict = await gate.check("Захар", "включи свет", "Готово.", "а в спальне")
    assert verdict == FollowupVerdict(accepted=True, probability=0.9)
    assert gate.window_ms == 4000
    [request] = requests
    assert str(request.url) == DECISIONS_API_URL
    assert request.headers["authorization"] == "Bearer jev-key"
    body = json.loads(request.content)
    assert body["model"] == "typesafe/jev-test"
    assert body["state"] == {"assistant_name": "Захар", "request": "включи свет",
                             "reply": "Готово.", "utterance": "а в спальне"}
    assert list(body["questions"]) == ["continuation"]
    assert body["questions"]["continuation"]["type"] == "noul"


@pytest.mark.parametrize("p, accepted", [(0.5, True), (0.49, False)])
async def test_accepted_from_min_confidence_up(p, accepted):
    gate, _ = _gate(_continuation(p))
    assert await gate.check("n", "r", "a", "u") == FollowupVerdict(accepted=accepted, probability=p)


def _timeout(request):
    raise httpx.ReadTimeout("timed out", request=request)


@pytest.mark.parametrize("respond", [
    lambda request: httpx.Response(500, json={"error": "boom"}),
    _timeout,
    lambda request: httpx.Response(200, json={"answers": {}}),
    lambda request: httpx.Response(200, text="not json"),
])
async def test_a_failed_check_drops_the_phrase_without_raising(respond):
    gate, requests = _gate(respond)
    assert await gate.check("n", "r", "a", "u") == FollowupVerdict(accepted=False, probability=None)
    assert len(requests) == 1


def test_off_by_default_and_a_config_without_the_slot_parses(tmp_path):
    svc = ConfigService({
        "stt": {"instances": {}},
        "llm": {"instances": {}},
        "tts": {"variants": [{"provider": "yandex"}]},
        "core": {"chains": [{"name": "main", "stt": "groq", "handler": "openrouter"}]},
    }, Deps(http_cloud=None, http_local=None), path=str(tmp_path / "config.json"))
    assert isinstance(svc.create("followup"), NoFollowup)
    assert svc.create("followup").backend_desc == "jev/off"
    followup = next(c for c in svc.catalog()["categories"] if c["id"] == "followup")
    assert followup["selected"] == "jev"
    assert followup["providers"][0]["schema"]["properties"]["window_ms"]["apply"] == "rebuild_backends"
