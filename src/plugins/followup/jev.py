"""Jev follow-up check: one OpenRouter Decisions API request per phrase of the window.

A phrase heard in the follow-up window goes to the jev decision model together with
the assistant's name, the last request and answer, with one yes/no question: is it addressed to the
assistant as a continuation of this dialog? It goes on when the model's probability
reaches min_confidence; any failure of the call drops it.

The model and the API key are not its own settings: every request reads them from the
jev handler instance (llm "jev") through Deps.instance_config, so they are set in one
place and an edit reaches the next request without a rebuild.
"""

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from src.followup import FollowupGate, FollowupVerdict, NoFollowup
from src.plugins.base import ADVANCED_FIELD_EXTRA, Deps, Provider, register
from src.plugins.llm.jev import DECISIONS_API_URL, JevProvider, _noul_answer

_CONTINUATION_QUESTION = {
    "type": "noul",
    # The name matters: without it a phrase opening with «Захар» reads to the model as
    # addressed to a person called Zakhar, and an explicit command in the window is lost.
    "instructions": (
        "The voice assistant, called `assistant_name`, has just answered the user and "
        "keeps listening for a few seconds without being called by its name. Is "
        "`utterance` addressed to the assistant as a continuation of this dialog (an "
        "answer to its question, a follow-up request, a correction, or a new request to "
        "it), rather than people talking to each other, a TV or radio, or noise? A "
        "phrase that calls the assistant by its name is addressed to it."
    ),
    "criteria": {"true": "", "false": ""},
}


class JevFollowupConfig(BaseModel):
    enabled: bool = Field(
        False,
        title="Follow-up",
        description=(
            "After the spoken answer keep listening for a few seconds without the wake "
            "word. A phrase heard there goes on only when the jev decision model takes "
            "it for a continuation of the dialog. Uses the model and API key of the jev "
            "handler; works on devices that support it."
        ),
    )
    window_ms: int = Field(
        5000,
        title="Window",
        description="How long to wait for speech to start after the answer.",
        json_schema_extra={"unit": "ms"},
    )
    min_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        title="Confidence threshold",
        description=(
            "The decision model's probability that the phrase continues the dialog must "
            "reach this for the phrase to go on."
        ),
    )
    timeout: float = Field(
        2.0,
        title="Request timeout",
        description="Past this the check gives up and the phrase is dropped.",
        json_schema_extra={**ADVANCED_FIELD_EXTRA, "unit": "s"},
    )


class JevFollowupGate(FollowupGate):
    """FollowupGate over one decision request per phrase."""

    def __init__(self, *, window_ms: int, min_confidence: float, timeout: float,
                 client: httpx.AsyncClient, instance_config):
        self.window_ms = window_ms
        self._min_confidence = min_confidence
        self._timeout = timeout
        self._client = client
        # (category, provider id) -> that instance's live config (Deps.instance_config).
        self._instance_config = instance_config

    async def check(self, name: str, request: str, reply: str,
                    utterance: str) -> FollowupVerdict:
        try:
            jev = self._instance_config("llm", JevProvider.id)
            resp = await self._client.post(
                DECISIONS_API_URL,
                headers={"Authorization": f"Bearer {jev.api_key}"},
                json={
                    "model": jev.model,
                    "state": {"assistant_name": name, "request": request, "reply": reply,
                              "utterance": utterance},
                    "questions": {"continuation": _CONTINUATION_QUESTION},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            probability = _noul_answer(resp.json()["answers"], "continuation")
        except Exception as e:  # noqa: BLE001 - the check never raises: a failure drops the phrase
            logger.warning(f"jev follow-up: decision model call failed: {type(e).__name__}: {e}")
            return FollowupVerdict(accepted=False, probability=None)
        accepted = probability >= self._min_confidence
        logger.info(
            f"jev follow-up: continuation={probability:.0%} "
            f"-> {'accept' if accepted else 'reject'}: {utterance!r}"
        )
        return FollowupVerdict(accepted=accepted, probability=probability)


@register
class JevFollowupProvider(Provider):
    category = "followup"
    id = "jev"
    label = "Jev (decision model)"
    ConfigModel = JevFollowupConfig
    uses_http_cloud = True

    def create(self, cfg: JevFollowupConfig, deps: Deps):
        if not cfg.enabled:
            # Disabled stage: the pipeline opens no follow-up window.
            return NoFollowup()
        return JevFollowupGate(
            window_ms=cfg.window_ms,
            min_confidence=cfg.min_confidence,
            timeout=cfg.timeout,
            client=deps.http_cloud,
            instance_config=deps.instance_config,
        )

    def describe(self, cfg: JevFollowupConfig) -> str:
        return self.id if cfg.enabled else f"{self.id}/off"
