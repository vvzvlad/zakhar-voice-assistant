"""Recognition chains: the ordered {STT, handler} pairs a request is offered to.

core.chains lists them. One captured utterance (or one text request) goes through
the ENABLED chains in list order until one answers. A chain is passed over when its
STT fails or hears nothing, when its handler declines (stage_errors.Declined), or
when its handler fails before any tool ran. A handler failing AFTER a tool ran stops
the sequence: the action already happened, so no later chain may repeat it.

The first chain's STT and every `parallel` chain's STT start at speech: the pipeline
opens their live sessions (open_stream) when speech starts, and run() starts all of
them (a session's finish() or a batch transcribe) concurrently the moment the
utterance ends. Any other chain's STT starts only when that chain is reached. The
results of parallel chains that are never reached are cancelled.

A chain whose STT and vocabulary equal an earlier chain's (parallel or not) opens no
session and starts no recognition: it takes that chain's transcript. Only a
recognition that succeeded is shared; after a failed one the chain recognizes the
utterance itself when reached.
"""

import asyncio
import time
from dataclasses import dataclass, field

from loguru import logger

from src import llm
from src.stage_errors import Declined, StageError
from src.stt import contains_stt_hallucination


def enabled_chains(chains) -> list:
    """The chains a request goes through, in order."""
    return [c for c in chains if c.enabled]


def starts_at_speech(rt, chains) -> list:
    """Enabled chains whose STT starts at speech: the first one and every parallel one,
    except a chain that takes an earlier chain's transcript (_transcript_key)."""
    seen = set()
    out = []
    for i, c in enumerate(enabled_chains(chains)):
        key = _transcript_key(rt, c)
        if (i == 0 or c.parallel) and key not in seen:
            out.append(c)
        seen.add(key)
    return out


def referenced_instances(chains) -> set[tuple[str, str]]:
    """(category, provider id) of every STT and handler an enabled chain uses — the
    backends the runtime keeps built."""
    refs = set()
    for c in enabled_chains(chains):
        refs |= {("stt", c.stt), ("llm", c.handler)}
    return refs


def spoken_fallback(err: StageError, cfg) -> str:
    """The handler's configured phrase spoken for its failure instead of the raw error."""
    return cfg.reply_rate_limit if err.kind == "rate_limit" else cfg.reply_error


def _backend(rt, category: str, provider_id: str):
    backend = getattr(rt, f"{category}_backends").get(provider_id)
    if backend is None:
        raise StageError(category, f"{category} backend {provider_id!r} is not loaded")
    return backend


def _desc(rt, category: str, provider_id: str) -> str:
    """'provider/model' identity of a built backend for log lines (the id when unstamped)."""
    backend = getattr(rt, f"{category}_backends").get(provider_id)
    return getattr(backend, "backend_desc", None) or provider_id


def _vocabulary(rt, chain, stt) -> list[str] | None:
    """The handler's vocabulary this chain restricts its STT to, or None."""
    if chain.narrow and getattr(stt, "supports_vocabulary", False):
        return rt.svc.vocabulary(chain.handler) or None
    return None


def _transcript_key(rt, chain) -> tuple:
    """Chains with equal keys hear one utterance the same: the same STT narrowed to the
    same vocabulary (None when not narrowed)."""
    vocab = _vocabulary(rt, chain, rt.stt_backends.get(chain.stt))
    return chain.stt, tuple(vocab) if vocab else None


def open_stream(rt, chain):
    """Open the chain's live STT session, or None for a batch-only backend. Raises
    when the backend is not loaded or the open fails (the caller then goes batch)."""
    stt = _backend(rt, "stt", chain.stt)
    opener = getattr(stt, "open_stream", None)
    if opener is None:
        return None
    vocab = _vocabulary(rt, chain, stt)
    return opener(vocabulary=vocab) if vocab else opener()


async def _recognize(rt, chain, pcm: bytes, stream) -> str:
    """The chain's transcript: its live session's final, else a batch transcribe."""
    if stream is not None:
        return await stream.finish()
    stt = _backend(rt, "stt", chain.stt)
    vocab = _vocabulary(rt, chain, stt)
    if vocab:
        return await stt.transcribe(pcm, vocabulary=vocab)
    return await stt.transcribe(pcm)


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _attempt(chain, *, voice: bool) -> dict:
    """One entry of the run record's `attempts`. `stt` stays None for a text request."""
    return {
        "kind": "chain", "chain": chain.name,
        "stt": chain.stt if voice else None, "handler": chain.handler,
        "text": "", "outcome": None, "error": None, "detail": None, "t_stt": 0, "t_handler": 0,
    }


@dataclass
class ChainRun:
    """What offering one request to the chains produced."""

    attempts: list = field(default_factory=list)  # run-record `attempts`, in chain order
    chain: str | None = None    # chain whose handler served it (answered, or failed after a tool)
    text: str = ""              # text the last handler that ran was given ("" when none ran)
    reply: str | None = None    # what to speak; None when no handler ran
    answered: bool = False      # a handler answered: the turn belongs in the context history
    result: llm.LlmResult | None = None  # the answering handler's result
    partial: dict = field(default_factory=dict)  # observability of the last failed handler
    error: StageError | None = None  # the last STT/handler failure when no handler answered
    t_stt: int = 0              # ms spent waiting on STT, all attempts
    t_llm: int = 0              # ms spent in handlers, all attempts


async def run(rt, *, device: str, request, pcm: bytes = b"", text: str | None = None,
              streams=None, on_text=None, on_filler=None, gate=None) -> ChainRun:
    """Offer a request to the enabled chains in order until one answers.

    Voice (text is None): `pcm` is the utterance and `streams` the live sessions
    opened at speech by open_stream(), keyed by chain name; `on_text(text, t_stt)`
    fires once, with the first non-empty transcript and the ms spent on STT so far.
    `await gate(text)` is asked about that transcript before on_text and before any
    handler; False stops the run there (its attempt's outcome is "rejected").
    Text: `text` enters every chain at
    its handler, no STT runs. `request(text)` builds the handler's LlmRequest and
    `on_filler` is passed to LlmStage.respond(). Exceptions other than StageError
    and Declined propagate."""
    chains = rt.core.chains
    active = enabled_chains(chains)
    voice = text is None
    out = ChainRun()
    early: dict[str, asyncio.Task] = {}
    # Successful recognitions of this utterance by _transcript_key, for the later
    # chains that share an earlier chain's STT and vocabulary.
    transcripts: dict[tuple, str] = {}
    if voice:
        streams = streams or {}
        for chain in starts_at_speech(rt, chains):
            early[chain.name] = asyncio.create_task(
                _recognize(rt, chain, pcm, streams.get(chain.name)))
    heard = False
    try:
        for i, chain in enumerate(active):
            attempt = _attempt(chain, voice=voice)
            out.attempts.append(attempt)
            utterance = text
            if voice:
                key = _transcript_key(rt, chain)
                t = time.perf_counter()
                if key in transcripts:
                    utterance = transcripts[key]
                else:
                    task = early.pop(chain.name, None)
                    try:
                        utterance = await (task if task is not None
                                           else _recognize(rt, chain, pcm, None))
                    except StageError as e:
                        attempt.update(outcome="stt_error", error=str(e), t_stt=_ms(t))
                        out.t_stt += attempt["t_stt"]
                        out.error = e
                        logger.error(f"{device}: STT [{_desc(rt, 'stt', chain.stt)}] "
                                     f"chain {chain.name!r} failed: {e}")
                        continue
                    transcripts[key] = utterance
                attempt["t_stt"] = _ms(t)
                out.t_stt += attempt["t_stt"]
                logger.info(f"{device}: 📝 STT [{_desc(rt, 'stt', chain.stt)}] chain "
                            f"{chain.name!r} ({attempt['t_stt'] / 1000:.2f}s): {utterance!r}")
                # A narrowed Vosk transcript of nothing but out-of-grammar "[unk]"
                # tokens is noise or speech outside the vocabulary: no words to hand on.
                if (not utterance.replace("[unk]", "").strip()
                        or contains_stt_hallucination(utterance)):
                    attempt["outcome"] = "empty"
                    continue
                if not heard and gate is not None and not await gate(utterance):
                    attempt.update(text=utterance, outcome="rejected")
                    break
                if not heard and on_text is not None:
                    on_text(utterance, out.t_stt)
                heard = True
            attempt["text"] = utterance
            out.text = utterance

            cfg = rt.svc.instance_config("llm", chain.handler)
            t = time.perf_counter()
            result = declined = failure = None
            try:
                backend = _backend(rt, "llm", chain.handler)
                logger.info(f"{device}: 🤖 → LLM [{_desc(rt, 'llm', chain.handler)}] "
                            f"chain {chain.name!r}: {utterance!r}")
                result = await llm.LlmStage(backend, rt.hub, cfg).respond(
                    request(utterance), on_filler=on_filler, has_next=i < len(active) - 1)
            except Declined as d:
                declined = d
            except StageError as e:
                failure = e
            attempt["t_handler"] = _ms(t)
            out.t_llm += attempt["t_handler"]
            if declined is not None:
                attempt.update(outcome="declined", detail=declined.reason or None)
                out.reply, out.partial = declined.reply, {}
                logger.info(f"{device}: ↪️ chain {chain.name!r} declined: {declined.reply!r}")
                continue
            if failure is not None:
                partial = getattr(failure, "partial", None) or {}
                attempt.update(outcome="handler_error", error=str(failure))
                out.error, out.partial = failure, partial
                logger.error(f"{device}: LLM chain {chain.name!r} failed: {failure}")
                if partial.get("tool_used"):
                    # The action already happened: stop here rather than let a later
                    # chain repeat it.
                    out.chain, out.reply = chain.name, cfg.reply_error_after_tools
                    break
                out.reply = spoken_fallback(failure, cfg)
                continue
            attempt["outcome"] = "answered"
            out.chain, out.reply, out.result = chain.name, result.reply, result
            out.answered, out.error = True, None
            logger.info(f"{device}: 💬 LLM reply chain {chain.name!r} "
                        f"({attempt['t_handler'] / 1000:.2f}s): {result.reply!r}")
            break
    finally:
        # Parallel chains never reached: drop their STT and record them as cancelled.
        for task in early.values():
            task.cancel()
        if early:
            await asyncio.gather(*early.values(), return_exceptions=True)
        for chain in active:
            if chain.name in early:
                out.attempts.append({**_attempt(chain, voice=voice), "outcome": "cancelled"})
    return out
