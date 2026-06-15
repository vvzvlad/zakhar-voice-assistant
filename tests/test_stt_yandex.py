"""Tests for the Yandex SpeechKit STT v3 streaming provider.

Two layers:
- Always-run: the pure StreamingTranscript accumulator + config/backend wiring
  (no grpc/yandexcloud needed).
- Session tests: drive YandexSttStream with a FAKE transport built from REAL
  stt_pb2.StreamingResponse messages — skipped when yandexcloud is absent.
"""

import pytest

import src.plugins  # noqa: F401  register all providers
from src.plugins.base import Deps, get_provider
from src.plugins.stt.yandex import YandexSttBackend, YandexSttConfig
from src.stage_errors import StageError
from src.stt import StreamingTranscript

# --- Always-run: StreamingTranscript accumulator -------------------------------


def test_streaming_transcript_partial_only():
    # No final ever arrived -> result falls back to the last partial.
    t = StreamingTranscript()
    t.add_partial("привет")
    t.add_partial("привет мир")
    assert t.result() == "привет мир"


def test_streaming_transcript_finals_joined_in_index_order():
    # Finals are keyed by index and joined in ascending index order, NOT insertion.
    t = StreamingTranscript()
    t.add_final(1, "мир")
    t.add_final(0, "привет")
    assert t.result() == "привет мир"


def test_streaming_transcript_normalized_overrides_raw_final():
    # final_refinement (normalized) replaces the raw final at the same index.
    t = StreamingTranscript()
    t.add_final(0, "двадцать пять")
    t.add_normalized(0, "25")
    assert t.result() == "25"


def test_streaming_transcript_empty_final_falls_back_to_partial():
    # An EMPTY final must not clobber a valid accumulated partial: Yandex v3 can
    # emit a final with empty/whitespace alternatives. The joined finals are empty,
    # so result() falls back to the last partial instead of returning "".
    t = StreamingTranscript()
    t.add_partial("привет мир")
    t.add_final(0, "")
    assert t.result() == "привет мир"


def test_streaming_transcript_whitespace_final_falls_back_to_partial():
    # Same as above, but the final is whitespace-only (still empty after strip()).
    t = StreamingTranscript()
    t.add_partial("привет мир")
    t.add_final(0, "   ")
    assert t.result() == "привет мир"


def test_streaming_transcript_nonempty_final_wins_over_partial():
    # A non-empty final still takes precedence over any accumulated partial.
    t = StreamingTranscript()
    t.add_partial("прив")
    t.add_final(0, "привет мир")
    assert t.result() == "привет мир"


def test_streaming_transcript_no_events_is_empty():
    assert StreamingTranscript().result() == ""


def test_streaming_transcript_empty_partial_is_ignored():
    # An empty partial must not clobber a previously captured one.
    t = StreamingTranscript()
    t.add_partial("привет")
    t.add_partial("")
    assert t.result() == "привет"


# --- Always-run: config defaults + backend wiring ------------------------------


def test_yandex_stt_config_defaults():
    cfg = YandexSttConfig()
    assert cfg.model == "general"
    assert cfg.language == "ru-RU"
    assert cfg.normalize is True
    assert cfg.folder_id == ""
    assert cfg.timeout == 30


def test_yandex_stt_backend_requires_api_key():
    with pytest.raises(ValueError):
        YandexSttBackend(
            api_key="", folder_id="", model="general", language="ru-RU",
            normalize=True, timeout=30,
        )


async def test_registry_yandex_stt_provider_creates_backend():
    # REGISTRY-based construction: the yandex STT provider's create() returns a
    # YandexSttBackend (no cloud HTTP client — it speaks gRPC directly).
    import httpx

    async with httpx.AsyncClient(verify=False) as cloud, httpx.AsyncClient(verify=False) as local:
        deps = Deps(http_cloud=cloud, http_local=local)
        backend = get_provider("stt", "yandex").create(
            YandexSttConfig(api_key="k"), deps
        )
        assert isinstance(backend, YandexSttBackend)
        assert backend.api_key == "k"
        assert backend.model == "general"
        assert backend.language == "ru-RU"


# --- Session tests: fake transport over REAL proto messages --------------------

# The session tests build REAL proto messages and drive YandexSttStream with a
# fake transport, so they need yandexcloud installed. The always-run tests above
# do NOT, so this skip gates only the session class below (a module-level
# importorskip would skip the whole file, including the pure-accumulator tests).
try:
    from yandex.cloud.ai.stt.v3 import stt_pb2  # noqa: E402

    from src.plugins.stt.yandex import YandexSttStream  # noqa: E402

    _HAVE_YANDEXCLOUD = True
except ImportError:  # pragma: no cover - exercised only when the dep is absent
    stt_pb2 = None
    _HAVE_YANDEXCLOUD = False

_requires_yandexcloud = pytest.mark.skipif(
    not _HAVE_YANDEXCLOUD, reason="yandexcloud (gRPC stubs) not installed"
)


def _partial_response(text: str):
    resp = stt_pb2.StreamingResponse()
    resp.partial.alternatives.add().text = text
    return resp


def _final_response(index: int, text: str):
    resp = stt_pb2.StreamingResponse()
    resp.final.alternatives.add().text = text
    resp.audio_cursors.final_index = index
    return resp


def _refinement_response(index: int, text: str):
    resp = stt_pb2.StreamingResponse()
    resp.final_refinement.normalized_text.alternatives.add().text = text
    resp.final_refinement.final_index = index
    return resp


def _make_transport(responses, *, sink=None, raises=None):
    """Build a fake transport (request_aiter, metadata) -> async iterator.

    When `sink` is provided, every consumed request is appended to it (so a test
    can assert the request ordering: session_options, chunk*, eou). When `raises`
    is set, the transport raises it instead of yielding responses.
    """

    def transport(request_aiter, metadata=None):
        async def _gen():
            # Consume the request stream so its generator runs to completion (and
            # the test can inspect ordering via `sink`).
            async for req in request_aiter:
                if sink is not None:
                    sink.append(req.WhichOneof("Event"))
            if raises is not None:
                raise raises
            for resp in responses:
                yield resp

        return _gen()

    return transport


def _new_stream(transport, *, timeout=5):
    return YandexSttStream(
        api_key="k", folder_id="", model="general", language="ru-RU",
        normalize=True, timeout=timeout, transport=transport,
    )


@_requires_yandexcloud
async def test_stream_request_ordering_session_options_chunks_eou():
    # The first request MUST be session_options, followed by one chunk per feed(),
    # then a trailing eou (forced end-of-utterance via the external classifier).
    sink: list[str] = []
    transport = _make_transport([_final_response(0, "привет")], sink=sink)
    stream = _new_stream(transport)
    stream.feed(b"\x01\x02" * 100)
    stream.feed(b"\x03\x04" * 100)
    text = await stream.finish()
    assert text == "привет"
    assert sink[0] == "session_options"
    assert sink[-1] == "eou"
    assert sink.count("chunk") == 2


@_requires_yandexcloud
async def test_stream_returns_final_text():
    transport = _make_transport([
        _partial_response("прив"),
        _final_response(0, "привет мир"),
    ])
    stream = _new_stream(transport)
    stream.feed(b"\x01\x02" * 100)
    assert await stream.finish() == "привет мир"


@_requires_yandexcloud
async def test_stream_normalization_override():
    # A final_refinement at the same index replaces the raw final's text.
    transport = _make_transport([
        _final_response(0, "двадцать пять"),
        _refinement_response(0, "25"),
    ])
    stream = _new_stream(transport)
    stream.feed(b"\x01\x02" * 100)
    assert await stream.finish() == "25"


@_requires_yandexcloud
async def test_stream_finish_raises_stage_error_on_transport_error():
    transport = _make_transport([], raises=RuntimeError("grpc boom"))
    stream = _new_stream(transport)
    stream.feed(b"\x01\x02" * 100)
    with pytest.raises(StageError) as ei:
        await stream.finish()
    assert ei.value.stage == "stt"
    assert "boom" in str(ei.value)


@_requires_yandexcloud
async def test_stream_aclose_is_idempotent():
    transport = _make_transport([_final_response(0, "привет")])
    stream = _new_stream(transport)
    stream.feed(b"\x01\x02" * 100)
    await stream.aclose()
    # A second close (and a feed after close) must be safe no-ops.
    await stream.aclose()
    stream.feed(b"\x05\x06" * 100)


@_requires_yandexcloud
async def test_backend_transcribe_batch_path_returns_final_text():
    # The batch fallback opens a streaming session under the hood; inject the fake
    # transport through the backend so transcribe() returns the final text.
    transport = _make_transport([_final_response(0, "привет из батча")])
    backend = YandexSttBackend(
        api_key="k", folder_id="", model="general", language="ru-RU",
        normalize=True, timeout=5, transport=transport,
    )
    text = await backend.transcribe(b"\x01\x02" * 1000)
    assert text == "привет из батча"


@_requires_yandexcloud
async def test_backend_transcribe_empty_pcm_returns_empty():
    transport = _make_transport([_final_response(0, "unused")])
    backend = YandexSttBackend(
        api_key="k", folder_id="", model="general", language="ru-RU",
        normalize=True, timeout=5, transport=transport,
    )
    assert await backend.transcribe(b"") == ""
