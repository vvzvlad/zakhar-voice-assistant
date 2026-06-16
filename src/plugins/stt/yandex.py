"""Yandex SpeechKit STT v3 brick: config schema, streaming gRPC session and backend.

True streaming recognition over the SpeechKit STT v3 gRPC bidi stream
(`speechkit.stt.v3`): PCM chunks are fed live during speech and recognition runs
DURING the utterance for low latency. The pipeline VAD owns end-pointing — it
feed()s chunks, then finish() forces the final transcript via an external EOU
classifier (Eou()), so OUR VAD decides when speech ends, not Yandex.

The gRPC stack (`grpc`, `yandex.cloud.ai.stt.v3`) is imported LAZILY inside
methods so this module stays importable WITHOUT `yandexcloud` installed (it is
auto-imported by src/plugins/__init__.py at process start, and the test/CI venv
may not have the dependency).
"""

import asyncio
import contextlib

from loguru import logger
from pydantic import BaseModel, Field

from src.plugins.base import SECRET_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend, StreamingSttSession, StreamingTranscript

# v3 streaming recognizer gRPC endpoint; the same for every deployment, so it is
# hardcoded rather than configurable.
YANDEX_STT_ENDPOINT = "stt.api.cloud.yandex.net:443"

# Batch fallback feeds the whole PCM in <=32000-byte chunks (~1 s at 16 kHz/16-bit
# mono) so a single utterance never arrives as one oversized AudioChunk.
_BATCH_CHUNK_BYTES = 32000

# Recognition models valid for RecognizeStreaming (SpeechKit STT v3). Static list
# (no runtime "list models" endpoint), kept in sync with the proto/docs by hand.
YANDEX_STT_MODELS = ["general", "general:rc", "general:deprecated"]

# Languages supported by the v3 "general" streaming model. "auto" enables Yandex's
# automatic language detection. Curated from the SpeechKit STT docs (kept in sync by
# hand, like the Yandex TTS voice catalog); ru-RU is the default. The language is
# passed as a WHITELIST hint (a guideline, not a strict rule), so the list does not
# need to be exhaustive.
YANDEX_STT_LANGUAGES = [
    "ru-RU", "en-US", "auto", "kk-KZ", "de-DE", "es-ES", "fi-FI", "fr-FR",
    "it-IT", "nl-NL", "pl-PL", "pt-PT", "pt-BR", "sv-SE", "tr-TR",
]


class YandexSttStream(StreamingSttSession):
    """Live SpeechKit STT v3 gRPC session.

    Chunks are queued by feed() (synchronous, hot-path safe) and drained by a
    background drive task that opens the bidi stream and accumulates partial /
    final / final_refinement events into a StreamingTranscript. finish() pushes a
    sentinel (which makes the request iterator emit the external Eou()), awaits
    the drive task, and returns the accumulated transcript.

    `transport` is a test injection seam: a callable (request_aiter, metadata) ->
    async-iterable-of-StreamingResponse. When None (production), the real gRPC
    channel/stub are built lazily here so the heavy dependency is only required
    when this backend is actually selected at runtime.
    """

    def __init__(self, *, api_key, model, language, normalize, timeout, transport=None):
        self.api_key = api_key
        self.model = model
        self.language = language
        self.normalize = normalize
        self.timeout = timeout

        # The proto module is needed by the request iterator to build messages; it
        # is imported lazily here. Tests inject `transport` but still run with
        # yandexcloud installed, so stt_pb2 is always importable in that path.
        from yandex.cloud.ai.stt.v3 import stt_pb2

        self._stt_pb2 = stt_pb2

        self._channel = None
        if transport is None:
            # Production: build the real secure channel + recognizer stub lazily.
            import grpc
            from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc

            self._channel = grpc.aio.secure_channel(
                YANDEX_STT_ENDPOINT, grpc.ssl_channel_credentials()
            )
            stub = stt_service_pb2_grpc.RecognizerStub(self._channel)
            transport = stub.RecognizeStreaming
        self._transport = transport

        # Auth metadata: Api-Key auth only (service-account key); the folder is
        # scoped by the key, so no x-folder-id is sent.
        self._metadata = (("authorization", f"Api-Key {self.api_key}"),)

        self._queue: asyncio.Queue = asyncio.Queue()
        self._accumulator = StreamingTranscript()
        self._error: Exception | None = None
        self._closed = False
        self._stopped = False  # set once the drive task ends (normally or via error/cancel)
        # open_stream() is called from on_start, which runs on the event loop, so
        # creating the drive task here is valid.
        self._drive_task = asyncio.create_task(self._drive())

    def _session_options(self):
        """Build the StreamingOptions sent as the first request: raw 16 kHz/16-bit
        mono PCM, language whitelist, optional text normalization, REAL_TIME model
        (low-latency partials) and an EXTERNAL EOU classifier so our pipeline VAD
        owns end-pointing (finish() emits Eou())."""
        stt_pb2 = self._stt_pb2
        return stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                model=self.model or "general",
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=16000,
                        audio_channel_count=1,
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=(
                        stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED
                        if self.normalize
                        else stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_DISABLED
                    ),
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[self.language],
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
            ),
            eou_classifier=stt_pb2.EouClassifierOptions(
                external_classifier=stt_pb2.ExternalEouClassifier()
            ),
        )

    async def _request_iter(self):
        """Yield the request stream: session_options first (required), then one
        AudioChunk per queued PCM block, then a forced Eou() once the sentinel
        arrives (half-close)."""
        stt_pb2 = self._stt_pb2
        yield stt_pb2.StreamingRequest(session_options=self._session_options())
        while True:
            item = await self._queue.get()
            if item is None:
                break  # finish() sentinel -> half-close after forcing EOU
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=item))
        yield stt_pb2.StreamingRequest(eou=stt_pb2.Eou())

    async def _drive(self):
        """Open the bidi stream and accumulate events live. Any error is stored on
        self._error (never raised out of the task) so finish() can convert it to a
        StageError while still exposing a best-effort partial on timeout."""
        try:
            call = self._transport(self._request_iter(), metadata=self._metadata)
            async for resp in call:
                event = resp.WhichOneof("Event")
                if event == "partial":
                    self._accumulator.add_partial(self._alt_text(resp.partial))
                elif event == "final":
                    self._accumulator.add_final(
                        resp.audio_cursors.final_index, self._alt_text(resp.final)
                    )
                elif event == "final_refinement":
                    # Normalized text REPLACES the raw final at the same index.
                    self._accumulator.add_normalized(
                        resp.final_refinement.final_index,
                        self._alt_text(resp.final_refinement.normalized_text),
                    )
                # "eou_update", "status_code" and anything else: ignored.
        except Exception as e:  # noqa: BLE001 - surfaced via self._error in finish()
            self._error = e
        finally:
            # The drive task has ended (normally, via error, or cancel): stop feed()
            # from enqueuing chunks nobody will read for the rest of the utterance.
            self._stopped = True

    @staticmethod
    def _alt_text(update) -> str:
        """First alternative's text from an AlternativeUpdate, or "" when empty."""
        alts = update.alternatives
        return alts[0].text if alts else ""

    def feed(self, pcm: bytes) -> None:
        # Hot audio path: synchronous, non-blocking, never raises.
        if self._closed or self._stopped or not pcm:
            return
        try:
            self._queue.put_nowait(pcm)
        except Exception:  # noqa: BLE001 - feed() must never raise into on_audio
            pass

    async def finish(self) -> str:
        if self._closed:
            return self._accumulator.result()
        # Push the sentinel: the request iterator forces the external Eou() and
        # half-closes, so the server emits the final(s) and completes the stream.
        self._queue.put_nowait(None)
        try:
            try:
                await asyncio.wait_for(self._drive_task, timeout=self.timeout)
            except asyncio.TimeoutError:
                # The server is slow to finalize: cancel the drive task and return a
                # best-effort partial if we have one, else surface the timeout.
                self._drive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._drive_task
                best = self._accumulator.result()
                if best:
                    return best
                raise StageError("stt", "Yandex STT v3 timed out")
            if self._error is not None:
                raise StageError("stt", f"Yandex STT v3 failed: {self._error}")
            return self._accumulator.result()
        finally:
            self._closed = True
            if self._channel is not None:
                with contextlib.suppress(Exception):
                    await self._channel.close()

    async def aclose(self) -> None:
        # Idempotent abandon: never raises (defensive cleanup path).
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        if self._drive_task is not None:
            self._drive_task.cancel()
            # CancelledError is a BaseException (not Exception) in 3.8+, so suppress
            # it explicitly alongside any other cleanup error.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drive_task
        if self._channel is not None:
            with contextlib.suppress(Exception):
                await self._channel.close()


class YandexSttBackend(SttBackend):
    """SpeechKit STT v3 backend. open_stream() drives the live gRPC recognizer;
    transcribe() is a batch fallback used only when streaming wasn't started
    (e.g. open_stream() failed)."""

    def __init__(self, *, api_key, model, language, normalize, timeout, transport=None):
        if not api_key:
            raise ValueError(
                "Yandex STT api_key is required (set stt.instances.yandex.api_key in data/config.json)"
            )
        self.api_key = api_key
        self.model = model
        self.language = language
        self.normalize = normalize
        self.timeout = timeout
        # Test injection seam: when set, every session uses this fake transport
        # instead of a real gRPC stub (see YandexSttStream.transport).
        self._transport = transport

    def open_stream(self):
        return YandexSttStream(
            api_key=self.api_key,
            model=self.model,
            language=self.language,
            normalize=self.normalize,
            timeout=self.timeout,
            transport=self._transport,
        )

    async def transcribe(self, pcm: bytes) -> str:
        """Batch fallback: open a streaming session, feed the whole PCM in bounded
        chunks, force the final and return the text. Used only when the live
        streaming path wasn't taken; honors the SttBackend contract."""
        if not pcm:
            return ""
        session = self.open_stream()
        try:
            for i in range(0, len(pcm), _BATCH_CHUNK_BYTES):
                session.feed(pcm[i:i + _BATCH_CHUNK_BYTES])
            return await session.finish()
        except StageError:
            raise
        except Exception as e:
            logger.error(f"Yandex STT v3 batch transcribe failed: {e}")
            raise StageError("stt", f"Yandex STT v3 failed: {e}") from e
        finally:
            await session.aclose()


class YandexSttConfig(BaseModel):
    api_key: str = Field("", json_schema_extra=SECRET_FIELD_EXTRA)
    model: str = Field("general", json_schema_extra={"widget": "select", "options": "dynamic"})
    language: str = Field("ru-RU", json_schema_extra={"widget": "select", "options": "dynamic"})
    normalize: bool = Field(
        True,
        title="Text normalization",
        description="Convert spoken numbers, dates and similar to their numeric form.",
    )
    timeout: int = 30


@register
class YandexSttProvider(Provider):
    category = "stt"
    id = "yandex"
    label = "Yandex SpeechKit"
    ConfigModel = YandexSttConfig
    uses_http_cloud = False

    def create(self, cfg: YandexSttConfig, deps: Deps):
        return YandexSttBackend(
            api_key=cfg.api_key,
            model=cfg.model,
            language=cfg.language,
            normalize=cfg.normalize,
            timeout=cfg.timeout,
        )

    def options(self, field: str, cfg: YandexSttConfig, deps: Deps, query: str = ""):
        if field == "model":
            return list(YANDEX_STT_MODELS)
        if field == "language":
            return list(YANDEX_STT_LANGUAGES)
        return None
