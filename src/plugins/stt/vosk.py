"""Vosk offline STT brick: config schema and in-process backend."""

import asyncio
import contextlib
import json
import os
from collections.abc import Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.logging_setup import capture_native_stderr
from src.plugins.base import LOCAL_MODEL_FIELD_EXTRA, Deps, Provider, register
from src.stage_errors import StageError
from src.stt import SttBackend, StreamingSttSession, StreamingTranscript


def _strip_unk(text: str) -> str:
    """Drop the Vosk out-of-grammar "[unk]" sentinel tokens so they never reach NLU."""
    return " ".join(t for t in text.split() if t != "[unk]")


class VoskStreamSession(StreamingSttSession):
    """Live Vosk recognition: PCM chunks are decoded DURING speech (the pipeline VAD
    owns end-pointing). feed() enqueues chunks (sync, hot-path safe); a background
    drive task drains the queue and runs the BLOCKING KaldiRecognizer.AcceptWaveform
    in a worker thread, accumulating Vosk segment finals / partials into a
    StreamingTranscript. finish() pushes a sentinel, awaits the drive task to drain
    the backlog, flushes FinalResult(), and returns the transcript.

    The recognizer is owned exclusively by this session and only ever touched from
    the drive task (and from finish() AFTER the drive task has ended), so there is
    never concurrent access to it."""

    def __init__(self, recognizer):
        self._rec = recognizer
        self._queue: asyncio.Queue = asyncio.Queue()
        self._accumulator = StreamingTranscript()
        self._final_index = 0
        self._error: Exception | None = None
        self._closed = False
        self._stopped = False  # set once the drive task ends (normally/error/cancel)
        # open_stream() is called from on_start/_wakeword_confirm, which run on the
        # event loop, so creating the drive task here is valid (same as Yandex).
        self._drive_task = asyncio.create_task(self._drive())

    def _decode_chunk(self, chunk: bytes) -> tuple[bool, str]:
        """Blocking decode of one chunk inside a worker thread. Returns
        (is_final, text): is_final True with the segment text when Vosk hits an
        internal endpoint, else False with the running partial. ALL recognizer
        access stays inside the worker thread (never on the loop)."""
        if self._rec.AcceptWaveform(chunk):
            return (True, json.loads(self._rec.Result()).get("text", ""))
        return (False, json.loads(self._rec.PartialResult()).get("partial", ""))

    def _final_text(self) -> str:
        """Flush the trailing segment via FinalResult() (blocking; worker thread)."""
        return json.loads(self._rec.FinalResult()).get("text", "")

    async def _drive(self):
        """Drain the queue, decoding each chunk in a worker thread and accumulating
        results. Any error is stored on self._error (never raised out of the task) so
        finish() converts it to a StageError. Mirrors YandexSttStream._drive."""
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break  # finish()/aclose() sentinel
                is_final, text = await asyncio.to_thread(self._decode_chunk, chunk)
                if is_final:
                    self._accumulator.add_final(self._final_index, _strip_unk(text))
                    self._final_index += 1
                else:
                    self._accumulator.add_partial(_strip_unk(text))
        except Exception as e:  # noqa: BLE001 - surfaced via self._error in finish()
            self._error = e
        finally:
            self._stopped = True

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
        # Push the sentinel so the drive task drains the remaining backlog and stops.
        self._queue.put_nowait(None)
        try:
            await self._drive_task
            if self._error is not None:
                raise StageError("stt", f"Vosk STT decode failed: {self._error}")
            # Flush the trailing segment (drive task has ended -> recognizer is free).
            final = _strip_unk(await asyncio.to_thread(self._final_text))
            self._accumulator.add_final(self._final_index, final)
            return self._accumulator.result()
        finally:
            self._closed = True

    async def aclose(self) -> None:
        # Idempotent abandon: never raises (defensive cleanup path).
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        if self._drive_task is not None:
            self._drive_task.cancel()
            # CancelledError is a BaseException, so suppress it explicitly.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._drive_task


class VoskSttBackend(SttBackend):
    """In-process Vosk backend (small offline model, 16 kHz mono CPU decode).

    The model is loaded once and shared; KaldiRecognizer instances are spawned
    per call (the model is thread-safe for spawning recognizers). Decoding is
    blocking C code, so it runs in a worker thread.

    Optionally grammar-restricted: when a live command-vocabulary accessor is
    supplied, each per-call recognizer is constrained to that closed vocabulary —
    read fresh per decode so NLU/LLM-selection changes take effect with no rebuild.
    """

    def __init__(self, model_path: str, *, model=None, vocabulary: "Callable[[], list[str]] | None" = None):
        # The model is injectable for testing. When None (production path), it is
        # loaded lazily so the heavy dependency/model are only required when the
        # Vosk backend is actually selected at runtime (never in tests/CI).
        if model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            # Capture the native WARN/ERR Kaldi can emit during the model load
            # into loguru (minimal window — only the load itself).
            with capture_native_stderr("vosk-stt"):
                model = Model(model_path)  # fail fast if the dir is missing
        self._model = model
        # Live closed-vocabulary accessor (or None). Read per decode, never cached.
        self._vocabulary = vocabulary

    def _current_vocab(self) -> list[str]:
        """Read the live closed command vocabulary (or [] if none / on any error).
        Called per decode so NLU edits and LLM-provider switches take effect with no
        backend rebuild. A vocabulary error must never break STT — fall back to full
        vocabulary."""
        if self._vocabulary is None:
            return []
        try:
            return list(self._vocabulary() or [])
        except Exception as e:
            logger.warning(f"Vosk STT vocabulary read failed; using full vocabulary: {e}")
            return []

    def _make_recognizer(self):
        """Build a KaldiRecognizer for the shared model. When a live command
        vocabulary is present, constrain the recognizer to that grammar (words plus
        the Vosk "[unk]" sentinel) — closed-vocabulary decode, much faster/more
        accurate for the fixed command set; otherwise a plain full-vocab recognizer.
        Overridable in tests."""
        from vosk import KaldiRecognizer

        # NO fd-2 capture here: this runs per transcribe in a worker thread, and
        # redirecting process-global fd 2 around every decode would swallow loguru
        # lines emitted by other threads. The model load in __init__ is the rare
        # window that's wrapped instead.
        words = self._current_vocab()
        if words:
            grammar = json.dumps(list(words) + ["[unk]"], ensure_ascii=False)
            return KaldiRecognizer(self._model, 16000, grammar)
        return KaldiRecognizer(self._model, 16000)

    def _decode(self, pcm: bytes) -> str:
        rec = self._make_recognizer()
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        text = json.loads(rec.FinalResult()).get("text", "").strip()
        # Drop the Vosk out-of-grammar sentinel so "[unk]" never reaches the NLU.
        return _strip_unk(text)

    def open_stream(self):
        """Open a live streaming session: build a fresh recognizer (reads the live
        command vocabulary once, like the batch path) and hand it to the session.
        Returning a session routes the pipeline through feed()/finish() instead of
        batch transcribe(). A build failure propagates to the pipeline, which logs
        and falls back to batch."""
        rec = self._make_recognizer()
        rec.SetWords(False)
        return VoskStreamSession(rec)

    async def transcribe(self, pcm: bytes) -> str:
        if not pcm:
            return ""
        # Speaker PCM is already 16 kHz mono 16-bit — no resampling needed.
        try:
            return await asyncio.to_thread(self._decode, pcm)
        except StageError:
            raise
        except Exception as e:
            # Honor the SttBackend contract: any decode failure (native Vosk error,
            # malformed recognizer JSON, ...) surfaces as StageError("stt", ...) so
            # the pipeline records the run as an STT error instead of dying raw.
            logger.error(f"Vosk STT decode failed: {str(e)}")
            raise StageError("stt", f"Vosk STT decode failed: {e}") from e


class VoskSttConfig(BaseModel):
    model_path: str = Field("models/vosk-model-small-ru-0.22", json_schema_extra=LOCAL_MODEL_FIELD_EXTRA)
    restrict_to_nlu: bool = Field(
        True,
        title="Restrict to Simple NLU vocabulary",
        description="When the active intent engine is Simple NLU, constrain Vosk "
        "recognition to its command vocabulary (alias phrases + action verbs + number "
        "words) via a Vosk grammar — big accuracy/speed win for the fixed command set. "
        "Automatically has NO effect when a free-form LLM is selected (the full model "
        "vocabulary is used then).",
    )


def _list_vosk_models(base_dir: str) -> list[dict]:
    """Scan base_dir for Vosk model directories: immediate subdirs that contain
    both an `am` and a `conf` subdirectory (the structural markers of a standard
    Vosk model). Returns [{"value": <dir path>, "label": <dir name>}, ...] sorted
    by label (case-insensitive). Any filesystem error yields an empty list."""
    try:
        names = os.listdir(base_dir)
    except OSError:
        return []
    out = []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(base_dir, name)
        if (
            os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "am"))
            and os.path.isdir(os.path.join(path, "conf"))
        ):
            out.append({"value": path, "label": name})
    out.sort(key=lambda o: o["label"].lower())
    return out


@register
class VoskSttProvider(Provider):
    category = "stt"
    id = "vosk"
    label = "Vosk (offline)"
    ConfigModel = VoskSttConfig

    def create(self, cfg: VoskSttConfig, deps: Deps):
        vocab = deps.command_vocabulary if cfg.restrict_to_nlu else None
        return VoskSttBackend(cfg.model_path, vocabulary=vocab)

    def options(self, field: str, cfg: VoskSttConfig, deps: Deps, query: str = ""):
        # Local-disk scan of installed Vosk models next to the configured path
        # (defaults to models/). Synchronous: return a plain list. The model path
        # is a directory; strip any trailing slash before dirname() so a value like
        # "models/foo/" scans "models" (its siblings) rather than descending into it.
        if field == "model_path":
            return _list_vosk_models(os.path.dirname(cfg.model_path.rstrip("/")) or "models")
        return None
