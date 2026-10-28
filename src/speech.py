"""Speech output of a device's pipeline: the TTS variants, the playback on the device,
the copy of an answer's stream kept for the run log, the end-of-phrase ack chime and
the early filler line.

SpeechMixin is mixed into Pipeline (src/pipeline.py), one instance per device. The host
class calls _init_speech() from its __init__ and provides: name, rt (the Runtime: svc,
tts_backends), core, stress_backend, has_speaker, play (the transport hook), _emit,
_emit_live, _backend_desc and _close_followup.
"""

import asyncio
import os
import time

from loguru import logger

from src.audio_prep import pcm_to_wav_bytes
from src.chime import build_ack_clip
from src.pipeline_events import StageEvent
from src.stage_errors import StageError
from src.tts import TtsBackend, pcm_rate


def _end_sentence(text: str) -> str:
    """`text` ending with a sentence-final mark. Without one the TTS engines read the
    end as unfinished and give it the wrong intonation, so a '.' is added — after any
    closing quote or bracket, and not to an empty text."""
    text = text.rstrip()
    core = text.rstrip("»\"”')]")
    if core and core[-1] not in ".!?…":
        text += "."
    return text


async def _synthesize_stream(backend, text: str, lang: str = "ru"):
    """Resolve `backend`'s streaming synthesis entry point.

    Backends on the TtsBackend contract always have synthesize_stream (the
    ABC ships a buffered single-chunk default); a duck-typed backend that
    only exposes synthesize() is routed through that same default adapter,
    so the streaming path never hard-requires the method. Takes the backend
    explicitly so callers keep their read-the-property-once pattern."""
    fn = getattr(backend, "synthesize_stream", None)
    if fn is None:
        return await TtsBackend.synthesize_stream(backend, text, lang)
    return await fn(text, lang)


async def _close_stream(chunks) -> None:
    """Best-effort aclose() of a chunk iterator that will never be (further)
    iterated. Guarded: the default buffered TtsBackend adapter and the
    streaming providers return async generators (which have aclose), but a
    plain duck-typed iterable may not. A close failure is logged, never
    raised, so it can't mask the original error on the failure paths."""
    aclose = getattr(chunks, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as e:  # noqa: BLE001 - closing is best-effort
        logger.debug(f"TTS chunk iterator close failed: {e!r}")


async def _tee_chunks(chunks, sink):
    """Pass-through wrapper over a TTS chunk async-iterator that also appends each
    chunk to `sink`, so _voice keeps a copy of the full streamed clip to store as an
    answer's TTS audio. On early close (aclose), the underlying iterator is closed
    too, so the backend's open response is always released."""
    try:
        async for chunk in chunks:
            sink.append(chunk)
            yield chunk
    finally:
        await _close_stream(chunks)


class _PrefetchedChunks:
    """A TTS chunk stream whose first chunk was already pulled: iterating yields that
    chunk, then the rest of the stream. aclose() closes the underlying stream even
    before the first iteration, so the backend's open response is always released."""

    def __init__(self, first, rest):
        self._first = first
        self._rest = rest

    @classmethod
    async def start(cls, chunks):
        """Pull the first chunk of `chunks` (an empty stream yields nothing). A
        failure while pulling it closes the stream and propagates."""
        rest = chunks.__aiter__()
        try:
            first = await anext(rest, None)
        except BaseException:
            await _close_stream(rest)
            raise
        return cls(first, rest)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._first is not None:
            chunk, self._first = self._first, None
            return chunk
        return await anext(self._rest)

    async def aclose(self):
        await _close_stream(self._rest)


class SpeechMixin:
    """The speech output of a Pipeline (see the module docstring)."""

    def _init_speech(self) -> None:
        # Speeches not yet played out (see _voice()).
        self._speaking = 0

        # Background "filler" play tasks (early "I'll go check it" lines). Kept in a
        # set so a still-playing filler isn't garbage-collected mid-flight; each task
        # removes itself on completion. Fire-and-forget — never awaited by the run.
        self._filler_tasks = set()

        # Background end-of-phrase "ack" chime tasks (the server-side «блям» played once
        # the phrase is accepted, see Pipeline._run). Tracked in a set for the same GC
        # reason as the fillers — fire-and-forget, never awaited by the run so STT isn't
        # delayed.
        self._ack_tasks = set()
        # Lazily-built cache of the ack clip: (source_key, mime, audio_bytes). The
        # generated chime is synthesized once; a configured sound_path is loaded and
        # re-loaded only when the path changes (the cache key is the resolved source).
        self._ack_clip: tuple[str, str, bytes] | None = None

    async def _start_tts(self, text: str, attempts: list | None = None):
        """Start synthesizing `text` on the first enabled TTS variant that starts.

        The enabled variants (tts.variants) are tried in config order. Each one's
        stream is opened and its FIRST chunk pulled, so a provider that fails before
        yielding any audio is passed over before anything reaches the device. Once a
        variant has started nothing switches: a later failure (mid-stream, playback)
        is the caller's. Appends one {"kind": "tts", "provider", "outcome": "spoke" |
        "error", "error", "t"} entry (t in ms) per variant tried to `attempts`.

        Returns (backend, mime, audio): `audio` is a chunk iterator yielding the
        pre-fetched chunk and then the rest, which the caller hands to play() (and
        closes). Raises the last variant's error when none started.
        """
        error = None
        for pid in self.rt.svc.tts_variants():
            backend = self.rt.tts_backends.get(pid)
            t = time.perf_counter()
            try:
                if backend is None:
                    raise StageError("tts", f"tts backend {pid!r} is not loaded")
                mime, chunks = await _synthesize_stream(backend, text, "ru")
                audio = await _PrefetchedChunks.start(chunks)
            except Exception as e:
                error = e
                if attempts is not None:
                    attempts.append({"kind": "tts", "provider": pid, "outcome": "error",
                                     "error": str(e), "t": int((time.perf_counter() - t) * 1000)})
                desc = self._backend_desc(backend) if backend is not None else pid
                logger.error(f"{self.name}: TTS [{desc}] variant {pid!r} failed to start: {e}")
                continue
            if attempts is not None:
                attempts.append({"kind": "tts", "provider": pid, "outcome": "spoke",
                                 "error": None, "t": int((time.perf_counter() - t) * 1000)})
            return backend, mime, audio
        raise error

    async def speak_reply(self, text: str, record: dict, *, accent: bool = True,
                          answer: bool = False):
        """Speak `text` as a run's reply and fill the run's log `record` with how it
        went: the one way a voice run's answer, a text request's reply and an
        announcement are spoken.

        The stress stage first (`accent`): '+' stress marks on the text so every TTS
        backend pronounces Russian correctly. RuAccent's output IS the canonical
        '+vowel' LLM->TTS contract, so the per-backend adaptation handles it unchanged.
        It records t_stress, and stress_text only when it changed the text (so the
        panel shows no no-op section); record["llm_text"] keeps the original reply.
        Isolated: a failure falls back to the unmarked text. Then the speech (_voice):
        the TTS variants until one starts, each in the record's attempts, and the
        playback (play_dry, play_gap_ms). t_tts is the time from the first TTS attempt
        to the start of the stream — the user-perceived TTS latency, the device plays
        while synthesis continues; the variants are tried back to back, so their times
        add up to it.

        An `answer` is the reply of the device's voice session: the stress stage's
        result goes to the panel live, TTS_START shows the device the spoken text (also
        on a device without a speaker, which gets the text only), and its speech is
        kept (audio_bytes, audio_fmt and the returned stored audio, raw PCM as WAV so
        the panel can play it). Its t_tts stands once its stream started, even if the
        playback then fails or the run is cancelled; another reply's only once it has
        played.

        A failure of the speech marks the record — result "error", and error stage
        "TTS" unless an earlier stage owns it (a failed LLM's spoken phrase keeps the
        LLM root cause) — is logged and raised.

        Returns (played, tts_audio, tts_mime): what the play hook returned (None on a
        device without a speaker) and an answer's stored speech (None, None for another
        reply)."""
        if accent:
            stress = self.stress_backend
            if stress is not None:
                ra_t = time.perf_counter()
                accented = text
                accent_failed = False
                try:
                    accented = await stress.accentize(text)
                except Exception as e:
                    accent_failed = True
                    logger.error(f"{self.name}: accent stage [{self._backend_desc(stress)}] failed: {e}")
                record["t_stress"] = int((time.perf_counter() - ra_t) * 1000)
                if accented != text:
                    record["stress_text"] = accented
                text = accented
                # Logged on the success path only; a failure is logged above.
                if not accent_failed:
                    logger.info(
                        f"{self.name}: ✒️ Accent [{self._backend_desc(stress)}] "
                        f"({record['t_stress'] / 1000:.2f}s): {text!r}"
                    )
                if answer:
                    self._emit_live(record)

        def tts_ms() -> int:
            return sum(a["t"] for a in record["attempts"] if a["kind"] == "tts")

        tts_audio = tts_mime = None
        done = False
        try:
            if answer:
                self._emit(StageEvent.TTS_START, {"text": text})
            played, clip = await self._voice(text, record["attempts"], keep=answer)
            done = True
            if played is not None:
                record["play_dry"] = played["dry"]
                record["play_gap_ms"] = played["gap_ms"]
            if clip is not None:
                mime, audio = clip
                record["audio_bytes"] = len(audio)
                record["audio_fmt"] = mime.partition(";")[0].rpartition("/")[2]
                rate = pcm_rate(mime)
                if rate is not None:
                    tts_audio, tts_mime = pcm_to_wav_bytes(audio, rate=rate), "audio/wav"
                else:
                    tts_audio, tts_mime = audio, mime
        except Exception as e:
            record["result"] = "error"
            if record["error_stage"] is None:
                record["error_stage"] = "TTS"
                record["error_text"] = str(e)
            logger.error(f"{self.name}: TTS failed: {e}")
            raise
        finally:
            # Also when the playback fails or the run is cancelled mid-speech (a stop
            # word, a newer wake): an answer's t_tts stands once its stream started.
            if done or (answer and any(a["kind"] == "tts" and a["outcome"] == "spoke"
                                       for a in record["attempts"])):
                record["t_tts"] = tts_ms()
        return played, tts_audio, tts_mime

    async def _voice(self, text: str, attempts: list | None = None, keep: bool = False):
        """Synthesize `text`, ended with a sentence-final mark (_end_sentence), via the
        first TTS variant that starts (see _start_tts) and play it on the device,
        returning once it has played: every speech — a reply (speak_reply), a filler,
        proactive speech (speak). A device without a speaker gets no synthesis at all.
        `attempts` collects the variants tried. Raises on failure (callers decide
        isolation).

        Returns (played, clip): what the play hook returned ({"outcome", "dry",
        "gap_ms"}; None for a device without a speaker) and, when `keep`, the speech as
        it was streamed, (mime, audio bytes); else None.
        """
        if not self.has_speaker:
            return None, None
        text = _end_sentence(text)
        # Counted until it has played: a speech queued on the device behind an answer
        # keeps that answer from opening a follow-up window (the answer's own count is
        # released by the time it decides).
        self._speaking += 1
        try:
            await self._close_followup()
            t = time.perf_counter()
            # The backend of the variant that started both synthesizes and labels the
            # log line, so they can never diverge.
            backend, mime, audio = await self._start_tts(text, attempts)
            logger.info(f"{self.name}: 🔊 TTS [{self._backend_desc(backend)}] "
                        f"stream start ({time.perf_counter() - t:.2f}s): {text!r}")
            chunks = []
            try:
                played = await self.play(mime, _tee_chunks(audio, chunks) if keep else audio)
            finally:
                # The tee only closes `audio` once it has started.
                await _close_stream(audio)
            if not keep:
                return played, None
            clip = b"".join(chunks)
            logger.info(f"{self.name}: 🔊 TTS played "
                        f"({time.perf_counter() - t:.2f}s total, {len(clip)} bytes)")
            return played, (mime, clip)
        finally:
            self._speaking -= 1

    async def speak(self, text: str) -> dict | None:
        """Public entry for proactive speech without a run record (the filler,
        external callers): the speech of `text` (_voice). Raises on failure (callers
        decide isolation).

        Returns what the play hook does ({"outcome", "dry", "gap_ms"}); None for a
        device without a speaker.
        """
        played, _ = await self._voice(text)
        return played

    def _schedule_filler(self, text: str) -> None:
        """Speak the filler `text` in the background (_deliver_filler), never awaited by
        the run. Tracked in _filler_tasks so the still-playing filler isn't
        garbage-collected; each task removes itself."""
        task = asyncio.create_task(self._deliver_filler(text))
        self._filler_tasks.add(task)
        task.add_done_callback(self._filler_tasks.discard)

    async def _deliver_filler(self, text: str) -> None:
        """Speak an early 'filler' line via speak().

        It is queued on the device ahead of the reply, so the user hears a short
        'I'll go check it' line while the slow tool + final LLM round run, instead of
        waiting in silence. Fully isolated: a synthesis/play failure must NEVER affect
        the main run.
        """
        try:
            logger.info(f"{self.name}: 🗣️ filler: {text!r}")
            await self.speak(text)
        except Exception as e:
            logger.error(f"{self.name}: filler failed: {e}")

    def _ack_clip_bytes(self) -> tuple[str, bytes]:
        """Return (mime, audio) for the end-of-phrase ack clip, building/caching once.

        Source resolution (read live off core.ack so it hot-applies): a configured
        sound_path that exists on disk is loaded; otherwise the two-tone chime is
        synthesized. The result is cached keyed by the resolved source, so the file is
        read / transcoded / the chime built only once and re-resolved only when
        sound_path changes. See build_ack_clip for the per-source build details.
        """
        path = (self.core.ack.sound_path or "").strip()
        use_file = bool(path) and os.path.isfile(path)
        source_key = path if use_file else "<generated>"
        cached = self._ack_clip
        if cached is not None and cached[0] == source_key:
            return cached[1], cached[2]
        mime, audio = build_ack_clip(path, name=self.name)
        self._ack_clip = (source_key, mime, audio)
        return mime, audio

    def _schedule_ack(self) -> None:
        """Schedule the fire-and-forget end-of-phrase ack chime, if enabled.

        Synchronous gate (no await before the decision): enabled toggle + a device
        speaker. Building the clip and the play call run inside the background task
        so STT is never delayed. Tracked in _ack_tasks so the still-playing chime
        isn't garbage-collected; each task removes itself.
        """
        if not self.core.ack.enabled or not self.has_speaker:
            return
        task = asyncio.create_task(self._play_ack())
        self._ack_tasks.add(task)
        task.add_done_callback(self._ack_tasks.discard)

    async def _play_ack(self) -> None:
        """Play the ack clip on the device. Fully isolated — a build/play failure must
        NEVER affect or delay the run (which keeps its own final TTS for the answer).
        """
        try:
            mime, audio = self._ack_clip_bytes()
            logger.info(f"{self.name}: 🔔 end-of-phrase ack")
            await self.play(mime, audio)
        except Exception as e:
            logger.error(f"{self.name}: end-of-phrase ack failed: {e}")
