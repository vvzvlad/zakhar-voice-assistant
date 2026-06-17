"""Route the Python standard ``logging`` module through loguru.

Third-party libraries (httpx, mcp, aioesphomeapi, ...) emit via stdlib
``logging`` and would otherwise bypass loguru entirely, printing raw lines that
do not match our unified ``timestamp | LEVEL | module:func:line - message``
format. Installing loguru's documented ``InterceptHandler`` on the root stdlib
logger forwards every record into loguru so the whole app shares one sink and
honours ``core.log_level``.
"""

import contextlib
import inspect
import logging
import os
import sys
import threading

from loguru import logger

# These loggers emit one INFO line per HTTP request / per MCP session, which
# would spam the unified log on every MCP tool call. Demote them to WARNING so
# routine chatter is silenced while warnings/errors still pass through.
_NOISY_LOGGERS = ("httpx", "httpcore", "mcp")

# fd-2 (stderr) redirection is process-global. Vosk/Kaldi decodes run in worker
# threads (asyncio.to_thread) that can overlap across devices, so a single global
# lock serializes the redirect window — two threads must never swap fd 2 at once.
_NATIVE_STDERR_LOCK = threading.Lock()


def _forward_native_line(line: str, prefix: str) -> None:
    """Forward one captured native-stderr line to loguru at the mapped level.

    Kaldi prefixes its messages with a level word ("WARNING", "ERROR",
    "ASSERTION", "LOG", ...). Map that onto a loguru level and prepend `prefix`
    so the routed line is attributable to the native component (e.g. ``vosk``).
    """
    text = line.rstrip("\n")
    if not text:
        return
    message = f"{prefix}: {text}"
    if text.startswith("WARNING"):
        logger.warning(message)
    elif text.startswith("ERROR") or text.startswith("ASSERTION"):
        logger.error(message)
    else:
        # Anything else (e.g. "LOG"/"VLOG") is verbose; keep it at DEBUG.
        logger.debug(message)


@contextlib.contextmanager
def capture_native_stderr(prefix: str):
    """Capture C-level stderr (fd 2) for a block and route it into loguru.

    Vosk/Kaldi native logging writes WARN/ERR lines straight to fd 2, bypassing
    loguru (``SetLogLevel(-1)`` only silences LOG/VLOG). This redirects fd 2 to a
    pipe for the duration of the block, drains the pipe in a daemon reader thread
    (so a full pipe buffer can never deadlock the native call), then restores fd 2
    and forwards each captured line to loguru via :func:`_forward_native_line`.

    Fully defensive: if any os-level call (pipe/dup/dup2) fails on a given
    platform/sandbox, it degrades to a plain pass-through (yields without
    redirecting) and never raises into the caller — routing native logs must
    never break the decode it wraps.
    """
    with _NATIVE_STDERR_LOCK:
        saved_fd = None
        read_fd = None
        write_fd = None
        reader = None
        chunks: list[bytes] = []
        try:
            # Save the real fd 2 so it can be restored, then point fd 2 at the
            # pipe's write end. Any failure here means we cannot redirect safely;
            # fall back to pass-through.
            saved_fd = os.dup(2)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 2)
        except Exception:
            # Redirection unsupported/failed — undo any partial state and yield
            # a plain pass-through so the caller is never affected.
            if saved_fd is not None:
                with contextlib.suppress(Exception):
                    os.dup2(saved_fd, 2)
                with contextlib.suppress(Exception):
                    os.close(saved_fd)
            for fd in (read_fd, write_fd):
                if fd is not None:
                    with contextlib.suppress(Exception):
                        os.close(fd)
            yield
            return

        def _drain() -> None:
            # Drain the pipe READ end until EOF (which arrives when fd 2 is
            # restored below, closing the last write end). Reading concurrently
            # with the block keeps a full pipe buffer from blocking the native
            # writer (and thus the decode).
            #
            # The reader thread has SOLE ownership of the READ fd: it wraps it in
            # an os.fdopen(..., closefd=True) file object and closes it on exit
            # (even on EOF/exception, via the with-block). The main thread never
            # closes read_fd itself, so the bounded join() below can't race a
            # concurrent close of the read end — only the daemon thread touches it.
            try:
                with os.fdopen(read_fd, "rb", closefd=True) as r:
                    while True:
                        data = r.read(65536)
                        if not data:
                            return
                        chunks.append(data)
            except Exception:
                return

        try:
            reader = threading.Thread(target=_drain, daemon=True)
            try:
                reader.start()
            except Exception:
                # The reader never ran, so it never took ownership of read_fd:
                # close it here so the read end is not leaked. fd 2 is still
                # restored (and write_fd closed) in the finally below.
                reader = None
                with contextlib.suppress(Exception):
                    os.close(read_fd)
            yield
        finally:
            # Restore fd 2 from the saved dup. This also drops our reference to
            # the pipe's write end on fd 2, so once write_fd is closed the reader
            # hits EOF and finishes.
            with contextlib.suppress(Exception):
                os.dup2(saved_fd, 2)
            with contextlib.suppress(Exception):
                os.close(saved_fd)
            with contextlib.suppress(Exception):
                os.close(write_fd)
            if reader is not None:
                # Bounded join: the reader exits on EOF; the timeout guards
                # against any pathological case so we never hang the decode path.
                # The reader owns and closes read_fd, so we must NOT close it here.
                reader.join(timeout=2.0)
            # Forward whatever was captured, one line at a time, after fd 2 is
            # back to normal (so loguru's own sink writes to the real stderr).
            captured = b"".join(chunks).decode("utf-8", "replace")
            for line in captured.splitlines():
                with contextlib.suppress(Exception):
                    _forward_native_line(line, prefix)


class InterceptHandler(logging.Handler):
    """Forward every stdlib logging record into loguru (loguru's documented recipe)."""

    def emit(self, record: logging.LogRecord) -> None:
        # Map the stdlib level name onto loguru's; fall back to the numeric level.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Unwind the stdlib frames so loguru reports the real caller, not this
        # handler. Start at emit()'s own frame (depth 0) and force the first step,
        # then keep walking while frames belong to the stdlib logging module.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str) -> None:
    """Install the single loguru sink and funnel stdlib logging into it."""
    logger.remove()
    logger.add(sys.stderr, level=level)
    # Replace root handlers with the interceptor; level=0 lets every record reach
    # loguru, which then applies the configured threshold. force=True drops handlers
    # any imported library may have installed via basicConfig.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # aioesphomeapi logs EVERY streamed audio frame (with the full PCM payload) at
    # DEBUG, which floods the log when the app runs at DEBUG. Pin it to INFO so the
    # useful connect/handshake lines stay but the per-frame audio dump is dropped.
    logging.getLogger("aioesphomeapi").setLevel(logging.INFO)
