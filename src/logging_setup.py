"""Route the Python standard ``logging`` module through loguru.

Third-party libraries (httpx, mcp, aioesphomeapi, ...) emit via stdlib
``logging`` and would otherwise bypass loguru entirely, printing raw lines that
do not match our unified ``timestamp | LEVEL | module:func:line - message``
format. Installing loguru's documented ``InterceptHandler`` on the root stdlib
logger forwards every record into loguru so the whole app shares one sink and
honours ``core.log_level``.
"""

import inspect
import logging
import sys

from loguru import logger

# These loggers emit one INFO line per HTTP request / per MCP session, which
# would spam the unified log on every MCP tool call. Demote them to WARNING so
# routine chatter is silenced while warnings/errors still pass through.
_NOISY_LOGGERS = ("httpx", "httpcore", "mcp")


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
