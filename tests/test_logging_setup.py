import logging

from loguru import logger

from src.logging_setup import _NOISY_LOGGERS, setup_logging


def test_stdlib_logging_is_routed_through_loguru():
    # setup_logging() calls logger.remove(), so add the capturing sink afterwards.
    setup_logging("DEBUG")
    records = []
    out = []
    sink_id = logger.add(records.append, level="INFO")
    # Format sink exposes the attributed caller so we can verify frame-unwinding.
    fmt_id = logger.add(out.append, level="INFO", format="{name}:{function}:{line} {message}")
    try:
        # Emit from a named local function so there is a real caller to attribute to.
        def _caller():
            logging.getLogger("some.third_party").info("hello from stdlib")

        _caller()
    finally:
        logger.remove(sink_id)
        logger.remove(fmt_id)
    assert any("hello from stdlib" in str(r) for r in records)
    captured = "".join(out)
    # The record must be attributed to the real caller, not the stdlib internals.
    assert "_caller" in captured
    assert "hello from stdlib" in captured
    assert "callHandlers" not in captured


def test_noisy_loggers_demoted_to_warning():
    setup_logging("DEBUG")
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING
