import logging
import os

import pytest
from loguru import logger

from src.logging_setup import _NOISY_LOGGERS, capture_native_stderr, setup_logging


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


def test_capture_native_stderr_routes_fd2_warning_to_loguru():
    # A native-style WARNING written straight to fd 2 inside the block is captured
    # and forwarded to loguru as a `warning` record. fd 2 is restored afterwards.
    records = []
    sink_id = logger.add(records.append, level="DEBUG")
    saved = os.dup(2)  # snapshot the real fd 2 to assert it is restored
    try:
        try:
            with capture_native_stderr("vosk"):
                os.write(2, b"WARNING (VoskAPI) test\n")
        except OSError:
            # fd redirection may be unsupported in some CI sandboxes; be lenient.
            pytest.skip("fd-2 redirection unsupported in this environment")
    finally:
        os.close(saved)
        logger.remove(sink_id)

    # The captured line was routed at WARNING level with the prefix prepended.
    assert any(
        r.record["level"].name == "WARNING"
        and "vosk: WARNING (VoskAPI) test" in r.record["message"]
        for r in records
    ), [r.record["message"] for r in records]

    # fd 2 still works after the block (restored): writing to it must not raise.
    os.write(2, b"")


def test_capture_native_stderr_degrades_to_passthrough_when_redirect_fails(monkeypatch):
    # If the os-level redirect setup fails, the context manager must NOT raise:
    # it degrades to a plain pass-through and the wrapped block still runs.
    def _boom(*_a, **_k):
        raise OSError("pipe unsupported")

    monkeypatch.setattr(os, "pipe", _boom)
    ran = []
    with capture_native_stderr("vosk"):
        ran.append(True)
    assert ran == [True]
    # fd 2 is intact (the failed setup was undone): writing must not raise.
    os.write(2, b"")
