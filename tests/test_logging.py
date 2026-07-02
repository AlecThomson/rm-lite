"""Tests for the rm_lite.utils.logging quiet_logs context manager."""

from __future__ import annotations

import logging
import threading

from rm_lite.utils.logging import logger, quiet_logs


def test_quiet_logs_restores_level() -> None:
    original_level = logger.level
    with quiet_logs(logging.ERROR):
        assert logger.level == logging.ERROR
    assert logger.level == original_level


def test_quiet_logs_nested_uses_most_restrictive_level() -> None:
    original_level = logger.level
    with quiet_logs(logging.WARNING):
        assert logger.level == logging.WARNING
        with quiet_logs(logging.ERROR):
            assert logger.level == logging.ERROR
        # The outer, less restrictive request must be back in effect, not the
        # original pre-quiet level.
        assert logger.level == logging.WARNING
    assert logger.level == original_level


def test_quiet_logs_concurrent_callers_do_not_under_suppress(caplog) -> None:
    """A concurrent, less-restrictive request must not leak a stricter one's messages.

    Regression test: dask's threaded scheduler runs one block-worker call per
    chunk concurrently, and rmsynth_3d's blocks (log_level=WARNING) and
    rmclean_3d's blocks (log_level=ERROR) can be in flight at the same time.
    A plain reentrant depth counter (set level on the first quiet_logs entry,
    restore on the last exit) would let whichever caller entered first "win"
    the level for the whole overlap, silently under-suppressing a
    concurrently-running stricter request.
    """
    entered_warning = threading.Event()
    release_warning = threading.Event()
    strict_saw_level = []

    def hold_warning_scope() -> None:
        with quiet_logs(logging.WARNING):
            entered_warning.set()
            release_warning.wait(timeout=5)

    def enter_error_scope() -> None:
        entered_warning.wait(timeout=5)
        with quiet_logs(logging.ERROR):
            strict_saw_level.append(logger.level)
            # Deliberately not using caplog.at_level here: it would set the
            # logger's level itself, defeating the point of this check.
            logger.warning("should be suppressed by the concurrent ERROR scope")

    warning_thread = threading.Thread(target=hold_warning_scope)
    error_thread = threading.Thread(target=enter_error_scope)

    warning_thread.start()
    error_thread.start()
    error_thread.join(timeout=5)
    release_warning.set()
    warning_thread.join(timeout=5)

    assert strict_saw_level == [logging.ERROR]
    assert "should be suppressed by the concurrent ERROR scope" not in caplog.text
