"""Logging module for arrakis"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf: str) -> int:
        self.buf = buf.strip("\r\n\t ")
        return len(buf)

    def flush(self) -> None:
        if self.logger is not None and self.level is not None:
            self.logger.log(self.level, self.buf)


class CustomFormatter(logging.Formatter):
    format_str = "%(module)s.%(funcName)s: %(message)s"

    FORMATS = {  # noqa: RUF012
        logging.DEBUG: f"%(levelname)s {format_str}",
        logging.INFO: f"%(levelname)s {format_str}",
        logging.WARNING: f"%(levelname)s {format_str}",
        logging.ERROR: f"%(levelname)s {format_str}",
        logging.CRITICAL: f"%(levelname)s {format_str}",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(
    name: str = "rmtools-lite", attach_handler: bool = True
) -> logging.Logger:
    """Will construct a logger object.

    Args:
        name (str, optional): Name of the logger to attempt to use. This is ignored if in a prefect flowrun. Defaults to 'arrakis'.
        attach_handler (bool, optional): Attacjes a custom StreamHandler. Defaults to True.

    Returns:
        logging.Logger: The appropriate logger
    """
    logging.captureWarnings(True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if attach_handler:
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Add formatter to ch
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger


logger = get_logger()


class _QuietState:
    """Multiset of active requested levels backing `quiet_logs` (avoids module globals)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_levels: list[int] = []
        self.level_before_quiet = logger.level


_quiet_state = _QuietState()


@contextmanager
def quiet_logs(level: int = logging.WARNING) -> Iterator[None]:
    """Temporarily raise the rm-lite logger's level to at least `level`.

    Reentrant and thread-safe: concurrent callers (e.g. dask workers running
    one call per chunk, possibly requesting different levels -- `rmsynth_3d`
    defaults to WARNING, `rmclean_3d` to ERROR) share a multiset of active
    requested levels. The logger's level is always the max (most restrictive)
    of whatever's currently active, recomputed on every entry/exit, and only
    restored to its original value once every caller has exited. Tracking a
    plain reentrant depth counter instead of the actual requested levels would
    let a concurrent, less-restrictive request silently under-suppress a
    stricter one running at the same time.
    """
    with _quiet_state.lock:
        if not _quiet_state.active_levels:
            _quiet_state.level_before_quiet = logger.level
        _quiet_state.active_levels.append(level)
        logger.setLevel(max(_quiet_state.active_levels))
    try:
        yield
    finally:
        with _quiet_state.lock:
            _quiet_state.active_levels.remove(level)
            if _quiet_state.active_levels:
                logger.setLevel(max(_quiet_state.active_levels))
            else:
                logger.setLevel(_quiet_state.level_before_quiet)
