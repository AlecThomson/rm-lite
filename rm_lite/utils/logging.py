"""Logging module for arrakis"""

from __future__ import annotations

import io
import logging


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super().__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)  # type: ignore[union-attr]


class CustomFormatter(logging.Formatter):
    format_str = "%(asctime)s.%(msecs)03d %(module)s - %(funcName)s: %(message)s"

    FORMATS = {  # noqa: RUF012
        logging.DEBUG: f"%(levelname)s {format_str}",
        logging.INFO: f"%(levelname)s {format_str}",
        logging.WARNING: f"%(levelname)s {format_str}",
        logging.ERROR: f"%(levelname)s {format_str}",
        logging.CRITICAL: f"%(levelname)s {format_str}",
    }

    def format(self, record):
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
    logger.setLevel(logging.WARNING)

    if attach_handler:
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Add formatter to ch
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger


logger = get_logger()
