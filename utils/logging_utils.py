"""
Logging infrastructure for model training, inference, and server monitoring.

Status: Clean
"""

import logging
import sys
import copy
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Console formatter adding ANSI color highlights according to log level."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure global root logger format and handlers.

    Args:
        level: Minimum log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path to persist log output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove pre-existing handlers to prevent duplicated logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
    else:
        console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Retrieve or create a namespaced logger instance.

    Args:
        name: Module or component name.
        level: Optional override log level.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
