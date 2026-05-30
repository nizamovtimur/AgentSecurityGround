"""Логирование security gate: INFO в консоль, DEBUG — только с -v."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

CONSOLE_LEVEL_DEFAULT = logging.INFO
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if level is None:
        level = (os.getenv("VALIDATOR_LOG_LEVEL") or "INFO").strip().upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
