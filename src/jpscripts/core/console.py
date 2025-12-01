"""Console output and logging configuration.

Provides Rich-based console output and logging setup:
    - console: Main Rich console for stdout
    - stderr_console: Rich console for stderr
    - setup_logging(): Configure logging with Rich handler
    - get_logger(): Get a named logger instance
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
stderr_console = Console(stderr=True)


def _normalize_level(level: str | int) -> int:
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def setup_logging(level: str | int = logging.INFO, verbose: bool = False) -> logging.Logger:
    """Configure logging with a Rich handler and return the app logger."""
    numeric_level = logging.DEBUG if verbose else _normalize_level(level)

    handler = RichHandler(
        console=stderr_console,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(numeric_level)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)
    root.addHandler(handler)

    logger = logging.getLogger("jpscripts")
    logger.handlers.clear()
    logger.setLevel(numeric_level)
    logger.addHandler(handler)

    return logger


def get_console(stderr: bool = False) -> Console:
    return stderr_console if stderr else console


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "jpscripts")
