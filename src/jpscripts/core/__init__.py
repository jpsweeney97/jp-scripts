"""Core shared infrastructure for jpscripts.

This package contains foundational utilities:
    - config: Application configuration management
    - console: Rich console output and logging
    - security: Path validation and safety checks
    - runtime: Runtime context management
    - result: Error handling patterns
    - tokens: Token budget management
"""

from __future__ import annotations

from . import config, console

__all__ = ["config", "console"]
