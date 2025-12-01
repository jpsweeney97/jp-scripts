"""Core functionality for jpscripts.

This package contains shared logic and utilities:
    - config: Application configuration management
    - console: Rich console output and logging
    - security: Path validation and safety checks
    - memory: Vector-based memory store
    - agent: AI agent execution framework
"""

from __future__ import annotations

from . import config, console

__all__ = ["config", "console"]
