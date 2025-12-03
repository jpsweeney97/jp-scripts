"""Backward-compatible import shim.

This module re-exports from the new location: jpscripts.core.security.command

Deprecated: Import directly from jpscripts.core.security instead.
"""

from jpscripts.core.security.command import (
    ALLOWED_BINARIES,
    ALLOWED_GIT_SUBCOMMANDS,
    FORBIDDEN_BINARIES,
    FORBIDDEN_GIT_SUBCOMMANDS,
    CommandVerdict,
    is_command_safe,
    validate_command,
)

__all__ = [
    "ALLOWED_BINARIES",
    "ALLOWED_GIT_SUBCOMMANDS",
    "FORBIDDEN_BINARIES",
    "FORBIDDEN_GIT_SUBCOMMANDS",
    "CommandVerdict",
    "is_command_safe",
    "validate_command",
]
