"""
Security utilities for workspace, path, and command validation.

This package consolidates all security-related functionality:
- Path validation and workspace sandboxing
- Command validation and allowlisting
- Rate limiting for resource protection
- Circuit breaker for runaway cost/churn prevention

Usage:
    from jpscripts.core.security import validate_path, validate_command
    from jpscripts.core.security import RateLimiter, CircuitBreaker

All exports are backward-compatible with the previous module-level imports.
"""

from __future__ import annotations

# Path validation (previously core/security.py)
from jpscripts.core.security.path import (
    FORBIDDEN_ROOTS,
    MAX_SYMLINK_DEPTH,
    _is_forbidden_path,
    _resolve_with_limit,
    _resolve_with_limit_async,
    is_git_workspace,
    is_path_safe,
    validate_and_open,
    validate_path,
    validate_path_async,
    validate_workspace_root,
    validate_workspace_root_async,
)

# Command validation (previously core/command_validation.py)
from jpscripts.core.security.command import (
    ALLOWED_BINARIES,
    ALLOWED_GIT_SUBCOMMANDS,
    FORBIDDEN_BINARIES,
    FORBIDDEN_GIT_SUBCOMMANDS,
    CommandVerdict,
    is_command_safe,
    validate_command,
)

# Rate limiting (previously core/rate_limit.py)
from jpscripts.core.security.rate_limit import (
    RateLimitExceeded,
    RateLimiter,
    rate_limited_call,
)

# Safety middleware (previously core/safety.py)
from jpscripts.core.security.safety import (
    check_circuit_breaker,
    estimate_tokens_from_args,
    guarded_execution,
    wrap_mcp_tool,
    wrap_with_breaker,
)

__all__ = [
    # Path validation
    "FORBIDDEN_ROOTS",
    "MAX_SYMLINK_DEPTH",
    "_is_forbidden_path",
    "_resolve_with_limit",
    "_resolve_with_limit_async",
    "is_git_workspace",
    "is_path_safe",
    "validate_and_open",
    "validate_path",
    "validate_path_async",
    "validate_workspace_root",
    "validate_workspace_root_async",
    # Command validation
    "ALLOWED_BINARIES",
    "ALLOWED_GIT_SUBCOMMANDS",
    "CommandVerdict",
    "FORBIDDEN_BINARIES",
    "FORBIDDEN_GIT_SUBCOMMANDS",
    "is_command_safe",
    "validate_command",
    # Rate limiting
    "RateLimitExceeded",
    "RateLimiter",
    "rate_limited_call",
    # Safety middleware
    "check_circuit_breaker",
    "estimate_tokens_from_args",
    "guarded_execution",
    "wrap_mcp_tool",
    "wrap_with_breaker",
]
