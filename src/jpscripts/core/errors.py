"""Core error types for jpscripts.

This module defines base exceptions that can be used across all subsystems
without creating circular import dependencies.
"""

from __future__ import annotations


class SecurityError(RuntimeError):
    """Raised when a security constraint is violated.

    Used for:
    - Circuit breaker trips
    - Unsafe tool invocations
    - Path traversal attempts
    - Policy violations
    """


__all__ = ["SecurityError"]
