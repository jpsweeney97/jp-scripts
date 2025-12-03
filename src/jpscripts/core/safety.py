"""Backward-compatible import shim.

This module re-exports from the new location: jpscripts.core.security.safety

Deprecated: Import directly from jpscripts.core.security instead.
"""

from jpscripts.core.security.safety import (
    check_circuit_breaker,
    estimate_tokens_from_args,
    guarded_execution,
    wrap_mcp_tool,
    wrap_with_breaker,
)

# Re-export SecurityError from errors (where it was originally exported)
from jpscripts.core.errors import SecurityError

__all__ = [
    "SecurityError",
    "check_circuit_breaker",
    "estimate_tokens_from_args",
    "guarded_execution",
    "wrap_mcp_tool",
    "wrap_with_breaker",
]
