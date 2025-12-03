"""Backward-compatible import shim.

This module re-exports from the new location: jpscripts.core.security.safety

Deprecated: Import directly from jpscripts.core.security instead.
"""

# Re-export SecurityError from result (consolidated error types)
from jpscripts.core.result import SecurityError
from jpscripts.core.security.safety import (
    check_circuit_breaker,
    estimate_tokens_from_args,
    guarded_execution,
    wrap_mcp_tool,
    wrap_with_breaker,
)

__all__ = [
    "SecurityError",
    "check_circuit_breaker",
    "estimate_tokens_from_args",
    "guarded_execution",
    "wrap_mcp_tool",
    "wrap_with_breaker",
]
