"""Backward-compatible import shim.

This module re-exports from the new location: jpscripts.core.security.rate_limit

Deprecated: Import directly from jpscripts.core.security instead.
"""

from jpscripts.core.security.rate_limit import (
    RateLimiter,
    RateLimitExceeded,
    rate_limited_call,
)

__all__ = [
    "RateLimitExceeded",
    "RateLimiter",
    "rate_limited_call",
]
