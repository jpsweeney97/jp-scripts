"""Safety middleware for circuit breaker enforcement.

This module provides reusable safety primitives that can wrap any async function
to enforce cost and resource limits via circuit breakers.

Usage:
    from jpscripts.core.safety import guarded_execution

    # As a decorator
    @guarded_execution("mcp-client", "tool:read_file")
    async def read_file(path: str) -> str:
        ...

    # Or wrap dynamically
    wrapped = wrap_with_breaker(func, "mcp-client", "tool:read_file", breaker)
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.errors import SecurityError
from jpscripts.core.runtime import CircuitBreaker, get_runtime

P = ParamSpec("P")
R = TypeVar("R")


def estimate_tokens_from_args(*args: Any, **kwargs: Any) -> int:
    """Estimate token count from argument sizes.

    This provides a rough estimate based on string lengths and structure.
    Used for circuit breaker enforcement when actual token counts are unavailable.

    Args:
        *args: Positional arguments to estimate
        **kwargs: Keyword arguments to estimate

    Returns:
        Estimated token count (minimum 1)
    """
    total = 0
    for arg in args:
        total += _estimate_value_tokens(arg)
    for value in kwargs.values():
        total += _estimate_value_tokens(value)
    return max(total, 1)  # Minimum 1 token


def _estimate_value_tokens(value: Any) -> int:
    """Estimate tokens for a single value recursively."""
    if isinstance(value, str):
        return max(1, len(value) // 4)  # ~4 chars per token
    if isinstance(value, bytes):
        return max(1, len(value) // 4)
    if isinstance(value, Path):
        return max(1, len(str(value)) // 4)
    if isinstance(value, (list, tuple)):
        return sum(_estimate_value_tokens(v) for v in value) or 1
    if isinstance(value, dict):
        return (
            sum(_estimate_value_tokens(k) + _estimate_value_tokens(v) for k, v in value.items())
            or 1
        )
    return 1  # Default for primitives (int, float, bool, None)


def check_circuit_breaker(
    breaker: CircuitBreaker,
    usage: TokenUsage,
    files_touched: list[Path],
    *,
    persona: str,
    context: str,
) -> None:
    """Check circuit breaker and raise SecurityError if triggered.

    This is the low-level enforcement function. Most callers should use
    guarded_execution() or wrap_with_breaker() instead.

    Args:
        breaker: The circuit breaker to check
        usage: Token usage for this operation
        files_touched: Files modified in this operation
        persona: Identifier for the caller (for reporting)
        context: Context description (for reporting)

    Raises:
        SecurityError: If circuit breaker is triggered
    """
    if breaker.check_health(usage, files_touched):
        return

    raise SecurityError(
        f"Circuit breaker tripped: {breaker.last_failure_reason} "
        f"(persona={persona}, context={context}, "
        f"cost_velocity={breaker.last_cost_velocity:.4f} USD/min, "
        f"file_churn={breaker.last_file_churn})"
    )


def guarded_execution(
    persona: str,
    context_label: str,
    *,
    breaker: CircuitBreaker | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator that enforces circuit breaker limits before executing a function.

    Args:
        persona: Identifier for the caller (e.g., "mcp-client", "agent")
        context_label: Context string for logging (e.g., "tool:read_file")
        breaker: Optional explicit breaker; defaults to runtime breaker

    Returns:
        Decorated async function

    Raises:
        SecurityError: If circuit breaker trips

    Example:
        @guarded_execution("agent", "tool:shell")
        async def run_shell(command: str) -> str:
            ...
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Get breaker (explicit or from runtime)
            cb = breaker
            if cb is None:
                cb = get_runtime().get_circuit_breaker()

            # Estimate tokens from arguments
            estimated_tokens = estimate_tokens_from_args(*args, **kwargs)
            usage = TokenUsage(prompt_tokens=estimated_tokens, completion_tokens=0)

            # Check circuit breaker
            check_circuit_breaker(
                cb,
                usage,
                files_touched=[],
                persona=persona,
                context=context_label,
            )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def wrap_with_breaker(
    func: Callable[..., Awaitable[str]],
    persona: str,
    context_label: str,
    breaker: CircuitBreaker,
) -> Callable[..., Awaitable[str]]:
    """Wrap a tool function with circuit breaker enforcement.

    This is the non-decorator form for dynamic wrapping (e.g., MCP tool registration).

    Args:
        func: The async function to wrap
        persona: Identifier for the caller
        context_label: Context string for logging
        breaker: The circuit breaker instance to use

    Returns:
        Wrapped async function
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        # Estimate tokens from arguments
        estimated_tokens = estimate_tokens_from_args(*args, **kwargs)
        usage = TokenUsage(prompt_tokens=estimated_tokens, completion_tokens=0)

        # Check circuit breaker
        check_circuit_breaker(
            breaker,
            usage,
            files_touched=[],
            persona=persona,
            context=context_label,
        )

        return await func(*args, **kwargs)

    return wrapper


def wrap_mcp_tool(
    func: Callable[..., Awaitable[str]],
    tool_name: str,
) -> Callable[..., Awaitable[str]]:
    """Wrap an MCP tool with circuit breaker enforcement.

    Uses the MCP-specific circuit breaker from the runtime context.
    This wrapper is applied during MCP server tool registration.

    Args:
        func: The async tool function to wrap
        tool_name: Name of the tool for logging

    Returns:
        Wrapped async function with safety enforcement
    """
    from jpscripts.core.runtime import get_runtime_or_none

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        # Get MCP breaker from runtime (if available)
        runtime = get_runtime_or_none()
        if runtime is not None:
            breaker = runtime.get_mcp_circuit_breaker()

            # Estimate tokens from arguments
            estimated_tokens = estimate_tokens_from_args(*args, **kwargs)
            usage = TokenUsage(prompt_tokens=estimated_tokens, completion_tokens=0)

            # Check circuit breaker
            check_circuit_breaker(
                breaker,
                usage,
                files_touched=[],
                persona="mcp-client",
                context=f"tool:{tool_name}",
            )

        return await func(*args, **kwargs)

    return wrapper


__all__ = [
    "SecurityError",
    "check_circuit_breaker",
    "estimate_tokens_from_args",
    "guarded_execution",
    "wrap_mcp_tool",
    "wrap_with_breaker",
]
