"""
Thread-safe runtime context for jp-scripts.

This module provides a context variable-based system for managing runtime
state across async tasks and threads without global mutable state.

Usage:
    from jpscripts.core.runtime import runtime_context, get_runtime

    # At entry point (main.py, mcp/server.py)
    with runtime_context(config) as ctx:
        # All code in this block can access context via get_runtime()
        do_work()

    # In any module
    def some_function():
        ctx = get_runtime()
        workspace = ctx.user.workspace_root
        config = ctx.config
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig

from jpscripts.core.cost_tracker import TokenUsage, get_pricing


@dataclass
class WarningState:
    """Track which warnings have been emitted to avoid spam."""

    semantic_unavailable: bool = False
    memory_degraded: bool = False


@dataclass
class CircuitBreaker:
    """Guardrails for runaway cost or churn during a single agent turn."""

    max_cost_velocity: Decimal
    max_file_churn: int
    model_id: str = "default"

    _last_check_timestamp: float | None = field(default=None, init=False, repr=False)
    last_cost_estimate: Decimal = field(default=Decimal("0"), init=False)
    last_cost_velocity: Decimal = field(default=Decimal("0"), init=False)
    last_file_churn: int = field(default=0, init=False)
    last_failure_reason: str | None = field(default=None, init=False)

    def check_health(self, usage: TokenUsage, files_touched: list[Path]) -> bool:
        """Evaluate whether current usage stays within guardrails."""
        cost = self._estimate_cost(usage)
        file_churn = self._count_unique_files(files_touched)
        velocity = self._compute_velocity(cost)

        self.last_cost_estimate = cost
        self.last_cost_velocity = velocity
        self.last_file_churn = file_churn

        if velocity > self.max_cost_velocity:
            self.last_failure_reason = "Cost velocity threshold exceeded"
            return False
        if file_churn > self.max_file_churn:
            self.last_failure_reason = "File churn threshold exceeded"
            return False

        self.last_failure_reason = None
        return True

    def _estimate_cost(self, usage: TokenUsage) -> Decimal:
        pricing = get_pricing(self.model_id)
        input_cost = (Decimal(usage.prompt_tokens) / Decimal(1_000_000)) * pricing["input"]
        output_cost = (Decimal(usage.completion_tokens) / Decimal(1_000_000)) * pricing["output"]
        return input_cost + output_cost

    def _compute_velocity(self, cost: Decimal) -> Decimal:
        now = monotonic()
        if self._last_check_timestamp is None:
            self._last_check_timestamp = now
            return cost

        elapsed_minutes = max((now - self._last_check_timestamp) / 60, 1 / 60)
        self._last_check_timestamp = now
        return cost / Decimal(str(elapsed_minutes))

    @staticmethod
    def _count_unique_files(files_touched: list[Path]) -> int:
        return len(set(files_touched))


@dataclass
class RuntimeContext:
    """Thread-local runtime context for all operations.

    This replaces all module-level globals with a context variable that
    is properly isolated between async tasks and threads.

    Attributes:
        config: The loaded AppConfig for this session
        workspace_root: Resolved workspace root path
        trace_id: Unique identifier for this execution trace
        dry_run: Whether to skip destructive operations
        warnings: Tracks emitted warnings to prevent duplicates
    """

    config: AppConfig
    workspace_root: Path
    trace_id: str = field(default_factory=lambda: uuid4().hex[:12])
    dry_run: bool = False
    warnings: WarningState = field(default_factory=WarningState)

    # Lazy-initialized optional components
    _rate_limiter: Any | None = field(default=None, repr=False)
    _cost_tracker: Any | None = field(default=None, repr=False)
    _circuit_breaker: CircuitBreaker | None = field(default=None, repr=False)

    def get_rate_limiter(self) -> Any:
        """Get or create the rate limiter for this context."""
        if self._rate_limiter is None:
            from jpscripts.core.rate_limit import RateLimiter

            self._rate_limiter = RateLimiter(
                max_calls=self.config.infra.shell_rate_limit_calls
                if hasattr(self.config, "shell_rate_limit_calls")
                else 100,
                window_seconds=self.config.infra.shell_rate_limit_window
                if hasattr(self.config, "shell_rate_limit_window")
                else 60.0,
            )
        return self._rate_limiter

    def get_cost_tracker(self) -> Any:
        """Get or create the cost tracker for this context."""
        if self._cost_tracker is None:
            from jpscripts.core.cost_tracker import CostTracker

            self._cost_tracker = CostTracker(
                model_id=self.config.ai.default_model,
            )
        return self._cost_tracker

    def get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create the circuit breaker for this context."""
        if self._circuit_breaker is None:
            max_velocity = getattr(self.config, "max_cost_velocity", Decimal("5.0"))
            max_churn = getattr(self.config, "max_file_churn", 12)
            self._circuit_breaker = CircuitBreaker(
                max_cost_velocity=Decimal(str(max_velocity)),
                max_file_churn=int(max_churn),
                model_id=getattr(self.config, "default_model", "default"),
            )
        return self._circuit_breaker


# Context variable for async/thread safety
_runtime_ctx: contextvars.ContextVar[RuntimeContext | None] = contextvars.ContextVar(
    "jp_runtime",
    default=None,
)


class NoRuntimeContextError(RuntimeError):
    """Raised when get_runtime() is called outside a runtime_context block."""

    def __init__(self) -> None:
        super().__init__(
            "No runtime context available. "
            "Wrap entrypoints in 'with runtime_context(config):' or use the CLI bootstrap that establishes runtime automatically."
        )


def get_runtime() -> RuntimeContext:
    """Get the current runtime context.

    Raises:
        NoRuntimeContextError: If called outside a runtime_context block

    Returns:
        The current RuntimeContext
    """
    ctx = _runtime_ctx.get()
    if ctx is None:
        raise NoRuntimeContextError()
    return ctx


def get_runtime_or_none() -> RuntimeContext | None:
    """Get the current runtime context, or None if not set.

    Use this for optional context access where a fallback is acceptable.
    """
    return _runtime_ctx.get()


def has_runtime() -> bool:
    """Check if a runtime context is currently active."""
    return _runtime_ctx.get() is not None


def set_runtime_context(ctx: RuntimeContext) -> contextvars.Token[RuntimeContext | None]:
    """Set the current runtime context (used for CLI bootstrap and tests)."""
    return _runtime_ctx.set(ctx)


def reset_runtime_context(token: contextvars.Token[RuntimeContext | None]) -> None:
    """Reset the runtime context to a previous token."""
    _runtime_ctx.reset(token)


def get_circuit_breaker() -> CircuitBreaker:
    """Convenience accessor for the active circuit breaker."""
    return get_runtime().get_circuit_breaker()


@contextmanager
def runtime_context(
    config: AppConfig,
    workspace: Path | None = None,
    *,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> Iterator[RuntimeContext]:
    """Context manager for establishing runtime state.

    This should be called at application entry points to establish
    the runtime context for all subsequent operations.

    Args:
        config: The loaded AppConfig
        workspace: Override workspace root (defaults to config.user.workspace_root)
        dry_run: Whether to skip destructive operations
        trace_id: Optional trace ID for correlation (auto-generated if not provided)

    Yields:
        The RuntimeContext for this execution

    Example:
        cfg, _ = load_config()
        with runtime_context(cfg) as ctx:
            # All code here can use get_runtime()
            run_agent(...)
    """
    resolved_workspace = (workspace or config.user.workspace_root).expanduser().resolve()

    ctx = RuntimeContext(
        config=config,
        workspace_root=resolved_workspace,
        trace_id=trace_id or uuid4().hex[:12],
        dry_run=dry_run or getattr(config, "dry_run", False),
    )

    token = _runtime_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _runtime_ctx.reset(token)


@contextmanager
def override_workspace(workspace: Path) -> Iterator[RuntimeContext]:
    """Temporarily override the workspace root within an existing context.

    Useful for operations that need to work in a different directory
    while preserving other context settings.

    Args:
        workspace: The new workspace root

    Raises:
        NoRuntimeContextError: If called outside a runtime_context block
    """
    current = get_runtime()
    resolved = workspace.expanduser().resolve()

    new_ctx = RuntimeContext(
        config=current.config,
        workspace_root=resolved,
        trace_id=current.trace_id,
        dry_run=current.dry_run,
        warnings=current.warnings,
        _rate_limiter=current._rate_limiter,
        _cost_tracker=current._cost_tracker,
    )

    token = _runtime_ctx.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _runtime_ctx.reset(token)


__all__ = [
    "CircuitBreaker",
    "NoRuntimeContextError",
    "RuntimeContext",
    "WarningState",
    "get_circuit_breaker",
    "get_runtime",
    "get_runtime_or_none",
    "has_runtime",
    "override_workspace",
    "runtime_context",
]
