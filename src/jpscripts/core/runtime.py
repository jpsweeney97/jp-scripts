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
        workspace = ctx.workspace_root
        config = ctx.config
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig


@dataclass
class WarningState:
    """Track which warnings have been emitted to avoid spam."""

    semantic_unavailable: bool = False
    memory_degraded: bool = False
    codex_missing: bool = False


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

    def get_rate_limiter(self) -> Any:
        """Get or create the rate limiter for this context."""
        if self._rate_limiter is None:
            from jpscripts.core.rate_limit import RateLimiter

            self._rate_limiter = RateLimiter(
                max_calls=self.config.shell_rate_limit_calls
                if hasattr(self.config, "shell_rate_limit_calls")
                else 100,
                window_seconds=self.config.shell_rate_limit_window
                if hasattr(self.config, "shell_rate_limit_window")
                else 60.0,
            )
        return self._rate_limiter

    def get_cost_tracker(self) -> Any:
        """Get or create the cost tracker for this context."""
        if self._cost_tracker is None:
            from jpscripts.core.cost_tracker import CostTracker

            self._cost_tracker = CostTracker(
                model_id=self.config.default_model,
            )
        return self._cost_tracker


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
            "Ensure code runs within a 'with runtime_context(...):' block."
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
        workspace: Override workspace root (defaults to config.workspace_root)
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
    resolved_workspace = (workspace or config.workspace_root).expanduser().resolve()

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
    "RuntimeContext",
    "WarningState",
    "get_runtime",
    "get_runtime_or_none",
    "has_runtime",
    "runtime_context",
    "override_workspace",
    "NoRuntimeContextError",
]
