"""Factory functions for creating AgentEngine instances with middleware.

This module provides factory functions that encapsulate the middleware
construction logic, implementing dependency injection for AgentEngine.

Usage:
    from jpscripts.agent.factory import build_default_middleware, create_agent

    # Option 1: Build middleware separately
    middleware = build_default_middleware(
        workspace_root=root,
        persona="Engineer",
        governance_enabled=True,
        render_prompt=...,
        fetch_response=...,
        parser=...,
    )
    engine = AgentEngine(middleware=middleware, ...)

    # Option 2: Use factory function
    engine = create_agent(
        persona="Engineer",
        workspace_root=root,
        ...
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .middleware import (
    AgentMiddleware,
    CircuitBreakerMiddleware,
    GovernanceMiddleware,
    TracingMiddleware,
)
from .models import Message, PreparedPrompt, ResponseT
from .tracing import TraceRecorder

if TYPE_CHECKING:
    from .engine import AgentEngine


def build_default_middleware(
    *,
    persona: str,
    workspace_root: Path | None = None,
    trace_dir: Path | None = None,
    governance_enabled: bool = True,
    render_prompt: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]] | None = None,
    fetch_response: Callable[[PreparedPrompt], Awaitable[str]] | None = None,
    parser: Callable[[str], ResponseT] | None = None,
) -> list[AgentMiddleware[ResponseT]]:
    """Build the default middleware stack for an AgentEngine.

    The default middleware stack includes (in execution order):
    1. GovernanceMiddleware - validates responses, may retry on violations
    2. CircuitBreakerMiddleware - enforces safety limits
    3. TracingMiddleware - records execution traces

    Args:
        persona: Agent persona name (for circuit breaker and tracing)
        workspace_root: Workspace root for governance checks (required if governance_enabled)
        trace_dir: Directory for trace files (defaults to ~/.jpscripts/traces)
        governance_enabled: Whether to include governance middleware
        render_prompt: Prompt rendering function (required if governance_enabled)
        fetch_response: Response fetching function (required if governance_enabled)
        parser: Response parser function (required if governance_enabled)

    Returns:
        List of middleware in execution order
    """
    middleware: list[AgentMiddleware[ResponseT]] = []

    # 1. Governance (validates responses, may retry)
    if governance_enabled and workspace_root is not None:
        if render_prompt is None or fetch_response is None or parser is None:
            raise ValueError(
                "render_prompt, fetch_response, and parser are required when governance_enabled=True"
            )
        middleware.append(
            GovernanceMiddleware(
                workspace_root=workspace_root,
                render_prompt=render_prompt,
                fetch_response=fetch_response,
                parser=parser,
                enabled=True,
            )
        )

    # 2. Circuit breaker (enforces safety limits)
    middleware.append(CircuitBreakerMiddleware(persona=persona))

    # 3. Tracing (records execution traces)
    trace_recorder = TraceRecorder(trace_dir or Path.home() / ".jpscripts" / "traces")
    middleware.append(
        TracingMiddleware(
            trace_recorder=trace_recorder,
            persona=persona,
        )
    )

    return middleware


def create_agent(
    *,
    persona: str,
    model: str,
    prompt_builder: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]],
    fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
    parser: Callable[[str], ResponseT],
    tools: Mapping[str, Callable[..., Awaitable[str]]] | None = None,
    workspace_root: Path | None = None,
    template_root: Path | None = None,
    trace_dir: Path | None = None,
    governance_enabled: bool = True,
    extra_middleware: Sequence[AgentMiddleware[ResponseT]] | None = None,
) -> AgentEngine[ResponseT]:
    """Create an AgentEngine with the default middleware stack.

    This is a convenience function that builds the default middleware
    and creates an AgentEngine in one step.

    Args:
        persona: Agent persona name
        model: LLM model identifier
        prompt_builder: Async function to build prompts from message history
        fetch_response: Async function to fetch LLM responses
        parser: Function to parse raw responses into structured format
        tools: Tool registry mapping names to async callables
        workspace_root: Workspace root for governance and path validation
        template_root: Root directory for templates
        trace_dir: Directory for trace files
        governance_enabled: Whether to enable governance middleware
        extra_middleware: Additional middleware to append after defaults

    Returns:
        Configured AgentEngine instance
    """
    # Import here to avoid circular dependency
    from .engine import AgentEngine

    # Build default middleware
    middleware = build_default_middleware(
        persona=persona,
        workspace_root=workspace_root,
        trace_dir=trace_dir,
        governance_enabled=governance_enabled,
        render_prompt=prompt_builder,
        fetch_response=fetch_response,
        parser=parser,
    )

    # Add extra middleware if provided
    if extra_middleware:
        middleware.extend(extra_middleware)

    return AgentEngine(
        persona=persona,
        model=model,
        prompt_builder=prompt_builder,
        fetch_response=fetch_response,
        parser=parser,
        tools=tools,
        workspace_root=workspace_root,
        template_root=template_root,
        trace_dir=trace_dir,
        middleware=middleware,
    )


__all__ = [
    "build_default_middleware",
    "create_agent",
]
