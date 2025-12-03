"""Agent middleware protocol and base implementations.

Middleware allows cross-cutting concerns to be added to the agent execution
pipeline without modifying the core AgentEngine.step() method.

This module provides:
- AgentMiddleware: Protocol for middleware components
- BaseMiddleware: Base class with no-op defaults
- StepContext: Context passed through the middleware pipeline
- TracingMiddleware: Records execution traces
- GovernanceMiddleware: Enforces governance rules
- CircuitBreakerMiddleware: Safety monitoring
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar


class MiddlewarePhase(StrEnum):
    """Phase of middleware execution.

    BEFORE: Executed before the agent step (can modify input)
    AFTER: Executed after the agent step (can modify output)
    """

    BEFORE = "before"
    AFTER = "after"


from pydantic import BaseModel

from jpscripts.core.cost_tracker import TokenUsage

from .circuit import enforce_circuit_breaker
from .governance import enforce_governance
from .models import AgentTraceStep, Message, PreparedPrompt
from .tracing import TraceRecorder, _get_tracer

if TYPE_CHECKING:
    pass

# Type for parsed response (must be BaseModel subclass)
R = TypeVar("R", bound=BaseModel)


@dataclass
class StepContext(Generic[R]):
    """Context passed through the middleware pipeline.

    Attributes:
        history: Conversation history leading to this step
        prepared: The prepared prompt (set after render)
        raw_response: Raw model output string (set after fetch)
        response: Parsed response object (set after parse)
        usage_snapshot: Token usage estimate (set after parse)
        files_touched: Files modified by this step
        metadata: Arbitrary key-value data for middleware communication
    """

    history: list[Message]
    prepared: PreparedPrompt | None = None
    raw_response: str | None = None
    response: R | None = None
    usage_snapshot: TokenUsage | None = None
    files_touched: list[Path] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class AgentMiddleware(Protocol[R]):
    """Protocol for agent middleware components.

    Middleware can intercept and modify the agent execution at two points:
    - before_step: After prompt is prepared but before model call
    - after_step: After response is parsed but before returning

    Middleware can:
    - Inspect/modify the StepContext
    - Raise exceptions to abort execution
    - Add metadata for downstream middleware
    """

    async def before_step(self, ctx: StepContext[R]) -> StepContext[R]:
        """Called after prompt preparation, before model invocation.

        Args:
            ctx: Current step context (prepared prompt is set)

        Returns:
            Modified context (can modify prepared prompt)

        Raises:
            Any exception to abort the step
        """
        ...

    async def after_step(self, ctx: StepContext[R]) -> StepContext[R]:
        """Called after response is parsed, before returning.

        Args:
            ctx: Current step context (response is set)

        Returns:
            Modified context (can modify response)

        Raises:
            Any exception to abort the step
        """
        ...


class BaseMiddleware(Generic[R]):
    """Base implementation with no-op defaults.

    Extend this to implement only the hooks you need.
    """

    async def before_step(self, ctx: StepContext[R]) -> StepContext[R]:
        return ctx

    async def after_step(self, ctx: StepContext[R]) -> StepContext[R]:
        return ctx


class TracingMiddleware(BaseMiddleware[R]):
    """Middleware for recording execution traces.

    Records to JSONL files and optionally OpenTelemetry.
    This extracts the tracing logic from AgentEngine._record_trace().
    """

    def __init__(
        self,
        trace_recorder: TraceRecorder,
        *,
        persona: str,
        otel_enabled: bool = True,
    ) -> None:
        self._recorder = trace_recorder
        self._persona = persona
        self._otel_enabled = otel_enabled

    async def after_step(self, ctx: StepContext[R]) -> StepContext[R]:
        """Record trace after response is complete."""
        from jpscripts.core.console import get_logger

        logger = get_logger(__name__)

        if ctx.response is None:
            return ctx

        try:
            step = AgentTraceStep(
                timestamp=datetime.now(UTC).isoformat(),
                agent_persona=self._persona,
                input_history=[{"role": msg.role, "content": msg.content} for msg in ctx.history],
                response=ctx.response.model_dump(),
                tool_output=ctx.metadata.get("tool_output"),
            )
            await self._recorder.append(step)

            if self._otel_enabled:
                self._emit_otel_span(ctx)

        except Exception as exc:
            logger.debug("Failed to record trace: %s", exc)

        return ctx

    def _emit_otel_span(self, ctx: StepContext[R]) -> None:
        tracer = _get_tracer()
        if tracer is None:
            return

        with tracer.start_as_current_span("agent.turn") as span:
            span.set_attribute("agent.persona", self._persona)

            if ctx.files_touched:
                span.set_attribute("code.files_touched", [str(p) for p in ctx.files_touched])

            if ctx.usage_snapshot is not None:
                span.set_attribute("usage.prompt_tokens", ctx.usage_snapshot.prompt_tokens)
                span.set_attribute("usage.completion_tokens", ctx.usage_snapshot.completion_tokens)
                span.set_attribute("usage.total_tokens", ctx.usage_snapshot.total_tokens)

            if ctx.response is not None:
                tool_call = getattr(ctx.response, "tool_call", None)
                if tool_call is not None:
                    span.add_event(
                        "tool_call",
                        {
                            "tool_call": tool_call.model_dump()
                            if hasattr(tool_call, "model_dump")
                            else str(tool_call)
                        },
                    )

            tool_output = ctx.metadata.get("tool_output")
            if tool_output:
                span.add_event("tool_output", {"output": str(tool_output)})


class GovernanceMiddleware(BaseMiddleware[R]):
    """Middleware for constitutional governance enforcement.

    Validates agent responses against governance rules and handles
    re-prompting for violations.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        render_prompt: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]],
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
        parser: Callable[[str], R],
        enabled: bool = True,
    ) -> None:
        self._workspace_root = workspace_root
        self._render_prompt = render_prompt
        self._fetch_response = fetch_response
        self._parser = parser
        self._enabled = enabled

    async def after_step(self, ctx: StepContext[R]) -> StepContext[R]:
        """Apply governance check after response is parsed."""
        if not self._enabled or self._workspace_root is None:
            return ctx

        if ctx.response is None or ctx.prepared is None or ctx.raw_response is None:
            return ctx

        # Delegate to existing governance logic
        response, prepared, raw = await enforce_governance(
            ctx.response,
            list(ctx.history),
            ctx.prepared,
            ctx.raw_response,
            self._workspace_root,
            self._render_prompt,
            self._fetch_response,
            self._parser,
        )

        # Update context with potentially modified values
        ctx.response = response
        ctx.prepared = prepared
        ctx.raw_response = raw

        return ctx


class CircuitBreakerMiddleware(BaseMiddleware[R]):
    """Middleware for safety monitoring and circuit breaking.

    Enforces token budget and file churn limits to prevent runaway agents.
    """

    def __init__(self, *, persona: str) -> None:
        self._persona = persona

    async def after_step(self, ctx: StepContext[R]) -> StepContext[R]:
        """Check circuit breaker after response."""
        if ctx.usage_snapshot is None:
            return ctx

        # This may raise SafetyLockdownError
        enforce_circuit_breaker(
            usage=ctx.usage_snapshot,
            files_touched=ctx.files_touched,
            persona=self._persona,
            context="agent_response",
        )

        return ctx


async def run_middleware_pipeline(
    middlewares: Sequence[AgentMiddleware[R]],
    ctx: StepContext[R],
    *,
    phase: MiddlewarePhase,
) -> StepContext[R]:
    """Execute middleware in order for given phase.

    Args:
        middlewares: List of middleware to execute.
        ctx: Current step context.
        phase: BEFORE for pre-step, AFTER for post-step.

    Returns:
        Modified context after all middleware.
    """
    for mw in middlewares:
        if phase == MiddlewarePhase.BEFORE:
            ctx = await mw.before_step(ctx)
        else:
            ctx = await mw.after_step(ctx)
    return ctx


__all__ = [
    "AgentMiddleware",
    "BaseMiddleware",
    "CircuitBreakerMiddleware",
    "GovernanceMiddleware",
    "MiddlewarePhase",
    "StepContext",
    "TracingMiddleware",
    "run_middleware_pipeline",
]
