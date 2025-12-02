"""Agent execution engine.

This module provides the main AgentEngine class which composes:
- Response fetching and parsing
- Governance enforcement
- Safety monitoring (circuit breaker)
- Trace recording
- Tool execution

The engine supports an optional middleware pipeline for extensibility.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Generic

from pydantic import BaseModel

from jpscripts.core.cost_tracker import TokenUsage

from .circuit import _estimate_token_usage
from .middleware import (
    AgentMiddleware,
    CircuitBreakerMiddleware,
    GovernanceMiddleware,
    StepContext,
    TracingMiddleware,
    run_middleware_pipeline,
)
from .models import (
    MemoryProtocol,
    Message,
    PreparedPrompt,
    ResponseT,
    ToolCall,
)
from .patching import extract_patch_paths
from .tools import execute_tool
from .tracing import TraceRecorder


class AgentEngine(Generic[ResponseT]):
    """Main agent execution engine with governance, tracing, and safety.

    This class composes the extracted modules to provide:
    - Response fetching and parsing
    - Governance enforcement via middleware
    - Safety monitoring (circuit breaker) via middleware
    - Trace recording via middleware
    - Tool execution

    The engine uses a middleware pipeline for extensibility. Default middleware
    can be disabled with use_default_middleware=False.
    """

    def __init__(
        self,
        *,
        persona: str,
        model: str,
        prompt_builder: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]],
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
        parser: Callable[[str], ResponseT],
        tools: Mapping[str, Callable[..., Awaitable[str]]] | None = None,
        memory: MemoryProtocol | None = None,
        template_root: Path | None = None,
        trace_dir: Path | None = None,
        workspace_root: Path | None = None,
        governance_enabled: bool = True,
        middleware: Sequence[AgentMiddleware[ResponseT]] | None = None,
        use_default_middleware: bool = True,
    ) -> None:
        self.persona = persona
        self.model = model
        self._prompt_builder = prompt_builder
        self._fetch_response = fetch_response
        self._parser = parser
        # Use unified tool registry if no tools provided
        if tools is not None:
            self._tools: Mapping[str, Callable[..., Awaitable[str]]] = tools
        else:
            from jpscripts.core.mcp_registry import get_tool_registry

            self._tools = get_tool_registry()
        self._memory = memory
        self._template_root = template_root
        self._trace_recorder = TraceRecorder(trace_dir or Path.home() / ".jpscripts" / "traces")
        self._workspace_root = workspace_root
        self._governance_enabled = governance_enabled
        self._last_usage_snapshot: TokenUsage | None = None
        self._last_files_touched: list[Path] = []

        # Build middleware pipeline
        self._middleware: list[AgentMiddleware[ResponseT]] = []

        if use_default_middleware:
            # Default middleware in execution order:
            # 1. Governance (validates responses, may retry)
            if governance_enabled and workspace_root is not None:
                self._middleware.append(
                    GovernanceMiddleware(
                        workspace_root=workspace_root,
                        render_prompt=self._render_prompt,
                        fetch_response=self._fetch_response,
                        parser=parser,
                        enabled=True,
                    )
                )

            # 2. Circuit breaker (enforces safety limits)
            self._middleware.append(CircuitBreakerMiddleware(persona=persona))

            # 3. Tracing (records execution traces)
            self._middleware.append(
                TracingMiddleware(
                    trace_recorder=self._trace_recorder,
                    persona=persona,
                )
            )

        # Add custom middleware after defaults
        if middleware:
            self._middleware.extend(middleware)

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> ResponseT:
        """Execute one agent turn with middleware pipeline.

        The step method:
        1. Renders the prompt
        2. Runs before_step middleware
        3. Fetches and parses response
        4. Computes usage metrics
        5. Runs after_step middleware (governance, circuit breaker, tracing)

        Args:
            history: Conversation history

        Returns:
            Parsed response from the model

        Raises:
            SafetyLockdownError: If circuit breaker is triggered
            ToolExecutionError: If governance violations persist
        """
        # Create context
        ctx: StepContext[ResponseT] = StepContext(history=list(history))

        # Phase 1: Render prompt
        ctx.prepared = await self._render_prompt(history)

        # Phase 2: Run before_step middleware
        ctx = await run_middleware_pipeline(self._middleware, ctx, phase="before")

        # Phase 3: Fetch and parse response (prepared is guaranteed set after render)
        if ctx.prepared is None:
            raise RuntimeError("Prepared prompt was None after middleware - should not happen")
        ctx.raw_response = await self._fetch_response(ctx.prepared)
        ctx.response = self._parser(ctx.raw_response)

        # Phase 4: Compute metrics for middleware
        ctx.usage_snapshot = _estimate_token_usage(ctx.prepared.prompt, ctx.raw_response)
        ctx.files_touched = await self._infer_files_touched(ctx.response)

        # Store for execute_tool access
        self._last_usage_snapshot = ctx.usage_snapshot
        self._last_files_touched = ctx.files_touched

        # Phase 5: Run after_step middleware (governance, circuit breaker, tracing)
        ctx = await run_middleware_pipeline(self._middleware, ctx, phase="after")

        # Return the (potentially modified) response
        if ctx.response is None:
            # Should not happen in normal flow, but handle defensively
            raise RuntimeError("Response was None after middleware pipeline")
        return ctx.response

    async def _infer_files_touched(self, response: BaseModel) -> list[Path]:
        if not hasattr(response, "file_patch"):
            return []

        file_patch = getattr(response, "file_patch", None)
        if not file_patch or self._workspace_root is None:
            return []

        return await extract_patch_paths(str(file_patch), self._workspace_root)

    async def execute_tool(self, call: ToolCall) -> str:
        """Execute a tool from the unified registry."""
        return await execute_tool(
            call,
            self._tools,
            persona=self.persona,
            last_usage=self._last_usage_snapshot,
            last_files_touched=self._last_files_touched,
        )


__all__ = [
    "AgentEngine",
]
