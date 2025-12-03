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
from jpscripts.core.result import Err, Ok

from .circuit import _estimate_token_usage
from .middleware import (
    AgentMiddleware,
    StepContext,
    run_middleware_pipeline,
)
from .models import (
    AgentError,
    AgentResult,
    MemoryProtocol,
    Message,
    PreparedPrompt,
    ResponseT,
    SafetyLockdownError,
    ToolCall,
)
from .patching import extract_patch_paths
from .tools import execute_tool


class AgentEngine(Generic[ResponseT]):
    """Main agent execution engine with middleware pipeline.

    This class composes the extracted modules to provide:
    - Response fetching and parsing
    - Extensible middleware pipeline
    - Tool execution

    The engine uses a middleware pipeline for features like governance,
    circuit breaker, and tracing. Use build_default_middleware() or
    create_agent() from the factory module to construct the default stack.

    Example:
        from jpscripts.agent.factory import build_default_middleware

        middleware = build_default_middleware(
            persona="Engineer",
            workspace_root=root,
            ...
        )
        engine = AgentEngine(middleware=middleware, ...)
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
        middleware: Sequence[AgentMiddleware[ResponseT]] | None = None,
    ) -> None:
        self.persona = persona
        self.model = model
        self._prompt_builder = prompt_builder
        self._fetch_response = fetch_response
        self._parser = parser
        # Tools must be explicitly provided by caller (dependency injection)
        # Pass get_tool_registry() from jpscripts.core.mcp_registry for full tool support
        # Or pass {} for tool-less operation (testing, trace replay)
        self._tools: Mapping[str, Callable[..., Awaitable[str]]] = (
            tools if tools is not None else {}
        )
        self._memory = memory
        self._template_root = template_root
        self._workspace_root = workspace_root
        self._last_usage_snapshot: TokenUsage | None = None
        self._last_files_touched: list[Path] = []

        # Middleware pipeline (caller provides via factory or manually)
        self._middleware: list[AgentMiddleware[ResponseT]] = list(middleware) if middleware else []

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> AgentResult[ResponseT]:
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
            Ok(response) on success, Err(AgentError) on failure.
            Possible error kinds:
            - "safety": Circuit breaker triggered (SafetyLockdownError)
            - "render": Prompt rendering failed
            - "parse": Response parsing failed
            - "middleware": Middleware pipeline error
        """
        try:
            # Create context
            ctx: StepContext[ResponseT] = StepContext(history=list(history))

            # Phase 1: Render prompt
            ctx.prepared = await self._render_prompt(history)

            # Phase 2: Run before_step middleware
            ctx = await run_middleware_pipeline(self._middleware, ctx, phase="before")

            # Phase 3: Fetch and parse response (prepared is guaranteed set after render)
            if ctx.prepared is None:
                return Err(
                    AgentError(
                        "Prepared prompt was None after middleware",
                        kind="middleware",
                    )
                )
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
                return Err(
                    AgentError(
                        "Response was None after middleware pipeline",
                        kind="middleware",
                    )
                )
            return Ok(ctx.response)

        except SafetyLockdownError as exc:
            return Err(
                AgentError(
                    f"Safety lockdown: {exc.report}",
                    kind="safety",
                    cause=exc,
                )
            )
        except Exception as exc:
            # Catch-all for unexpected errors (parsing, middleware, etc.)
            return Err(
                AgentError(
                    f"Agent step failed: {exc}",
                    kind="unknown",
                    cause=exc,
                )
            )

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
