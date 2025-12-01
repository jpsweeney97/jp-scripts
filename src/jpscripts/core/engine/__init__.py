"""Agent engine subsystem for jpscripts.

This package provides the core agent execution engine including:
- AgentEngine: Main agent loop with governance, tracing, and safety
- Response parsing and validation
- Tool execution
- Trace recording
- Safety monitoring

Public API:
- AgentEngine: Main agent execution class
- parse_agent_response: Parse agent JSON responses
- run_safe_shell: Execute commands safely
- load_template_environment: Load Jinja2 templates
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic

from pydantic import BaseModel

from jpscripts.core.cost_tracker import TokenUsage

from .governance_enforcer import enforce_governance
from .models import (
    AgentResponse,
    AgentTraceStep,
    MemoryProtocol,
    Message,
    PreparedPrompt,
    ResponseT,
    SafetyLockdownError,
    ToolCall,
)
from .response_handler import (
    _clean_json_payload,
    _extract_balanced_json,
    _extract_from_code_fence,
    _extract_thinking_content,
    _find_last_valid_json,
    _split_thought_and_json,
    parse_agent_response,
)
from .safety_monitor import (
    _approximate_tokens,
    _build_black_box_report,
    _estimate_token_usage,
    _get_tokenizer,
    enforce_circuit_breaker,
)
from .tool_executor import (
    AUDIT_PREFIX,
    execute_tool,
    load_template_environment,
    run_safe_shell,
)
from .trace_recorder import (
    TraceRecorder,
    _get_tracer,
    _load_otel_deps,
)


class AgentEngine(Generic[ResponseT]):
    """Main agent execution engine with governance, tracing, and safety.

    This class composes the extracted modules to provide:
    - Response fetching and parsing
    - Governance enforcement
    - Safety monitoring (circuit breaker)
    - Trace recording
    - Tool execution
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

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> ResponseT:
        prepared = await self._render_prompt(history)
        raw = await self._fetch_response(prepared)
        response = self._parser(raw)

        # Apply governance check if enabled and workspace_root is set
        if self._governance_enabled and self._workspace_root is not None:
            response, prepared, raw = await enforce_governance(
                response,
                history,
                prepared,
                raw,
                self._workspace_root,
                self._render_prompt,
                self._fetch_response,
                self._parser,
            )

        usage_snapshot = _estimate_token_usage(prepared.prompt, raw)
        files_touched = await self._infer_files_touched(response)
        self._last_usage_snapshot = usage_snapshot
        self._last_files_touched = files_touched

        enforce_circuit_breaker(
            usage=usage_snapshot,
            files_touched=files_touched,
            persona=self.persona,
            context="agent_response",
        )

        await self._record_trace(history, response)
        return response

    async def _record_trace(
        self, history: Sequence[Message], response: BaseModel, tool_output: str | None = None
    ) -> None:
        from jpscripts.core.console import get_logger

        logger = get_logger(__name__)
        try:
            step = AgentTraceStep(
                timestamp=datetime.now(UTC).isoformat(),
                agent_persona=self.persona,
                input_history=[{"role": msg.role, "content": msg.content} for msg in history],
                response=response.model_dump(),
                tool_output=tool_output,
            )
            await self._trace_recorder.append(step)
            tracer = _get_tracer()
            if tracer is not None:
                files_touched = [str(path) for path in self._last_files_touched]
                usage_snapshot = self._last_usage_snapshot
                with tracer.start_as_current_span("agent.turn") as span:
                    span.set_attribute("agent.persona", self.persona)
                    if files_touched:
                        span.set_attribute("code.files_touched", files_touched)
                    if usage_snapshot is not None:
                        span.set_attribute("usage.prompt_tokens", usage_snapshot.prompt_tokens)
                        span.set_attribute(
                            "usage.completion_tokens", usage_snapshot.completion_tokens
                        )
                        span.set_attribute("usage.total_tokens", usage_snapshot.total_tokens)
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call is not None:
                        span.add_event(
                            "tool_call",
                            {
                                "tool_call": tool_call.model_dump()
                                if hasattr(tool_call, "model_dump")
                                else str(tool_call)
                            },
                        )
                    if tool_output:
                        span.add_event("tool_output", {"output": tool_output})
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to record trace: %s", exc)

    async def _infer_files_touched(self, response: BaseModel) -> list[Path]:
        if not hasattr(response, "file_patch"):
            return []

        file_patch = getattr(response, "file_patch", None)
        if not file_patch or self._workspace_root is None:
            return []

        # Lazy import to avoid circular dependency
        from jpscripts.core.agent.patching import extract_patch_paths

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


# Re-export everything for backwards compatibility
__all__ = [
    # Tool execution
    "AUDIT_PREFIX",
    # Engine class
    "AgentEngine",
    # Models
    "AgentResponse",
    "AgentTraceStep",
    "MemoryProtocol",
    "Message",
    "PreparedPrompt",
    "ResponseT",
    "SafetyLockdownError",
    "ToolCall",
    # Trace recording
    "TraceRecorder",
    # Safety
    "_approximate_tokens",
    "_build_black_box_report",
    # Response handling
    "_clean_json_payload",
    "_estimate_token_usage",
    "_extract_balanced_json",
    "_extract_from_code_fence",
    "_extract_thinking_content",
    "_find_last_valid_json",
    "_get_tokenizer",
    "_get_tracer",
    "_load_otel_deps",
    "_split_thought_and_json",
    "enforce_circuit_breaker",
    # Governance
    "enforce_governance",
    "execute_tool",
    "load_template_environment",
    "parse_agent_response",
    "run_safe_shell",
]
