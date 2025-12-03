"""Tool execution and safe shell runner.

This module provides:
- ToolExecutor Protocol for formal tool execution interface
- Tool execution from the unified registry
- Safe shell command execution with validation (str-returning wrapper)
- Template environment loading

The run_safe_shell here is a thin wrapper around core.sys.run_safe_shell
that converts the Result type to a string for agent/LLM consumption.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jinja2 import Environment

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.result import Err, Ok
from jpscripts.core.safety import estimate_tokens_from_args
from jpscripts.core.templates import get_template_environment
from jpscripts.system import run_safe_shell as core_run_safe_shell

from .circuit import enforce_circuit_breaker
from .models import ToolCall


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Specification for an available tool.

    Used by ToolExecutor to describe available tools to the agent.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol defining the interface for tool execution.

    Implementations handle:
    - Discovering available tools
    - Executing tool calls
    - Returning results as strings for LLM consumption

    This protocol enables dependency injection and testing of
    agent components without requiring real tool implementations.
    """

    async def execute(self, call: ToolCall) -> str:
        """Execute a tool call and return the result.

        Args:
            call: The tool call to execute (name + arguments)

        Returns:
            Tool output as a string suitable for LLM consumption
        """
        ...

    def available_tools(self) -> list[ToolSpec]:
        """Return list of available tools.

        Returns:
            List of ToolSpec describing each available tool
        """
        ...


logger = get_logger(__name__)

AUDIT_PREFIX = "audit.shell"


async def execute_tool(
    call: ToolCall,
    tools: Mapping[str, Callable[..., Awaitable[str]]],
    *,
    persona: str,
    last_usage: TokenUsage | None = None,
    last_files_touched: list[Path] | None = None,
) -> str:
    """Execute a tool from the unified registry.

    Tools are discovered from jpscripts.mcp.tools and called with
    arguments unpacked as keyword arguments.

    Args:
        call: The tool call to execute
        tools: Mapping of tool names to async callables
        persona: Agent persona for circuit breaker reporting
        last_usage: Last known token usage for circuit breaker
        last_files_touched: Files touched in previous operations

    Returns:
        Tool output as a string
    """
    normalized = call.tool.strip().lower()
    if normalized not in tools:
        return f"Unknown tool: {call.tool}"

    # Use provided usage or estimate from tool arguments
    if last_usage is not None:
        usage = last_usage
    else:
        estimated = estimate_tokens_from_args(**call.arguments)
        usage = TokenUsage(prompt_tokens=estimated, completion_tokens=0)

    files_touched = list(last_files_touched or [])

    enforce_circuit_breaker(
        usage=usage,
        files_touched=files_touched,
        persona=persona,
        context=f"tool:{normalized}",
    )

    try:
        # Call tool with unpacked kwargs (tools have proper signatures)
        return await tools[normalized](**call.arguments)
    except TypeError as exc:
        # Handle argument mismatch errors gracefully
        return f"Tool '{call.tool}' argument error: {exc}"
    except Exception as exc:
        return f"Tool '{call.tool}' failed: {exc}"


def load_template_environment(template_root: Path) -> Environment:
    """Load a Jinja2 template environment from the given directory.

    This is a convenience wrapper around core.templates.get_template_environment
    that provides backwards compatibility without custom filters.
    """
    return get_template_environment(template_root, with_filters=False)


async def run_safe_shell(
    command: str, root: Path, audit_prefix: str, config: AppConfig | None = None
) -> str:
    """Shared safe shell runner for AgentEngine and MCP.

    This is a thin wrapper around core.sys.run_safe_shell that converts
    the Result type to a string for agent/LLM consumption.

    Uses tokenized command validation to enforce:
    - Allowlisted binaries only (read-only operations)
    - No shell metacharacters (pipes, redirects, etc.)
    - Path validation (no workspace escape)
    - Forbidden flag detection

    Args:
        command: The shell command to execute
        root: Workspace root directory (commands must stay within)
        audit_prefix: Prefix for audit log entries
        config: Optional app config for sandbox settings

    Returns:
        Command output on success, error message on failure
    """
    result = await core_run_safe_shell(command, root, audit_prefix, config)

    match result:
        case Err(error):
            # Log the error with audit prefix
            logger.warning("%s.reject error=%s command=%r", audit_prefix, error, command)
            # Convert to user-friendly error message
            error_str = str(error)
            if "blocked by policy" in error_str.lower():
                return f"SecurityError: {error}"
            if "parse" in error_str.lower():
                return f"Unable to parse command; simplify quoting. ({error})"
            return f"Failed to run command: {error}"
        case Ok(cmd_result):
            if cmd_result.returncode != 0:
                logger.warning(
                    "%s.fail code=%s cmd=%r", audit_prefix, cmd_result.returncode, command
                )
                return f"Command failed with exit code {cmd_result.returncode}"
            combined = (cmd_result.stdout + cmd_result.stderr).strip()
            return combined or "Command produced no output."


__all__ = [
    "AUDIT_PREFIX",
    "ToolExecutor",
    "ToolSpec",
    "execute_tool",
    "load_template_environment",
    "run_safe_shell",
]
