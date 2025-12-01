"""Tool execution and safe shell runner.

This module provides:
- Tool execution from the unified registry
- Safe shell command execution with validation
- Template environment loading
"""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.result import Err
from jpscripts.core.system import CommandResult, get_sandbox

from .models import ToolCall
from .safety_monitor import enforce_circuit_breaker

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
    from .safety_monitor import _estimate_token_usage

    normalized = call.tool.strip().lower()
    if normalized not in tools:
        return f"Unknown tool: {call.tool}"

    usage = last_usage or _estimate_token_usage("", "")
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
    """Load a Jinja2 template environment from the given directory."""
    return Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)


async def run_safe_shell(
    command: str, root: Path, audit_prefix: str, config: AppConfig | None = None
) -> str:
    """Shared safe shell runner for AgentEngine and MCP.

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
    # Use tokenized validation instead of regex
    verdict, reason = validate_command(command, root)

    if verdict != CommandVerdict.ALLOWED:
        logger.warning(
            "%s.reject verdict=%s reason=%r command=%r",
            audit_prefix,
            verdict.name,
            reason,
            command,
        )
        # Map verdict to user-friendly error message
        if verdict == CommandVerdict.BLOCKED_FORBIDDEN:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_NOT_ALLOWLISTED:
            return f"SecurityError: Command not permitted by policy. {reason}"
        if verdict == CommandVerdict.BLOCKED_PATH_ESCAPE:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_DANGEROUS_FLAG:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_METACHAR:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_UNPARSEABLE:
            return f"Unable to parse command; simplify quoting. ({reason})"
        return f"SecurityError: {reason}"

    # Parse command for execution
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        logger.warning("%s.reject parse_error=%s", audit_prefix, exc)
        return f"Unable to parse command; simplify quoting. ({exc})"

    if not tokens:
        return "Invalid command argument."

    runner = get_sandbox(config)
    run_result = await runner.run_command(tokens, root, env=None)
    if isinstance(run_result, Err):
        logger.warning("%s.reject runner_error=%s", audit_prefix, run_result.error)
        return f"Failed to run command: {run_result.error}"

    result: CommandResult = run_result.value
    if result.returncode != 0:
        logger.warning("%s.fail code=%s cmd=%r", audit_prefix, result.returncode, command)
        return f"Command failed with exit code {result.returncode}"

    combined = (result.stdout + result.stderr).strip()
    return combined or "Command produced no output."


__all__ = [
    "AUDIT_PREFIX",
    "execute_tool",
    "load_template_environment",
    "run_safe_shell",
]
