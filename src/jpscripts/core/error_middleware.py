"""
Centralized error formatting for CLI, MCP, and agent contexts.

This module provides consistent error formatting across different execution
contexts, ensuring errors are presented appropriately for each interface.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from jpscripts.core.result import (
    Err,
    JPScriptsError,
    ModelProviderError,
    Result,
    SecurityError,
    ToolExecutionError,
    ValidationError,
    WorkspaceError,
)


class ErrorSeverity(Enum):
    """Severity levels for error display."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass(frozen=True, slots=True)
class FormattedError:
    """A formatted error ready for display."""

    message: str
    severity: ErrorSeverity
    code: str
    details: dict[str, Any]
    traceback: str | None = None


def _error_code(exc: Exception) -> str:
    """Derive an error code from exception type."""
    if isinstance(exc, SecurityError):
        return "SECURITY_ERROR"
    if isinstance(exc, ValidationError):
        return "VALIDATION_ERROR"
    if isinstance(exc, ToolExecutionError):
        return "TOOL_ERROR"
    if isinstance(exc, ModelProviderError):
        return "PROVIDER_ERROR"
    if isinstance(exc, WorkspaceError):
        return "WORKSPACE_ERROR"
    if isinstance(exc, JPScriptsError):
        return "JP_ERROR"
    if isinstance(exc, FileNotFoundError):
        return "FILE_NOT_FOUND"
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED"
    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    return "UNEXPECTED_ERROR"


def _severity(exc: Exception) -> ErrorSeverity:
    """Determine severity based on exception type."""
    if isinstance(exc, SecurityError):
        return ErrorSeverity.CRITICAL
    if isinstance(exc, (ValidationError, FileNotFoundError)):
        return ErrorSeverity.WARNING
    if isinstance(exc, ToolExecutionError):
        return ErrorSeverity.ERROR
    return ErrorSeverity.ERROR


def format_error(
    exc: Exception,
    *,
    include_traceback: bool = False,
) -> FormattedError:
    """Format an exception into a structured error.

    Args:
        exc: The exception to format
        include_traceback: Whether to include full traceback (for debugging)

    Returns:
        FormattedError ready for display
    """
    code = _error_code(exc)
    severity = _severity(exc)

    details: dict[str, Any] = {}
    if isinstance(exc, JPScriptsError):
        details = exc.context.copy()

    if isinstance(exc, OSError):
        if exc.filename:
            details["path"] = str(exc.filename)
        if exc.errno:
            details["errno"] = exc.errno

    tb = None
    if include_traceback:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    return FormattedError(
        message=str(exc),
        severity=severity,
        code=code,
        details=details,
        traceback=tb,
    )


# ---------------------------------------------------------------------------
# CLI Formatting (Rich panels with colors)
# ---------------------------------------------------------------------------


def format_for_cli(error: FormattedError) -> str:
    """Format error for CLI display with Rich markup.

    Returns a string with Rich markup for colored output.
    """
    color_map = {
        ErrorSeverity.INFO: "blue",
        ErrorSeverity.WARNING: "yellow",
        ErrorSeverity.ERROR: "red",
        ErrorSeverity.CRITICAL: "bold red",
    }
    color = color_map.get(error.severity, "red")

    parts = [f"[{color}]{error.code}[/{color}]: {error.message}"]

    if error.details:
        detail_lines = [f"  {k}: {v}" for k, v in error.details.items()]
        parts.append("\n".join(detail_lines))

    if error.traceback:
        parts.append(f"\n[dim]{error.traceback}[/dim]")

    return "\n".join(parts)


def format_for_cli_panel(error: FormattedError) -> dict[str, Any]:
    """Return parameters for a Rich Panel displaying the error.

    Usage:
        from rich.panel import Panel
        panel = Panel(**format_for_cli_panel(error))
    """
    color_map = {
        ErrorSeverity.INFO: "blue",
        ErrorSeverity.WARNING: "yellow",
        ErrorSeverity.ERROR: "red",
        ErrorSeverity.CRITICAL: "bold red",
    }
    color = color_map.get(error.severity, "red")

    body_parts = [error.message]
    if error.details:
        detail_lines = [f"{k}: {v}" for k, v in error.details.items()]
        body_parts.append("\n".join(detail_lines))

    return {
        "renderable": "\n\n".join(body_parts),
        "title": error.code,
        "border_style": color,
    }


# ---------------------------------------------------------------------------
# MCP Formatting (JSON error responses)
# ---------------------------------------------------------------------------


def format_for_mcp(error: FormattedError) -> str:
    """Format error as JSON for MCP protocol.

    Returns a JSON string suitable for MCP tool responses.
    """
    payload = {
        "error": error.code,
        "message": error.message,
    }

    if error.details:
        payload["details"] = error.details

    return json.dumps(payload)


def format_exception_for_mcp(exc: Exception) -> str:
    """Convenience function to format an exception directly for MCP."""
    formatted = format_error(exc, include_traceback=False)
    return format_for_mcp(formatted)


# ---------------------------------------------------------------------------
# Agent Formatting (Structured context for retry logic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentErrorContext:
    """Structured error context for agent retry logic."""

    code: str
    message: str
    recoverable: bool
    suggested_action: str | None
    details: dict[str, Any]


def format_for_agent(error: FormattedError) -> AgentErrorContext:
    """Format error for agent consumption with retry guidance.

    Returns structured context to help the agent decide how to recover.
    """
    # Determine if error is recoverable
    recoverable = error.severity in {ErrorSeverity.WARNING, ErrorSeverity.ERROR}
    if error.code == "SECURITY_ERROR":
        recoverable = False
    if error.code == "VALIDATION_ERROR":
        recoverable = True

    # Suggest recovery action
    action = None
    if error.code == "FILE_NOT_FOUND":
        action = "Use list_directory to verify path exists before retrying"
    elif error.code == "PERMISSION_DENIED":
        action = "Check file permissions or try a different path"
    elif error.code == "VALIDATION_ERROR":
        action = "Review input parameters and correct values"
    elif error.code == "TOOL_ERROR":
        action = "Review tool output and try alternative approach"
    elif error.code == "PROVIDER_ERROR":
        action = "Wait and retry, or try a different model"

    return AgentErrorContext(
        code=error.code,
        message=error.message,
        recoverable=recoverable,
        suggested_action=action,
        details=error.details,
    )


def format_exception_for_agent(exc: Exception) -> AgentErrorContext:
    """Convenience function to format an exception directly for agent."""
    formatted = format_error(exc, include_traceback=False)
    return format_for_agent(formatted)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def result_to_cli(result: Result) -> str:
    """Format a Result for CLI output."""
    if result.is_ok():
        return str(result.unwrap())
    error = format_error(result.error)
    return format_for_cli(error)


def result_to_mcp(result: Result) -> str:
    """Format a Result for MCP output."""
    if result.is_ok():
        value = result.unwrap()
        if isinstance(value, str):
            return value
        return json.dumps(value)
    error = format_error(result.error)
    return format_for_mcp(error)


__all__ = [
    # Types
    "ErrorSeverity",
    "FormattedError",
    "AgentErrorContext",
    # Formatting functions
    "format_error",
    "format_for_cli",
    "format_for_cli_panel",
    "format_for_mcp",
    "format_for_agent",
    "format_exception_for_mcp",
    "format_exception_for_agent",
    # Result helpers
    "result_to_cli",
    "result_to_mcp",
]
