"""Tests for core/error_middleware.py - centralized error formatting."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jpscripts.core.error_middleware import (
    AgentErrorContext,
    ErrorSeverity,
    FormattedError,
    format_error,
    format_exception_for_agent,
    format_exception_for_mcp,
    format_for_agent,
    format_for_cli,
    format_for_cli_panel,
    format_for_mcp,
    result_to_cli,
    result_to_mcp,
)
from jpscripts.core.result import (
    Err,
    JPScriptsError,
    ModelProviderError,
    Ok,
    SecurityError,
    ToolExecutionError,
    ValidationError,
    WorkspaceError,
)

# ---------------------------------------------------------------------------
# Test _error_code (via format_error)
# ---------------------------------------------------------------------------


class TestErrorCode:
    """Test error code derivation from exception types."""

    def test_security_error_code(self) -> None:
        exc = SecurityError("path traversal attempt")
        formatted = format_error(exc)
        assert formatted.code == "SECURITY_ERROR"

    def test_validation_error_code(self) -> None:
        exc = ValidationError("invalid input")
        formatted = format_error(exc)
        assert formatted.code == "VALIDATION_ERROR"

    def test_tool_execution_error_code(self) -> None:
        exc = ToolExecutionError("command failed")
        formatted = format_error(exc)
        assert formatted.code == "TOOL_ERROR"

    def test_model_provider_error_code(self) -> None:
        exc = ModelProviderError("API rate limited")
        formatted = format_error(exc)
        assert formatted.code == "PROVIDER_ERROR"

    def test_workspace_error_code(self) -> None:
        exc = WorkspaceError("not a git repo")
        formatted = format_error(exc)
        assert formatted.code == "WORKSPACE_ERROR"

    def test_jp_scripts_error_code(self) -> None:
        exc = JPScriptsError("generic jp error")
        formatted = format_error(exc)
        assert formatted.code == "JP_ERROR"

    def test_file_not_found_error_code(self) -> None:
        exc = FileNotFoundError("missing.txt")
        formatted = format_error(exc)
        assert formatted.code == "FILE_NOT_FOUND"

    def test_permission_error_code(self) -> None:
        exc = PermissionError("access denied")
        formatted = format_error(exc)
        assert formatted.code == "PERMISSION_DENIED"

    def test_timeout_error_code(self) -> None:
        exc = TimeoutError("operation timed out")
        formatted = format_error(exc)
        assert formatted.code == "TIMEOUT"

    def test_unexpected_error_code(self) -> None:
        exc = RuntimeError("something unexpected")
        formatted = format_error(exc)
        assert formatted.code == "UNEXPECTED_ERROR"


# ---------------------------------------------------------------------------
# Test _severity (via format_error)
# ---------------------------------------------------------------------------


class TestErrorSeverity:
    """Test severity determination from exception types."""

    def test_security_error_is_critical(self) -> None:
        exc = SecurityError("breach attempt")
        formatted = format_error(exc)
        assert formatted.severity == ErrorSeverity.CRITICAL

    def test_validation_error_is_warning(self) -> None:
        exc = ValidationError("bad input")
        formatted = format_error(exc)
        assert formatted.severity == ErrorSeverity.WARNING

    def test_file_not_found_is_warning(self) -> None:
        exc = FileNotFoundError("missing")
        formatted = format_error(exc)
        assert formatted.severity == ErrorSeverity.WARNING

    def test_tool_execution_error_is_error(self) -> None:
        exc = ToolExecutionError("failed")
        formatted = format_error(exc)
        assert formatted.severity == ErrorSeverity.ERROR

    def test_generic_exception_is_error(self) -> None:
        exc = RuntimeError("unknown")
        formatted = format_error(exc)
        assert formatted.severity == ErrorSeverity.ERROR


# ---------------------------------------------------------------------------
# Test format_error
# ---------------------------------------------------------------------------


class TestFormatError:
    """Test the main format_error function."""

    def test_basic_exception_formatting(self) -> None:
        exc = ValueError("invalid value")
        formatted = format_error(exc)
        assert formatted.message == "invalid value"
        assert formatted.code == "UNEXPECTED_ERROR"
        assert formatted.details == {}
        assert formatted.traceback is None

    def test_jpscripts_error_includes_context(self) -> None:
        exc = SecurityError("path escape", context={"path": "/etc/passwd", "root": "/home"})
        formatted = format_error(exc)
        assert formatted.details["path"] == "/etc/passwd"
        assert formatted.details["root"] == "/home"

    def test_oserror_includes_filename(self) -> None:
        exc = OSError(2, "No such file", "/path/to/file.txt")
        formatted = format_error(exc)
        assert formatted.details["path"] == "/path/to/file.txt"
        assert formatted.details["errno"] == 2

    def test_oserror_without_filename(self) -> None:
        exc = OSError("generic OS error")
        formatted = format_error(exc)
        assert "path" not in formatted.details

    def test_include_traceback_true(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError as exc:
            formatted = format_error(exc, include_traceback=True)
            assert formatted.traceback is not None
            assert "ValueError" in formatted.traceback
            assert "test error" in formatted.traceback

    def test_include_traceback_false(self) -> None:
        exc = ValueError("test")
        formatted = format_error(exc, include_traceback=False)
        assert formatted.traceback is None


# ---------------------------------------------------------------------------
# Test format_for_cli
# ---------------------------------------------------------------------------


class TestFormatForCli:
    """Test CLI formatting with Rich markup."""

    def test_info_severity_uses_blue(self) -> None:
        error = FormattedError(
            message="informational",
            severity=ErrorSeverity.INFO,
            code="INFO_CODE",
            details={},
        )
        result = format_for_cli(error)
        assert "[blue]INFO_CODE[/blue]" in result

    def test_warning_severity_uses_yellow(self) -> None:
        error = FormattedError(
            message="warning message",
            severity=ErrorSeverity.WARNING,
            code="WARN_CODE",
            details={},
        )
        result = format_for_cli(error)
        assert "[yellow]WARN_CODE[/yellow]" in result

    def test_error_severity_uses_red(self) -> None:
        error = FormattedError(
            message="error message",
            severity=ErrorSeverity.ERROR,
            code="ERR_CODE",
            details={},
        )
        result = format_for_cli(error)
        assert "[red]ERR_CODE[/red]" in result

    def test_critical_severity_uses_bold_red(self) -> None:
        error = FormattedError(
            message="critical issue",
            severity=ErrorSeverity.CRITICAL,
            code="CRIT_CODE",
            details={},
        )
        result = format_for_cli(error)
        assert "[bold red]CRIT_CODE[/bold red]" in result

    def test_includes_message(self) -> None:
        error = FormattedError(
            message="the actual message",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        result = format_for_cli(error)
        assert "the actual message" in result

    def test_includes_details_when_present(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={"key1": "value1", "key2": "value2"},
        )
        result = format_for_cli(error)
        assert "key1: value1" in result
        assert "key2: value2" in result

    def test_includes_traceback_when_present(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
            traceback="Traceback (most recent call last):\n  File ...",
        )
        result = format_for_cli(error)
        assert "[dim]Traceback" in result


# ---------------------------------------------------------------------------
# Test format_for_cli_panel
# ---------------------------------------------------------------------------


class TestFormatForCliPanel:
    """Test Rich Panel parameter generation."""

    def test_returns_panel_params(self) -> None:
        error = FormattedError(
            message="panel message",
            severity=ErrorSeverity.ERROR,
            code="PANEL_CODE",
            details={},
        )
        params = format_for_cli_panel(error)
        assert "renderable" in params
        assert "title" in params
        assert "border_style" in params

    def test_title_is_error_code(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="MY_CODE",
            details={},
        )
        params = format_for_cli_panel(error)
        assert params["title"] == "MY_CODE"

    def test_border_style_matches_severity(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.CRITICAL,
            code="CODE",
            details={},
        )
        params = format_for_cli_panel(error)
        assert params["border_style"] == "bold red"

    def test_renderable_includes_details(self) -> None:
        error = FormattedError(
            message="main message",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={"extra": "info"},
        )
        params = format_for_cli_panel(error)
        assert "main message" in params["renderable"]
        assert "extra: info" in params["renderable"]


# ---------------------------------------------------------------------------
# Test format_for_mcp
# ---------------------------------------------------------------------------


class TestFormatForMcp:
    """Test MCP JSON formatting."""

    def test_returns_valid_json(self) -> None:
        error = FormattedError(
            message="mcp error",
            severity=ErrorSeverity.ERROR,
            code="MCP_CODE",
            details={},
        )
        result = format_for_mcp(error)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_includes_error_code(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="THE_CODE",
            details={},
        )
        result = format_for_mcp(error)
        parsed = json.loads(result)
        assert parsed["error"] == "THE_CODE"

    def test_includes_message(self) -> None:
        error = FormattedError(
            message="the message",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        result = format_for_mcp(error)
        parsed = json.loads(result)
        assert parsed["message"] == "the message"

    def test_includes_details_when_present(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={"detail_key": "detail_value"},
        )
        result = format_for_mcp(error)
        parsed = json.loads(result)
        assert parsed["details"]["detail_key"] == "detail_value"

    def test_omits_details_when_empty(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        result = format_for_mcp(error)
        parsed = json.loads(result)
        assert "details" not in parsed


class TestFormatExceptionForMcp:
    """Test convenience function for MCP exception formatting."""

    def test_formats_exception_directly(self) -> None:
        exc = ValidationError("bad input", context={"field": "name"})
        result = format_exception_for_mcp(exc)
        parsed = json.loads(result)
        assert parsed["error"] == "VALIDATION_ERROR"
        assert parsed["message"] == "bad input [field=name]"


# ---------------------------------------------------------------------------
# Test format_for_agent
# ---------------------------------------------------------------------------


class TestFormatForAgent:
    """Test agent context formatting with recovery guidance."""

    def test_returns_agent_error_context(self) -> None:
        error = FormattedError(
            message="agent error",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        result = format_for_agent(error)
        assert isinstance(result, AgentErrorContext)

    def test_security_error_not_recoverable(self) -> None:
        error = FormattedError(
            message="security violation",
            severity=ErrorSeverity.CRITICAL,
            code="SECURITY_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.recoverable is False

    def test_validation_error_is_recoverable(self) -> None:
        error = FormattedError(
            message="bad input",
            severity=ErrorSeverity.WARNING,
            code="VALIDATION_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.recoverable is True

    def test_warning_severity_is_recoverable(self) -> None:
        error = FormattedError(
            message="warning",
            severity=ErrorSeverity.WARNING,
            code="SOME_WARNING",
            details={},
        )
        result = format_for_agent(error)
        assert result.recoverable is True

    def test_error_severity_is_recoverable(self) -> None:
        error = FormattedError(
            message="error",
            severity=ErrorSeverity.ERROR,
            code="SOME_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.recoverable is True

    def test_file_not_found_suggested_action(self) -> None:
        error = FormattedError(
            message="missing",
            severity=ErrorSeverity.WARNING,
            code="FILE_NOT_FOUND",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is not None
        assert "list_directory" in result.suggested_action

    def test_permission_denied_suggested_action(self) -> None:
        error = FormattedError(
            message="denied",
            severity=ErrorSeverity.ERROR,
            code="PERMISSION_DENIED",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is not None
        assert "permissions" in result.suggested_action.lower()

    def test_validation_error_suggested_action(self) -> None:
        error = FormattedError(
            message="invalid",
            severity=ErrorSeverity.WARNING,
            code="VALIDATION_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is not None
        assert "parameters" in result.suggested_action.lower()

    def test_tool_error_suggested_action(self) -> None:
        error = FormattedError(
            message="failed",
            severity=ErrorSeverity.ERROR,
            code="TOOL_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is not None
        assert "alternative" in result.suggested_action.lower()

    def test_provider_error_suggested_action(self) -> None:
        error = FormattedError(
            message="rate limited",
            severity=ErrorSeverity.ERROR,
            code="PROVIDER_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is not None
        assert "retry" in result.suggested_action.lower()

    def test_unknown_error_no_suggested_action(self) -> None:
        error = FormattedError(
            message="unknown",
            severity=ErrorSeverity.ERROR,
            code="UNEXPECTED_ERROR",
            details={},
        )
        result = format_for_agent(error)
        assert result.suggested_action is None

    def test_includes_details(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={"key": "value"},
        )
        result = format_for_agent(error)
        assert result.details["key"] == "value"


class TestFormatExceptionForAgent:
    """Test convenience function for agent exception formatting."""

    def test_formats_exception_directly(self) -> None:
        exc = ToolExecutionError("command failed")
        result = format_exception_for_agent(exc)
        assert result.code == "TOOL_ERROR"
        assert result.recoverable is True


# ---------------------------------------------------------------------------
# Test result_to_cli and result_to_mcp
# ---------------------------------------------------------------------------


class TestResultToCli:
    """Test Result formatting for CLI."""

    def test_ok_result_returns_value_string(self) -> None:
        result = Ok("success value")
        output = result_to_cli(result)
        assert output == "success value"

    def test_ok_result_with_non_string(self) -> None:
        result = Ok(42)
        output = result_to_cli(result)
        assert output == "42"

    def test_err_result_returns_formatted_error(self) -> None:
        exc = SecurityError("blocked")
        result = Err(exc)
        output = result_to_cli(result)
        assert "SECURITY_ERROR" in output
        assert "blocked" in output


class TestResultToMcp:
    """Test Result formatting for MCP."""

    def test_ok_result_with_string_returns_string(self) -> None:
        result = Ok("raw string")
        output = result_to_mcp(result)
        assert output == "raw string"

    def test_ok_result_with_dict_returns_json(self) -> None:
        result: Ok[dict[str, Any]] = Ok({"key": "value"})
        output = result_to_mcp(result)
        parsed = json.loads(output)
        assert parsed["key"] == "value"

    def test_err_result_returns_formatted_error(self) -> None:
        exc = ValidationError("invalid")
        result = Err(exc)
        output = result_to_mcp(result)
        parsed = json.loads(output)
        assert parsed["error"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Test ErrorSeverity enum
# ---------------------------------------------------------------------------


class TestErrorSeverityEnum:
    """Test ErrorSeverity enum values."""

    def test_all_severities_exist(self) -> None:
        assert ErrorSeverity.INFO is not None
        assert ErrorSeverity.WARNING is not None
        assert ErrorSeverity.ERROR is not None
        assert ErrorSeverity.CRITICAL is not None


# ---------------------------------------------------------------------------
# Test FormattedError dataclass
# ---------------------------------------------------------------------------


class TestFormattedErrorDataclass:
    """Test FormattedError dataclass properties."""

    def test_is_frozen(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        with pytest.raises(AttributeError):
            error.message = "new"  # type: ignore[misc]

    def test_default_traceback_is_none(self) -> None:
        error = FormattedError(
            message="msg",
            severity=ErrorSeverity.ERROR,
            code="CODE",
            details={},
        )
        assert error.traceback is None


# ---------------------------------------------------------------------------
# Test AgentErrorContext dataclass
# ---------------------------------------------------------------------------


class TestAgentErrorContextDataclass:
    """Test AgentErrorContext dataclass properties."""

    def test_is_frozen(self) -> None:
        ctx = AgentErrorContext(
            code="CODE",
            message="msg",
            recoverable=True,
            suggested_action=None,
            details={},
        )
        with pytest.raises(AttributeError):
            ctx.code = "NEW"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        ctx = AgentErrorContext(
            code="CODE",
            message="msg",
            recoverable=False,
            suggested_action="do something",
            details={"key": "val"},
        )
        assert ctx.code == "CODE"
        assert ctx.message == "msg"
        assert ctx.recoverable is False
        assert ctx.suggested_action == "do something"
        assert ctx.details == {"key": "val"}
