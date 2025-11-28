"""
Unified Result types and error hierarchy for jp-scripts.

This module provides:
1. Result[T, E] type for explicit error handling
2. Domain-specific exception hierarchy
3. Helper functions for Result operations

Usage:
    from jpscripts.core.result import Ok, Err, Result, JPScriptsError

    def risky_operation() -> Result[str, SecurityError]:
        if unsafe:
            return Err(SecurityError("Path escapes workspace"))
        return Ok("success")

    result = risky_operation()
    if result.is_ok():
        print(result.value)
    else:
        print(result.error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, overload

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E", bound=Exception)
F = TypeVar("F", bound=Exception)


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Represents a successful result containing a value."""

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        """Return the contained value."""
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Return the contained value (ignores default for Ok)."""
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """Apply a function to the contained value."""
        return Ok(fn(self.value))

    def map_err(self, fn: Callable[[E], F]) -> Ok[T]:
        """No-op for Ok - returns self unchanged."""
        return self

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that may fail."""
        return fn(self.value)


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Represents a failed result containing an error."""

    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        """Raise the contained error."""
        raise self.error

    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default

    def map(self, fn: Callable[[T], U]) -> Err[E]:
        """No-op for Err - returns self unchanged."""
        return self

    def map_err(self, fn: Callable[[E], F]) -> Err[F]:
        """Apply a function to the contained error."""
        return Err(fn(self.error))

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Err[E]:
        """Short-circuit for Err - returns self unchanged."""
        return self


# Type alias for Result
Result = Ok[T] | Err[E]


# ---------------------------------------------------------------------------
# Domain-specific error hierarchy
# ---------------------------------------------------------------------------


class JPScriptsError(Exception):
    """Base exception for all jp-scripts errors.

    All custom exceptions should inherit from this class to enable
    consistent error handling across the codebase.
    """

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


class SecurityError(JPScriptsError):
    """Raised for security violations.

    Examples:
    - Path traversal attempts
    - Workspace escape via symlinks
    - Forbidden command execution
    - Rate limit exceeded
    """

    pass


class ConfigurationError(JPScriptsError):
    """Raised for configuration issues.

    Examples:
    - Missing required config fields
    - Invalid config values
    - Config file parse errors
    """

    pass


class ToolExecutionError(JPScriptsError):
    """Raised when a tool fails to execute.

    Examples:
    - Shell command failed
    - File operation failed
    - External binary not found
    """

    pass


class ModelProviderError(JPScriptsError):
    """Raised when LLM provider fails.

    Examples:
    - API authentication failed
    - Rate limit exceeded
    - Model not available
    - Response parse error
    """

    pass


class ValidationError(JPScriptsError):
    """Raised for input validation failures.

    Examples:
    - Invalid argument types
    - Missing required arguments
    - Value out of range
    """

    pass


class WorkspaceError(JPScriptsError):
    """Raised for workspace-related issues.

    Examples:
    - Workspace not found
    - Not a git repository
    - Permission denied
    """

    pass


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def try_result(fn: Callable[[], T], error_type: type[E] = JPScriptsError) -> Result[T, E]:
    """Execute a function and wrap the result in Ok/Err.

    Args:
        fn: Function to execute
        error_type: Exception type to catch (default: JPScriptsError)

    Returns:
        Ok(value) on success, Err(exception) on failure
    """
    try:
        return Ok(fn())
    except error_type as exc:
        return Err(exc)


def collect_results(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Collect a list of Results into a Result of list.

    Returns Err on first error, Ok(list) if all succeed.
    """
    values: list[T] = []
    for result in results:
        if result.is_err():
            return result  # type: ignore[return-value]
        values.append(result.unwrap())
    return Ok(values)


__all__ = [
    # Result types
    "Ok",
    "Err",
    "Result",
    # Error hierarchy
    "JPScriptsError",
    "SecurityError",
    "ConfigurationError",
    "ToolExecutionError",
    "ModelProviderError",
    "ValidationError",
    "WorkspaceError",
    # Helpers
    "try_result",
    "collect_results",
]
