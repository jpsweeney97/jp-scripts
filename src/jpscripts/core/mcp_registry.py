from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

from pydantic import ValidationError, validate_call


class ToolValidationError(RuntimeError):
    """Raised when an MCP tool receives invalid input."""


P = ParamSpec("P")
R = TypeVar("R")

# Type alias for tool functions (matches jpscripts.mcp.tools.ToolFunction)
ToolFunction = Callable[..., Awaitable[str]]


def strict_tool_validator(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Wrap a callable with pydantic runtime validation, re-raising validation errors
    as ToolValidationError to preserve the MCP error boundary.
    """
    validated = validate_call(config={"strict": True})(fn)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return validated(*args, **kwargs)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper


def get_tool_registry() -> dict[str, ToolFunction]:
    """Get the unified tool registry.

    This is the single source of truth for all MCP tools, used by both
    AgentEngine and the MCP server.

    Returns:
        Dictionary mapping tool_name -> async tool function.
    """
    # Import here to avoid circular imports
    from jpscripts.mcp.tools import discover_tools

    return discover_tools()


# ---------------------------------------------------------------------------
# Deprecated functions - kept for backward compatibility during migration
# ---------------------------------------------------------------------------


def import_tool_modules(module_names: object) -> list[object]:
    """DEPRECATED: Use get_tool_registry() instead.

    This function is kept for backward compatibility but always returns
    an empty list. Tool discovery is now handled by discover_tools().
    """
    warnings.warn(
        "import_tool_modules is deprecated; use get_tool_registry() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []


def iter_mcp_tools(
    modules: object,
    *,
    metadata_extractor: object,
) -> list[tuple[str, Callable[..., object]]]:
    """DEPRECATED: Use get_tool_registry() instead.

    This function is kept for backward compatibility but always returns
    an empty list. Tool discovery is now handled by discover_tools().
    """
    warnings.warn(
        "iter_mcp_tools is deprecated; use get_tool_registry() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []
