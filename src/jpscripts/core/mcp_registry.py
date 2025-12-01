from __future__ import annotations

import functools
import warnings
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeGuard, TypeVar

from pydantic import ValidationError, validate_call


class ToolValidationError(RuntimeError):
    """Raised when an MCP tool receives invalid input."""


P = ParamSpec("P")
R = TypeVar("R")

# Type alias for tool functions (matches jpscripts.mcp.tools.ToolFunction)
ToolFunction = Callable[..., Awaitable[str]]

# Single source of truth for the tool metadata attribute name
TOOL_METADATA_ATTR = "__mcp_tool_metadata__"


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

    return wrapper


def is_mcp_tool(obj: Any) -> TypeGuard[ToolFunction]:
    """Check if an object is decorated with @tool.

    This is the single source of truth for determining if a callable
    is an MCP tool, used by tool discovery.
    """
    if not callable(obj):
        return False
    return hasattr(obj, TOOL_METADATA_ATTR)


def get_tool_metadata(obj: object) -> dict[str, Any] | None:
    """Return metadata for decorated tool functions.

    This is the single source of truth for accessing tool metadata,
    used by both MCP server and CLI commands.
    """
    if not callable(obj):
        return None

    metadata = getattr(obj, TOOL_METADATA_ATTR, None)
    if metadata is None:
        return None

    if metadata == {}:
        return {}

    return metadata if isinstance(metadata, dict) else None


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


__all__ = [
    "TOOL_METADATA_ATTR",
    "ToolFunction",
    "ToolValidationError",
    "get_tool_metadata",
    "get_tool_registry",
    "import_tool_modules",
    "is_mcp_tool",
    "iter_mcp_tools",
    "strict_tool_validator",
]
