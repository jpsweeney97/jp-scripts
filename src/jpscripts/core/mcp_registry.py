"""MCP tool registry and validation.

Note: This module has been moved to jpscripts.capabilities.registry.
This file re-exports for backward compatibility.

Manages registration and metadata for MCP (Model Context Protocol) tools:
    - Tool function validation with Pydantic
    - Tool metadata storage and retrieval
    - Input validation decorators
"""

from __future__ import annotations

import warnings

# Re-export everything from canonical location
from jpscripts.capabilities.registry import (
    TOOL_METADATA_ATTR,
    ToolFunction,
    ToolValidationError,
    get_tool_metadata,
    get_tool_registry,
    is_mcp_tool,
    strict_tool_validator,
)

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
) -> list[tuple[str, object]]:
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
