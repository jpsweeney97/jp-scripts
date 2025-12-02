"""MCP tools package with auto-discovery.

This package contains all MCP tool implementations:
    - filesystem: File read/write/search
    - git: Repository operations
    - memory: Vector store queries
    - system: Shell command execution
    - And more...

Note: Tool discovery has been moved to jpscripts.capabilities.registry.
This module re-exports for backward compatibility.
"""

from __future__ import annotations

# Re-export from canonical location
from jpscripts.capabilities.registry import (
    TOOL_METADATA_ATTR,
    ToolFunction,
    discover_tool_module_names,
    discover_tools,
    get_tool_metadata,
    is_mcp_tool,
)

# Legacy aliases for backward compatibility
_TOOL_METADATA_ATTR = TOOL_METADATA_ATTR
_is_mcp_tool = is_mcp_tool

# Legacy export - compute on import
TOOL_MODULES: list[str] = discover_tool_module_names()

__all__ = [
    "TOOL_METADATA_ATTR",
    "TOOL_MODULES",
    "ToolFunction",
    "discover_tool_module_names",
    "discover_tools",
    "get_tool_metadata",
    "is_mcp_tool",
]
