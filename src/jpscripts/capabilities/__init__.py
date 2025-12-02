"""Capabilities package - unified tool registry and discovery.

This package provides a single source of truth for tool registration,
discovery, and validation across both MCP server and agent subsystems.
"""

from __future__ import annotations

from jpscripts.capabilities.registry import (
    TOOL_METADATA_ATTR,
    ToolFunction,
    ToolValidationError,
    discover_tool_module_names,
    discover_tools,
    get_tool_metadata,
    get_tool_registry,
    is_mcp_tool,
    strict_tool_validator,
)

__all__ = [
    "TOOL_METADATA_ATTR",
    "ToolFunction",
    "ToolValidationError",
    "discover_tool_module_names",
    "discover_tools",
    "get_tool_metadata",
    "get_tool_registry",
    "is_mcp_tool",
    "strict_tool_validator",
]
