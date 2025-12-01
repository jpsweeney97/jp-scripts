"""MCP tools package with auto-discovery.

This package contains all MCP tool implementations:
    - filesystem: File read/write/search
    - git: Repository operations
    - memory: Vector store queries
    - system: Shell command execution
    - And more...
"""

from __future__ import annotations

import pkgutil
import warnings
from importlib import import_module

from jpscripts.core.mcp_registry import (
    TOOL_METADATA_ATTR,
    ToolFunction,
    is_mcp_tool,
)

_PACKAGE_NAME = "jpscripts.mcp.tools"
# Re-export for backwards compatibility
_TOOL_METADATA_ATTR = TOOL_METADATA_ATTR
_is_mcp_tool = is_mcp_tool


def _discover_tool_module_names() -> list[str]:
    """Dynamically discover tool module names using pkgutil.

    Handles both normal package installations and zipapp/compiled scenarios
    where __path__ may not exist or be empty.

    Returns:
        List of fully qualified module names (e.g., 'jpscripts.mcp.tools.filesystem').
    """
    try:
        package = import_module(_PACKAGE_NAME)
    except ImportError as exc:
        warnings.warn(
            f"Failed to import {_PACKAGE_NAME}: {exc}. Tool discovery disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # Get __path__ - may be None in frozen/zipapp scenarios
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        warnings.warn(
            f"Package {_PACKAGE_NAME} has no __path__. Falling back to empty tool list.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # Convert to list if needed (zipimport uses custom iterables)
    try:
        path_list = list(package_path)
    except TypeError:
        warnings.warn(
            f"Package {_PACKAGE_NAME}.__path__ is not iterable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    if not path_list:
        return []

    # Discover modules
    modules: list[str] = []
    try:
        for module_info in pkgutil.iter_modules(path_list, prefix=f"{_PACKAGE_NAME}."):
            # Skip private modules (starting with underscore)
            module_name = module_info.name.split(".")[-1]
            if module_name.startswith("_"):
                continue
            modules.append(module_info.name)
    except Exception as exc:
        warnings.warn(
            f"Error during tool discovery: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    return sorted(modules)


def discover_tool_module_names() -> list[str]:
    """Public wrapper around tool module discovery."""
    return _discover_tool_module_names()


# _is_mcp_tool is now aliased from core.mcp_registry.is_mcp_tool above


def discover_tools() -> dict[str, ToolFunction]:
    """Discover all MCP tools from tool modules.

    Scans all tool modules in jpscripts.mcp.tools for functions decorated
    with @tool and returns a dictionary mapping tool names to their callables.

    Returns:
        Dictionary mapping tool_name -> async tool function.
    """
    tools: dict[str, ToolFunction] = {}
    module_names = discover_tool_module_names()

    for module_name in module_names:
        try:
            module = import_module(module_name)
        except ImportError as exc:
            warnings.warn(
                f"Failed to import tool module {module_name}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Scan module for @tool decorated functions
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(module, attr_name, None)
            if _is_mcp_tool(obj):
                tool_func = obj
                tool_name = getattr(tool_func, "__name__", attr_name)
                if tool_name in tools:
                    warnings.warn(
                        f"Duplicate tool name '{tool_name}' in {module_name}; "
                        f"previous registration from {tools[tool_name].__module__} will be overwritten.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                tools[tool_name] = tool_func

    return tools


# Legacy export for backward compatibility during migration
TOOL_MODULES: list[str] = discover_tool_module_names()

__all__ = ["TOOL_MODULES", "ToolFunction", "discover_tool_module_names", "discover_tools"]
