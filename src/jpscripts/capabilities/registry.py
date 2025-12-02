"""Unified tool registry and discovery.

This module is the single source of truth for:
- Tool function validation (Pydantic-based)
- Tool metadata storage and retrieval
- Tool discovery from jpscripts.mcp.tools package
- Tool registry access for both MCP server and agent

Usage:
    from jpscripts.capabilities.registry import get_tool_registry, is_mcp_tool

    # Get all available tools
    tools = get_tool_registry()

    # Check if a function is a tool
    if is_mcp_tool(some_func):
        metadata = get_tool_metadata(some_func)
"""

from __future__ import annotations

import functools
import pkgutil
import warnings
from collections.abc import Awaitable, Callable
from importlib import import_module
from typing import Any, ParamSpec, TypeGuard, TypeVar

from pydantic import ValidationError, validate_call


class ToolValidationError(RuntimeError):
    """Raised when an MCP tool receives invalid input."""


P = ParamSpec("P")
R = TypeVar("R")

# Type alias for tool functions
ToolFunction = Callable[..., Awaitable[str]]

# Single source of truth for the tool metadata attribute name
TOOL_METADATA_ATTR = "__mcp_tool_metadata__"

# Package to scan for tool modules
_TOOL_PACKAGE_NAME = "jpscripts.mcp.tools"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def strict_tool_validator(fn: Callable[P, R]) -> Callable[P, R]:
    """Wrap a callable with Pydantic runtime validation.

    Re-raises validation errors as ToolValidationError to preserve
    the MCP error boundary.
    """
    validated = validate_call(config={"strict": True})(fn)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return validated(*args, **kwargs)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

    return wrapper


# ---------------------------------------------------------------------------
# Tool Detection and Metadata
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Module Discovery
# ---------------------------------------------------------------------------


def discover_tool_module_names(package_name: str = _TOOL_PACKAGE_NAME) -> list[str]:
    """Dynamically discover tool module names using pkgutil.

    Handles both normal package installations and zipapp/compiled scenarios
    where __path__ may not exist or be empty.

    Args:
        package_name: The package to scan for modules

    Returns:
        List of fully qualified module names (e.g., 'jpscripts.mcp.tools.filesystem').
    """
    try:
        package = import_module(package_name)
    except ImportError as exc:
        warnings.warn(
            f"Failed to import {package_name}: {exc}. Tool discovery disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # Get __path__ - may be None in frozen/zipapp scenarios
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        warnings.warn(
            f"Package {package_name} has no __path__. Falling back to empty tool list.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    # Convert to list if needed (zipimport uses custom iterables)
    try:
        path_list = list(package_path)
    except TypeError:
        warnings.warn(
            f"Package {package_name}.__path__ is not iterable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    if not path_list:
        return []

    # Discover modules
    modules: list[str] = []
    try:
        for module_info in pkgutil.iter_modules(path_list, prefix=f"{package_name}."):
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


# ---------------------------------------------------------------------------
# Tool Discovery
# ---------------------------------------------------------------------------


def discover_tools(package_name: str = _TOOL_PACKAGE_NAME) -> dict[str, ToolFunction]:
    """Discover all MCP tools from tool modules.

    Scans all tool modules in the specified package for functions decorated
    with @tool and returns a dictionary mapping tool names to their callables.

    Args:
        package_name: The package to scan for tool modules

    Returns:
        Dictionary mapping tool_name -> async tool function.
    """
    tools: dict[str, ToolFunction] = {}
    module_names = discover_tool_module_names(package_name)

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
            if is_mcp_tool(obj):
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


def get_tool_registry() -> dict[str, ToolFunction]:
    """Get the unified tool registry.

    This is the single source of truth for all MCP tools, used by both
    AgentEngine and the MCP server.

    Returns:
        Dictionary mapping tool_name -> async tool function.
    """
    return discover_tools()


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
