from __future__ import annotations

import pkgutil
import warnings
from importlib import import_module

_PACKAGE_NAME = "jpscripts.mcp.tools"


def _discover_tool_modules() -> list[str]:
    """Dynamically discover tool modules using pkgutil.

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
            f"Package {_PACKAGE_NAME} has no __path__. "
            "Falling back to empty tool list.",
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


# Eagerly discover at import time for backward compatibility
TOOL_MODULES: list[str] = _discover_tool_modules()

__all__ = ["TOOL_MODULES"]
