from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from jpscripts.mcp.tools import TOOL_MODULES


def _iter_tools():
    for module_name in TOOL_MODULES:
        module = importlib.import_module(module_name)
        for obj in module.__dict__.values():
            if getattr(obj, "__mcp_tool_metadata__", None) is None:
                continue
            if callable(obj):
                yield module_name, obj


def test_all_mcp_tools_are_strictly_typed() -> None:
    assert TOOL_MODULES, "TOOL_MODULES is empty; cannot perform compliance scan."

    discovered = list(_iter_tools())
    assert discovered, "No MCP tools discovered; ensure TOOL_MODULES lists tool modules."

    issues: list[str] = []
    for module_name, func in discovered:
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                issues.append(f"Tool '{func.__name__}' in '{module_name}' missing type hint for argument '{name}'.")
            elif param.annotation is Any:
                issues.append(f"Tool '{func.__name__}' in '{module_name}' uses Any for argument '{name}'.")
        if sig.return_annotation is inspect.Signature.empty:
            issues.append(f"Tool '{func.__name__}' in '{module_name}' missing return type annotation.")
        elif sig.return_annotation is Any:
            issues.append(f"Tool '{func.__name__}' in '{module_name}' uses Any as return type.")
        if not getattr(func, "__tool_error_handler__", False):
            issues.append(f"Tool '{func.__name__}' in '{module_name}' is missing @tool_error_handler wrapping.")

    if issues:
        pytest.fail("\n".join(issues))


def test_engine_core_tools_registered_in_mcp() -> None:
    """Ensure AgentEngine core tools are present in MCP registry."""
    from jpscripts.core.engine import ENGINE_CORE_TOOLS  # local import to avoid cycles

    discovered = {func.__name__ for _, func in _iter_tools()}
    missing = set(ENGINE_CORE_TOOLS) - discovered
    assert not missing, f"MCP registry missing AgentEngine tools: {', '.join(sorted(missing))}"
