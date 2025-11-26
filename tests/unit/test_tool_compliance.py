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

    if issues:
        pytest.fail("\n".join(issues))
