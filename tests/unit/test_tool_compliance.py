from __future__ import annotations

import inspect
from typing import Any

import pytest

from jpscripts.mcp.tools import discover_tools


def test_all_mcp_tools_are_strictly_typed() -> None:
    """Ensure all tools in the unified registry have proper type annotations."""
    tools = discover_tools()
    assert tools, "discover_tools() returned empty; tool discovery failed."

    issues: list[str] = []
    for tool_name, func in tools.items():
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                issues.append(f"Tool '{tool_name}' missing type hint for argument '{name}'.")
            elif param.annotation is Any:
                issues.append(f"Tool '{tool_name}' uses Any for argument '{name}'.")
        if sig.return_annotation is inspect.Signature.empty:
            issues.append(f"Tool '{tool_name}' missing return type annotation.")
        elif sig.return_annotation is Any:
            issues.append(f"Tool '{tool_name}' uses Any as return type.")
        if not getattr(func, "__tool_error_handler__", False):
            issues.append(f"Tool '{tool_name}' is missing @tool_error_handler wrapping.")

    if issues:
        pytest.fail("\n".join(issues))


def test_unified_registry_consistency() -> None:
    """Ensure discover_tools() returns deterministic results."""
    first_call = discover_tools()
    second_call = discover_tools()

    assert set(first_call.keys()) == set(second_call.keys()), (
        "discover_tools() returned different tool sets on consecutive calls"
    )
