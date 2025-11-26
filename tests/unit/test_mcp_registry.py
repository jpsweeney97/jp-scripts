from __future__ import annotations

from types import ModuleType

import pytest

from jpscripts.core.mcp_registry import ToolValidationError, strict_tool_validator
from jpscripts.mcp import tool
from jpscripts.mcp.server import register_tools


def test_strict_tool_validator_raises_tool_error_on_invalid_input() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    validated_add = strict_tool_validator(add)

    with pytest.raises(ToolValidationError):
        _ = validated_add("1", "2")


def test_register_tools_rejects_untyped_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("fake_mod")

    @tool()
    def bad(x):  # type: ignore[no-untyped-def]
        return str(x)

    module.bad = bad

    def fake_import(_names):
        return [module]

    class DummyMCP:
        def __init__(self) -> None:
            self.called = False

        def add_tool(self, func, **_metadata):
            self.called = True

    monkeypatch.setattr("jpscripts.mcp.server._import_tool_modules", fake_import)
    mcp = DummyMCP()

    with pytest.raises(RuntimeError):
        register_tools(mcp, module_names=["fake_mod"])

    assert mcp.called is False
