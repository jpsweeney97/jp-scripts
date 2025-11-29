from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

if TYPE_CHECKING:
    class ToolValidationError(Exception):
        ...

    def strict_tool_validator(func: Callable[..., object]) -> Callable[..., object]:
        ...

    def tool() -> Callable[[Callable[..., object]], Callable[..., object]]:
        ...

    def register_tools(mcp: object) -> object:
        ...
else:  # pragma: no cover - runtime imports
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
    """Ensure tools with missing type hints cause RuntimeError during registration."""

    typed_tool = tool()

    @typed_tool
    async def bad(x) -> str:  # type: ignore[no-untyped-def]
        return str(x["value"])

    def fake_discover_tools() -> dict[str, Callable[..., Awaitable[str]]]:
        return {"bad": bad}

    class DummyMCP:
        def __init__(self) -> None:
            self.called = False

        def add_tool(self, func: Callable[..., Awaitable[str]], **_metadata: object) -> None:
            self.called = True

    monkeypatch.setattr("jpscripts.mcp.server.discover_tools", fake_discover_tools)
    mcp = DummyMCP()

    with pytest.raises(RuntimeError, match="missing a type hint"):
        register_tools(mcp)

    assert mcp.called is False
