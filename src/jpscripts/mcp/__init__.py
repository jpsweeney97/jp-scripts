from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

logger = get_logger("mcp")
config: AppConfig | None = None

_TOOL_METADATA_ATTR = "__mcp_tool_metadata__"
ToolCallable = Callable[..., Any]


def set_config(value: AppConfig | None) -> None:
    """Store the loaded configuration for use inside tool modules."""
    global config
    config = value


def get_config() -> AppConfig | None:
    """Return the current configuration."""
    return config


def tool(**metadata: Any) -> Callable[[ToolCallable], ToolCallable]:
    """Mark a function as an MCP tool and attach optional registration metadata."""

    def decorator(fn: ToolCallable) -> ToolCallable:
        setattr(fn, _TOOL_METADATA_ATTR, metadata)
        return fn

    return decorator


def get_tool_metadata(obj: object) -> dict[str, Any] | None:
    """Return metadata for decorated tool functions."""
    if not callable(obj):
        return None

    metadata = getattr(obj, _TOOL_METADATA_ATTR, None)
    if metadata is None:
        return None

    if metadata == {}:
        return {}

    return metadata if isinstance(metadata, dict) else None


__all__ = ["config", "get_config", "get_tool_metadata", "logger", "set_config", "tool"]
