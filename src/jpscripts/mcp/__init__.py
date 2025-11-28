from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import functools
import inspect
import json
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.mcp_registry import strict_tool_validator

logger = get_logger("mcp")

# Legacy module-level state (deprecated - use runtime context)
# Kept for backward compatibility during migration
_legacy_config: AppConfig | None = None

P = ParamSpec("P")
R = TypeVar("R")
_TOOL_METADATA_ATTR = "__mcp_tool_metadata__"
ToolAsyncCallable = Callable[P, Awaitable[str]]


def set_config(value: AppConfig | None) -> None:
    """Store the loaded configuration for use inside tool modules.

    DEPRECATED: Prefer establishing a runtime_context() at entry points.
    This function maintains backward compatibility during migration.
    """
    global _legacy_config
    _legacy_config = value


def get_config() -> AppConfig | None:
    """Return the current configuration.

    This function first checks the runtime context (preferred), then
    falls back to legacy module-level state for backward compatibility.

    Returns:
        The current AppConfig, or None if not configured.
    """
    # Prefer runtime context if available
    from jpscripts.core.runtime import get_runtime_or_none

    ctx = get_runtime_or_none()
    if ctx is not None:
        return ctx.config

    # Fall back to legacy module-level state
    return _legacy_config


def tool(**metadata: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as an MCP tool and attach optional registration metadata."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        wrapped = strict_tool_validator(fn)
        setattr(wrapped, _TOOL_METADATA_ATTR, metadata)
        return functools.wraps(fn)(wrapped)

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


def _extract_error_path(
    exc: BaseException, args: tuple[Any, ...], kwargs: dict[str, Any], fn: Callable[..., Awaitable[str]]
) -> str:
    candidate: str | Path | None = None

    if isinstance(exc, OSError):
        filename = getattr(exc, "filename", None) or getattr(exc, "filename2", None)
        if filename:
            candidate = filename

    if candidate is None:
        try:
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            path_arg = bound.arguments.get("path")
            if isinstance(path_arg, (str, Path)):
                candidate = path_arg
        except Exception:
            candidate = None

    if candidate is None and "path" in kwargs:
        maybe_path = kwargs["path"]
        if isinstance(maybe_path, (str, Path)):
            candidate = maybe_path

    if candidate is None:
        for arg in args:
            if isinstance(arg, Path):
                candidate = arg
                break

    return str(candidate) if candidate is not None else ""


def _format_error(error_code: str, message: str, path: str) -> str:
    payload = {"error": error_code, "message": message, "path": path}
    return json.dumps(payload)


def tool_error_handler(fn: ToolAsyncCallable[P]) -> ToolAsyncCallable[P]:
    """Decorate a tool to provide consistent error handling.

    Args:
        fn: Asynchronous tool callable to wrap.

    Returns:
        Callable that formats errors as JSON while logging unexpected failures.
    """

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except FileNotFoundError as exc:
            return _format_error("FileNotFound", str(exc), _extract_error_path(exc, args, kwargs, fn))
        except PermissionError as exc:
            return _format_error("PermissionDenied", str(exc), _extract_error_path(exc, args, kwargs, fn))
        except IsADirectoryError as exc:
            return _format_error("IsADirectory", str(exc), _extract_error_path(exc, args, kwargs, fn))
        except Exception as exc:
            logger.exception("Unhandled error in tool %s", fn.__name__)
            return _format_error(
                "UnexpectedError",
                f"Unexpected error: {exc}",
                _extract_error_path(exc, args, kwargs, fn),
            )

    setattr(wrapper, "__tool_error_handler__", True)
    return wrapper


__all__ = [
    "get_config",
    "get_tool_metadata",
    "logger",
    "set_config",
    "tool",
    "tool_error_handler",
]
