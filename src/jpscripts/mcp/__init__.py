from __future__ import annotations

import asyncio
import functools
import inspect
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.mcp_registry import (
    TOOL_METADATA_ATTR,
    get_tool_metadata,
    strict_tool_validator,
)
from jpscripts.core.runtime import get_runtime

logger = get_logger("mcp")

P = ParamSpec("P")
R = TypeVar("R")
# Re-export for backwards compatibility
_TOOL_METADATA_ATTR = TOOL_METADATA_ATTR
ToolAsyncCallable = Callable[P, Awaitable[str]]


def set_config(value: AppConfig | None) -> None:
    """DEPRECATED: No-op for backward compatibility.

    Runtime context is now established at entry points via runtime_context().
    This function does nothing but is kept to avoid import errors during migration.
    """
    # No-op: runtime context is the only source of truth
    pass


def get_config() -> AppConfig:
    """Return the current configuration from the runtime context.

    Raises:
        NoRuntimeContextError: If called outside a runtime_context() block.

    Returns:
        The current AppConfig.
    """
    return get_runtime().config


def tool(**metadata: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a function as an MCP tool and attach optional registration metadata."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        wrapped = strict_tool_validator(fn)
        setattr(wrapped, TOOL_METADATA_ATTR, metadata)
        return functools.wraps(fn)(wrapped)

    return decorator


# get_tool_metadata is now imported from core.mcp_registry and re-exported


def _extract_error_path(
    exc: BaseException,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fn: Callable[..., Awaitable[str]],
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
            return _format_error(
                "FileNotFound", str(exc), _extract_error_path(exc, args, kwargs, fn)
            )
        except PermissionError as exc:
            return _format_error(
                "PermissionDenied", str(exc), _extract_error_path(exc, args, kwargs, fn)
            )
        except IsADirectoryError as exc:
            return _format_error(
                "IsADirectory", str(exc), _extract_error_path(exc, args, kwargs, fn)
            )
        except Exception as exc:
            logger.exception("Unhandled error in tool %s", fn.__name__)
            return _format_error(
                "UnexpectedError",
                f"Unexpected error: {exc}",
                _extract_error_path(exc, args, kwargs, fn),
            )

    wrapper.__tool_error_handler__ = True  # type: ignore[attr-defined]  # custom marker attr
    return wrapper


__all__ = [
    "get_config",
    "get_tool_metadata",
    "logger",
    "set_config",
    "tool",
    "tool_error_handler",
]
