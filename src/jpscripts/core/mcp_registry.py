from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any, ParamSpec, TypeVar

from pydantic import ValidationError, validate_call


class ToolValidationError(RuntimeError):
    """Raised when an MCP tool receives invalid input."""


P = ParamSpec("P")
R = TypeVar("R")


def strict_tool_validator(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Wrap a callable with pydantic runtime validation, re-raising validation errors
    as ToolValidationError to preserve the MCP error boundary.
    """
    validated = validate_call(config={"strict": True})(fn)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return validated(*args, **kwargs)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper


def import_tool_modules(module_names: Iterable[str]) -> list[ModuleType]:
    """Import MCP tool modules, skipping failures gracefully."""
    modules: list[ModuleType] = []
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception:
            # Errors are logged by callers using richer context; ignore here.
            continue
    return modules


def iter_mcp_tools(
    modules: Iterable[ModuleType],
    *,
    metadata_extractor: Callable[[object], dict[str, Any] | None],
) -> Iterable[tuple[str, Callable[..., Any]]]:
    """
    Yield (name, callable) for decorated MCP tools within the provided modules.

    Args:
        modules: Iterable of imported modules to scan.
        metadata_extractor: Callable that returns metadata for a candidate function
            (typically jpscripts.mcp.get_tool_metadata).
    """
    for module in modules:
        for candidate in module.__dict__.values():
            metadata = metadata_extractor(candidate)
            if metadata is None:
                continue
            name = getattr(candidate, "__name__", None)
            if not name:
                continue
            if not callable(candidate):
                continue
            yield name, candidate
