from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

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
    validated = validate_call(fn, config={"strict": True})

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return validated(*args, **kwargs)
        except ValidationError as exc:
            raise ToolValidationError(str(exc)) from exc

    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper
