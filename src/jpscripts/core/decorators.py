from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

import typer

from jpscripts.core.config import ConfigError
from jpscripts.core.console import console
from jpscripts.git import GitOperationError

F = TypeVar("F", bound=Callable[..., Any])


def _handle_exception(exc: Exception) -> NoReturn:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code=1)


def handle_exceptions(func: F) -> F:
    """Decorate CLI entrypoints to present friendly errors and exit cleanly."""

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (GitOperationError, ConfigError, PermissionError) as exc:
                _handle_exception(exc)

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (GitOperationError, ConfigError, PermissionError) as exc:
            _handle_exception(exc)

    return sync_wrapper  # type: ignore[return-value]


__all__ = ["handle_exceptions"]
