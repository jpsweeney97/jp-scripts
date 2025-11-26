from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

from mcp.server.fastmcp import FastMCP

from jpscripts.core.config import load_config
from jpscripts.mcp import get_tool_metadata, logger, set_config
from jpscripts.mcp.tools import TOOL_MODULES


def _load_configuration() -> None:
    """Load and store configuration for tools to consume."""
    try:
        cfg, _ = load_config()
    except Exception as exc:
        logger.error("Failed to load config during MCP startup", exc_info=exc)
        set_config(None)
    else:
        set_config(cfg)
        logger.info("MCP Server loaded config from %s", cfg.notes_dir)


def _iter_tools(modules: Iterable[ModuleType]) -> Iterable[tuple[Callable[..., Any], dict[str, Any]]]:
    for module in modules:
        for candidate in module.__dict__.values():
            metadata = get_tool_metadata(candidate)
            if metadata is not None:
                yield candidate, metadata


def _import_tool_modules(module_names: Iterable[str]) -> list[ModuleType]:
    modules: list[ModuleType] = []
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:
            logger.error("Failed to import tool module %s", module_name, exc_info=exc)
    return modules


def register_tools(mcp: FastMCP, module_names: Iterable[str] | None = None) -> None:
    modules = _import_tool_modules(module_names or TOOL_MODULES)
    for func, metadata in _iter_tools(modules):
        try:
            mcp.add_tool(func, **metadata)
        except Exception as exc:
            logger.error("Failed to register tool %s", getattr(func, "__name__", repr(func)), exc_info=exc)


def create_server() -> FastMCP:
    _load_configuration()
    server = FastMCP("jpscripts")
    register_tools(server)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
