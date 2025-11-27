from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import inspect
from jpscripts.core.config import load_config
from jpscripts.core.engine import ENGINE_CORE_TOOLS
from jpscripts.core.mcp_registry import import_tool_modules, iter_mcp_tools
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


def register_tools(mcp: FastMCP, module_names: Iterable[str] | None = None, engine_tools: set[str] | None = None) -> None:
    modules = import_tool_modules(module_names or TOOL_MODULES)
    registered_names: set[str] = set()
    for name, func in iter_mcp_tools(modules, metadata_extractor=get_tool_metadata):
        metadata = get_tool_metadata(func) or {}
        signature = inspect.signature(func)
        for name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise RuntimeError(
                    f"MCP tool '{getattr(func, '__name__', repr(func))}' argument '{name}' is missing a type hint."
                )
        registered_names.add(name)
        try:
            mcp.add_tool(func, **metadata)
        except Exception as exc:
            logger.error("Failed to register tool %s", getattr(func, "__name__", repr(func)), exc_info=exc)
    expected = engine_tools or ENGINE_CORE_TOOLS
    missing_engine_tools = expected - registered_names
    if missing_engine_tools:
        logger.warning(
            "MCP tool registry missing AgentEngine tools: %s",
            ", ".join(sorted(missing_engine_tools)),
        )
    extra_tools = registered_names - expected
    if extra_tools:
        logger.warning("MCP registered tools not present in AgentEngine: %s", ", ".join(sorted(extra_tools)))


def create_server() -> FastMCP:
    _load_configuration()
    server = FastMCP("jpscripts")
    register_tools(server, engine_tools=ENGINE_CORE_TOOLS)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
