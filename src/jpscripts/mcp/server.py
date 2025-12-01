"""MCP server implementation using FastMCP.

Creates and configures the MCP server with:
    - Tool auto-discovery
    - Runtime context establishment
    - Configuration loading
"""

from __future__ import annotations

import contextvars
import inspect

from mcp.server.fastmcp import FastMCP

from jpscripts.core.config import AppConfig, load_config
from jpscripts.core.runtime import RuntimeContext, _runtime_ctx
from jpscripts.mcp import get_tool_metadata, logger, set_config
from jpscripts.mcp.tools import discover_tools

# Token for the long-running runtime context
_runtime_token: contextvars.Token[RuntimeContext | None] | None = None


def _establish_runtime_context(cfg: AppConfig) -> None:
    """Establish a long-running runtime context for the MCP server.

    Unlike typical context manager usage, MCP servers run indefinitely,
    so we establish the context at startup and don't reset it.
    """
    global _runtime_token
    from uuid import uuid4

    ctx = RuntimeContext(
        config=cfg,
        workspace_root=cfg.workspace_root.expanduser().resolve(),
        trace_id=f"mcp-{uuid4().hex[:8]}",
        dry_run=False,
    )
    _runtime_token = _runtime_ctx.set(ctx)
    logger.info("Runtime context established: trace_id=%s", ctx.trace_id)


def _load_configuration() -> AppConfig | None:
    """Load and store configuration for tools to consume.

    Returns:
        The loaded AppConfig, or None on failure.
    """
    try:
        cfg, _ = load_config()
    except Exception as exc:
        logger.error("Failed to load config during MCP startup", exc_info=exc)
        set_config(None)
        return None
    else:
        # Establish runtime context (preferred) and legacy config (fallback)
        _establish_runtime_context(cfg)
        set_config(cfg)
        logger.info("MCP Server loaded config from %s", cfg.notes_dir)
        return cfg


def register_tools(mcp: FastMCP) -> None:
    """Register all discovered tools with the MCP server.

    Uses the unified tool registry from discover_tools() to ensure
    AgentEngine and MCP server use identical tool sets.
    """
    tools = discover_tools()
    registered_count = 0

    for tool_name, func in tools.items():
        metadata = get_tool_metadata(func) or {}

        # Validate type hints
        signature = inspect.signature(func)
        for param_name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise RuntimeError(
                    f"MCP tool '{tool_name}' argument '{param_name}' is missing a type hint."
                )

        try:
            mcp.add_tool(func, **metadata)
            registered_count += 1
        except Exception as exc:
            logger.error("Failed to register tool %s", tool_name, exc_info=exc)

    logger.info("Registered %d MCP tools from unified registry", registered_count)


def create_server() -> FastMCP:
    _load_configuration()
    server = FastMCP("jpscripts")
    register_tools(server)
    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
