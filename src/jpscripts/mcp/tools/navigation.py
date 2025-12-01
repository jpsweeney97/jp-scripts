"""MCP navigation tools for workspace exploration.

Provides tools for navigating the workspace:
    - list_recent_files: Recently modified files
    - Surface related memories for context
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jpscripts.core import memory as memory_core
from jpscripts.core import nav as nav_core
from jpscripts.core.result import Err, Ok
from jpscripts.core.runtime import get_runtime
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def list_recent_files(limit: int = 20) -> str:
    """List files modified recently in the current workspace root and surface related memories."""
    ctx = get_runtime()
    root = ctx.workspace_root

    match await nav_core.scan_recent(
        root,
        max_depth=3,
        include_dirs=False,
        ignore_dirs=set(ctx.config.ignore_dirs),
    ):
        case Err(err):
            return f"Error scanning recent files: {err.message}"
        case Ok(entries):
            lines = [
                f"{e.path.relative_to(root) if e.path.is_relative_to(root) else e.path}"
                for e in entries[:limit]
            ]

    query_hint = " ".join(Path(line).stem for line in lines[:5]) or Path.cwd().name

    memories = await asyncio.to_thread(
        memory_core.query_memory,
        query_hint,
        limit=3,
        config=ctx.config,
    )

    mem_block = "\n".join(memories) if memories else "No related memories."
    recent_block = "\n".join(lines) if lines else "No recent files found."

    return f"Recent files:\n{recent_block}\n\nRelevant memories:\n{mem_block}"


@tool()
@tool_error_handler
async def list_projects() -> str:
    """List known projects (via zoxide)."""
    match await nav_core.get_zoxide_projects():
        case Err(err):
            return f"Error listing projects: {err.message}"
        case Ok(paths):
            return "\n".join(paths) if paths else "No projects found."
