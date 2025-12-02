"""MCP search tools for code search.

Provides tools for searching the codebase:
    - search_codebase: Regex search via ripgrep
    - Path validation for security
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path

from jpscripts.core.result import Err
from jpscripts.core.runtime import get_runtime
from jpscripts.core.security import validate_path_safe_async
from jpscripts.mcp import tool, tool_error_handler
from jpscripts.net import search as search_core


@tool()
@tool_error_handler
async def search_codebase(pattern: str, path: str = ".") -> str:
    """
    Search the codebase using ripgrep (grep).
    Returns the raw text matches with line numbers.
    """
    ctx = get_runtime()
    root = ctx.workspace_root

    max_chars = getattr(ctx.config, "max_file_context_chars", 50000)
    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    path_result = await validate_path_safe_async(candidate, root)
    if isinstance(path_result, Err):
        return f"Error: {path_result.error.message}"
    search_root = path_result.value
    safe_pattern = re.escape(pattern)

    result = await asyncio.to_thread(
        search_core.run_ripgrep,
        safe_pattern,
        search_root,
        line_number=True,
        context=1,
        max_chars=max_chars,
    )
    if not result:
        return "No matches found."

    return result


@tool()
@tool_error_handler
async def find_todos(path: str = ".") -> str:
    """
    Scan for TODO/FIXME/HACK comments in the codebase.
    Returns a JSON list of objects: {type, file, line, text}.
    """
    ctx = get_runtime()
    root = ctx.workspace_root

    scan_root = Path.cwd() if path == "." else Path(path).expanduser()
    path_result = await validate_path_safe_async(scan_root, root)
    if isinstance(path_result, Err):
        return f"Error: {path_result.error.message}"
    target = path_result.value

    entries = await search_core.scan_todos(target)

    if not entries:
        return "[]"

    return json.dumps([dataclasses.asdict(e) for e in entries], indent=2)
