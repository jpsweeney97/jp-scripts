from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path

from jpscripts.core import search as search_core
from jpscripts.core.security import validate_path
from jpscripts.mcp import get_config, tool


@tool()
async def search_codebase(pattern: str, path: str = ".") -> str:
    """
    Search the codebase using ripgrep (grep).
    Returns the raw text matches with line numbers.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        max_chars = getattr(cfg, "max_file_context_chars", 50000)
        root = cfg.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        search_root = validate_path(candidate, root)
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
    except Exception as e:
        return f"Error searching codebase: {str(e)}"


@tool()
async def find_todos(path: str = ".") -> str:
    """
    Scan for TODO/FIXME/HACK comments in the codebase.
    Returns a JSON list of objects: {type, file, line, text}.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        scan_root = Path.cwd() if path == "." else Path(path).expanduser()
        sandbox_root = cfg.workspace_root.expanduser()

        try:
            target = validate_path(scan_root, sandbox_root)
        except PermissionError:
            return f"Error: Access denied. {path} is outside workspace {sandbox_root}."

        entries = await search_core.scan_todos(target)

        if not entries:
            return "[]"

        return json.dumps([dataclasses.asdict(e) for e in entries], indent=2)

    except Exception as e:
        return f"Error scanning todos: {str(e)}"
