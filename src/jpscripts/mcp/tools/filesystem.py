from __future__ import annotations

import asyncio
import os
from pathlib import Path

from jpscripts.core.context import read_file_context
from jpscripts.core.security import validate_path
from jpscripts.mcp import get_config, logger, tool


@tool()
async def read_file(path: str) -> str:
    """
    Read the content of a file (truncated to JP_MAX_FILE_CONTEXT_CHARS).
    Use this to inspect code, config files, or logs.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        root = cfg.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        target = validate_path(candidate, root)
        if not target.exists():
            return f"Error: File {target} does not exist."
        if not target.is_file():
            return f"Error: {target} is not a file."

        max_chars = getattr(cfg, "max_file_context_chars", 50000)
        total_size = target.stat().st_size
        content = await asyncio.to_thread(read_file_context, target, max_chars)
        if content is None:
            return f"Error: Could not read file {target} (unsupported encoding or IO error)."
        if total_size > max_chars:
            content += (
                f"\n\n[SYSTEM WARNING: File truncated at {max_chars} chars. "
                f"Total size: {total_size}. Use read_file_paged(path, offset={max_chars}) to read more.]"
            )
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool()
async def read_file_paged(path: str, offset: int = 0, limit: int = 20000) -> str:
    """
    Read a file segment starting at byte offset. Use this to read large files.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        root = cfg.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        target = validate_path(candidate, root)

        if not target.exists():
            return f"Error: File {target} does not exist."
        if not target.is_file():
            return f"Error: {target} is not a file."
        if offset < 0:
            return "Error: offset must be non-negative."
        if limit <= 0:
            return "Error: limit must be positive."

        def _read_slice() -> str:
            with target.open("rb") as fh:
                fh.seek(offset)
                data = fh.read(limit)
            return data.decode("utf-8", errors="replace")

        return await asyncio.to_thread(_read_slice)
    except Exception as e:
        return f"Error reading file segment: {str(e)}"


@tool()
async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Create or overwrite a file with the given content.
    Enforces workspace sandbox. Requires overwrite=True to replace existing files.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        root = cfg.workspace_root.expanduser()
        target = validate_path(Path(path).expanduser(), root)

        if target.exists() and not overwrite:
            return f"Error: File {target.name} already exists. Pass overwrite=True to replace it."

        target.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> int:
            target.write_text(content, encoding="utf-8")
            return len(content.encode("utf-8"))

        size = await asyncio.to_thread(_write)
        logger.info("Wrote %d bytes to %s", size, target)
        return f"Successfully wrote {target.name} ({size} bytes)."

    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool()
async def list_directory(path: str) -> str:
    """
    List contents of a directory (like ls).
    Returns a list of 'd: dir_name' and 'f: file_name'.
    """
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."

        root = cfg.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        target = validate_path(candidate, root)
        if not target.exists():
            return f"Error: Path {target} does not exist."
        if not target.is_dir():
            return f"Error: {target} is not a directory."

        def _ls() -> str:
            entries: list[str] = []
            with os.scandir(target) as it:
                for entry in it:
                    prefix = "d" if entry.is_dir() else "f"
                    entries.append(f"{prefix}: {entry.name}")
            return "\n".join(sorted(entries))

        return await asyncio.to_thread(_ls)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
