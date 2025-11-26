from __future__ import annotations

import asyncio

from jpscripts.core.notes_impl import append_to_daily_note
from jpscripts.mcp import get_config, tool


@tool()
async def append_daily_note(message: str) -> str:
    """Append a log entry to the user's daily note system."""
    try:
        cfg = get_config()
        if cfg is None:
            return "Config not loaded."
        target_dir = cfg.notes_dir.expanduser()
        path = await asyncio.to_thread(append_to_daily_note, target_dir, message)
        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        return f"Error appending note: {str(e)}"
