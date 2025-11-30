from __future__ import annotations

from jpscripts.core.notes_impl import append_to_daily_note
from jpscripts.core.runtime import get_runtime
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def append_daily_note(message: str) -> str:
    """Append a log entry to the user's daily note system."""
    try:
        ctx = get_runtime()
        target_dir = ctx.config.notes_dir.expanduser()
        path = await append_to_daily_note(target_dir, message)
        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        return f"Error appending note: {e!s}"
