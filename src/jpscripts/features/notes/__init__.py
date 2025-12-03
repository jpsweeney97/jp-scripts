"""Notes feature.

This module provides note-taking and daily note functionality.
"""

from jpscripts.features.notes.service import (
    append_to_daily_note,
    ensure_notes_dir,
    get_today_path,
)

__all__ = [
    "append_to_daily_note",
    "ensure_notes_dir",
    "get_today_path",
]
