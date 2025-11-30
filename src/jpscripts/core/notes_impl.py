from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path


def ensure_notes_dir(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)


def get_today_path(notes_dir: Path) -> Path:
    today = dt.date.today().isoformat()
    return notes_dir / f"{today}.md"


async def append_to_daily_note(notes_dir: Path, message: str) -> Path:
    """
    Appends a timestamped message to today's note file.
    Returns the path to the note file that was modified.
    """
    ensure_notes_dir(notes_dir)
    note_path = get_today_path(notes_dir)

    timestamp = dt.datetime.now().strftime("%H:%M")

    def _write() -> None:
        with note_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] {message}\n")

    await asyncio.to_thread(_write)

    return note_path
