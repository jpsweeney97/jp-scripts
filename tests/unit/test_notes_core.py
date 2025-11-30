from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from jpscripts.core import notes_impl


def test_ensure_notes_dir_creates_directory(tmp_path: Path) -> None:
    """Verify it creates the directory if it doesn't exist."""
    target = tmp_path / "notes"
    assert not target.exists()

    notes_impl.ensure_notes_dir(target)

    assert target.exists()
    assert target.is_dir()


def test_get_today_path_format() -> None:
    """Verify the filename format matches YYYY-MM-DD.md."""
    fake_root = Path("/tmp/notes")

    # FIX: Patch the 'dt' module reference INSIDE notes_impl, not datetime.date globally
    with patch("jpscripts.core.notes_impl.dt") as mock_dt:
        mock_dt.date.today.return_value = datetime.date(2025, 11, 24)

        result = notes_impl.get_today_path(fake_root)

    assert result == Path("/tmp/notes/2025-11-24.md")


@pytest.mark.asyncio
async def test_append_to_daily_note_creates_and_appends(tmp_path: Path) -> None:
    """Verify it creates a file and appends content with a timestamp."""
    notes_dir = tmp_path / "my-notes"

    # 1. First write (creates file)
    path = await notes_impl.append_to_daily_note(notes_dir, "First entry")

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "First entry" in content
    assert "- [" in content  # Check for timestamp bracket

    # 2. Second write (appends)
    await notes_impl.append_to_daily_note(notes_dir, "Second entry")

    content_updated = path.read_text(encoding="utf-8")
    lines = content_updated.strip().splitlines()

    assert len(lines) == 2
    assert "First entry" in lines[0]
    assert "Second entry" in lines[1]
