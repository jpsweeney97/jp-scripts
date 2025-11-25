from __future__ import annotations

from pathlib import Path


def validate_path(path: str | Path, root: Path) -> Path:
    """
    Resolve a user-supplied path and ensure it stays within the allowed root.
    Raises PermissionError if the path escapes the root (including via symlinks).
    """
    root_resolved = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()

    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise PermissionError(f"Access to {candidate} is outside allowed root {root_resolved}")

    return candidate
