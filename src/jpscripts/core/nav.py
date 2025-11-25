from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RecentEntry:
    path: Path
    mtime: float
    is_dir: bool

def scan_recent(root: Path, max_depth: int, include_dirs: bool, ignore_dirs: set[str]) -> list[RecentEntry]:
    """
    Scan the root directory for recently modified files/directories.
    Returns a list of RecentEntry objects sorted by mtime (descending).
    """
    entries: list[RecentEntry] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    ignored = set(ignore_dirs)

    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.name in ignored:
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                    except OSError:
                        continue

                    if include_dirs or not is_dir:
                        entries.append(RecentEntry(path=Path(entry.path), mtime=mtime, is_dir=is_dir))

                    if is_dir and depth < max_depth:
                        stack.append((Path(entry.path), depth + 1))
        except OSError:
            continue

    return sorted(entries, key=lambda e: e.mtime, reverse=True)

def get_zoxide_projects() -> list[str]:
    """Query zoxide for a list of frequent directories."""
    zoxide = shutil.which("zoxide")
    if not zoxide:
        # In core we raise errors rather than printing
        raise RuntimeError("zoxide binary not found")

    try:
        proc = subprocess.run(
            [zoxide, "query", "-l"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"zoxide query failed: {exc.stderr or exc}")

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
