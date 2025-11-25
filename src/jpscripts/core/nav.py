from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RecentEntry:
    path: Path
    mtime: float
    is_dir: bool

def _scan_recent_sync(root: Path, max_depth: int, include_dirs: bool, ignore_dirs: set[str]) -> list[RecentEntry]:
    """Synchronous implementation of the scan."""
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

async def scan_recent(root: Path, max_depth: int, include_dirs: bool, ignore_dirs: set[str]) -> list[RecentEntry]:
    """
    Async wrapper for filesystem scanning.
    Offloads the blocking IO to a thread so the UI loop stays responsive.
    """
    return await asyncio.to_thread(_scan_recent_sync, root, max_depth, include_dirs, ignore_dirs)

async def get_zoxide_projects() -> list[str]:
    """Async query to zoxide for frequent directories."""
    zoxide = shutil.which("zoxide")
    if not zoxide:
        raise RuntimeError("zoxide binary not found")

    proc = await asyncio.create_subprocess_exec(
        zoxide, "query", "-l",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"zoxide query failed: {stderr.decode().strip()}")

    return [line.strip() for line in stdout.decode().splitlines() if line.strip()]
