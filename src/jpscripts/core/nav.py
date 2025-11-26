from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

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


async def _iter_null_stream(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Yield null-terminated paths from a byte stream.

    Args:
        stream: Async stdout reader from a subprocess.

    Yields:
        Decoded path strings separated by null bytes.
    """
    buffer = b""
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        buffer += chunk
        parts = buffer.split(b"\0")
        buffer = parts.pop()
        for raw in parts:
            if raw:
                yield raw.decode("utf-8", errors="replace")
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


async def _scan_recent_with_rg(root: Path, max_depth: int, include_dirs: bool, ignore_dirs: set[str]) -> list[RecentEntry]:
    """Use ripgrep to enumerate recent files efficiently.

    Args:
        root: Directory to scan.
        max_depth: Maximum depth to traverse.
        include_dirs: Whether to include directory entries.
        ignore_dirs: Directory names to exclude.

    Returns:
        Sorted recent entries from ripgrep output.

    Raises:
        RuntimeError: If ripgrep fails when available.
    """
    rg = shutil.which("rg")
    if not rg:
        return await asyncio.to_thread(_scan_recent_sync, root, max_depth, include_dirs, ignore_dirs)

    cmd = [
        rg,
        "--files",
        "--null",
        "--sortr=modified",
    ]
    if (root / ".gitignore").exists():
        cmd.extend(["--ignore-file", ".gitignore"])
    for entry in ignore_dirs:
        cmd.extend(["--glob", f"!{entry}/*"])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    if proc.stdout is None:
        raise RuntimeError("ripgrep did not provide stdout for scanning.")

    entries: dict[Path, RecentEntry] = {}

    async for rel_path in _iter_null_stream(proc.stdout):
        candidate = (root / rel_path).resolve()
        try:
            relative = candidate.relative_to(root.resolve())
        except ValueError:
            continue

        depth = max(len(relative.parts) - 1, 0)
        if depth > max_depth:
            continue

        try:
            stat_result = await asyncio.to_thread(candidate.stat)
        except OSError:
            continue

        entries[candidate] = RecentEntry(path=candidate, mtime=stat_result.st_mtime, is_dir=False)

        if include_dirs:
            parent = candidate.parent
            if parent != root and parent not in entries and (len(parent.relative_to(root).parts) - 1) <= max_depth:
                try:
                    parent_stat = await asyncio.to_thread(parent.stat)
                    entries[parent] = RecentEntry(path=parent, mtime=parent_stat.st_mtime, is_dir=True)
                except OSError:
                    continue

    stderr = b""
    if proc.stderr is not None:
        stderr = await proc.stderr.read()
    await proc.wait()
    if proc.returncode not in (0, None):
        error_text = stderr.decode("utf-8", errors="replace").strip() or "ripgrep scan failed"
        raise RuntimeError(error_text)

    return sorted(entries.values(), key=lambda e: e.mtime, reverse=True)


async def scan_recent(root: Path, max_depth: int, include_dirs: bool, ignore_dirs: set[str]) -> list[RecentEntry]:
    """
    Async wrapper for filesystem scanning using ripgrep when available.
    """
    return await _scan_recent_with_rg(root, max_depth, include_dirs, ignore_dirs)

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
