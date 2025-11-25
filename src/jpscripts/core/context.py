from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

HARD_FILE_CONTEXT_LIMIT = 100_000

async def run_and_capture(command: str, cwd: Path) -> str:
    """Run a shell command and return combined stdout/stderr."""
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (stdout + stderr).decode("utf-8", errors="replace")

def resolve_files_from_output(output: str, root: Path) -> set[Path]:
    """Parse command output for file paths that exist in the workspace."""
    found = set()

    for match in FILE_PATTERN.finditer(output):
        raw_path = match.group("path")
        # Clean up potential noise
        clean_path = raw_path.strip(".'\"()")

        candidate = (root / clean_path).resolve()

        # Security/Sanity check: File must exist and be inside root
        try:
            if candidate.is_file() and root in candidate.parents:
                found.add(candidate)
        except OSError:
            continue

    return found

async def gather_context(command: str, root: Path) -> tuple[str, set[Path]]:
    """
    Run a command, capture output, and find relevant files.
    Returns (output_log, set_of_paths).
    """
    output = await run_and_capture(command, root)
    files = resolve_files_from_output(output, root)
    return output, files


def read_file_context(path: Path, max_chars: int) -> str | None:
    """
    Read file content safely and truncate to max_chars.
    Returns None on any read/encoding error.
    """
    limit = max(0, min(max_chars, HARD_FILE_CONTEXT_LIMIT))
    try:
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read(limit)
    except (OSError, UnicodeDecodeError):
        return None
    return text
