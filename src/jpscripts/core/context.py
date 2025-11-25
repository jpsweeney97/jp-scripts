from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from jpscripts.core.console import console

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+\.[a-z0-9]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

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
    console.print(f"[dim]Running diagnostic: {command}[/dim]")
    output = await run_and_capture(command, root)
    files = resolve_files_from_output(output, root)
    return output, files
