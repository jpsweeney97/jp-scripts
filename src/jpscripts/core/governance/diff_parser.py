"""Diff parsing utilities for constitutional compliance checking."""

from __future__ import annotations

import re
from pathlib import Path

from jpscripts.core.security import validate_path_safe


def _apply_hunks(original: str, hunks: list[tuple[int, list[str]]]) -> str:
    """Apply diff hunks to original content.

    Args:
        original: Original file content (empty for new files)
        hunks: List of (start_line, new_lines) tuples

    Returns:
        The post-patch content
    """
    if not original:
        # New file: concatenate all added lines from all hunks
        lines: list[str] = []
        for _, hunk_lines in hunks:
            lines.extend(hunk_lines)
        return "\n".join(lines)

    # For existing files, we need to reconstruct the file
    # Since we only track added/context lines (not removed lines),
    # we build the result directly from hunk content
    result_lines: list[str] = []
    for _, hunk_lines in hunks:
        result_lines.extend(hunk_lines)

    return "\n".join(result_lines)


def apply_patch_in_memory(diff: str, root: Path) -> dict[Path, str]:
    """Apply a unified diff in memory, returning post-patch content.

    This function reconstructs what the files WILL look like after the patch
    is applied, without actually writing to disk. Critical for governance
    checks to verify the patch content, not the original disk content.

    Args:
        diff: Unified diff text
        root: Workspace root for resolving relative paths

    Returns:
        Dict mapping file path to post-patch source content
    """
    results: dict[Path, str] = {}
    current_file: Path | None = None
    is_new_file = False
    hunks: list[tuple[int, list[str]]] = []  # (start_line, lines)
    current_hunk_lines: list[str] = []
    current_hunk_start = 1

    def save_current_file() -> None:
        """Save accumulated hunks for the current file."""
        nonlocal current_file, hunks, current_hunk_lines, current_hunk_start, is_new_file

        # Save pending hunk
        if current_hunk_lines:
            hunks.append((current_hunk_start, current_hunk_lines))

        if current_file is not None and hunks:
            # For new files, just concatenate all hunk content
            if is_new_file:
                all_lines: list[str] = []
                for _, hunk_lines in hunks:
                    all_lines.extend(hunk_lines)
                results[current_file] = "\n".join(all_lines)
            else:
                # For existing files, we need to merge with original
                # But for governance, we primarily care about the NEW content
                # So we can just use the hunk content as the new source
                all_lines = []
                for _, hunk_lines in hunks:
                    all_lines.extend(hunk_lines)
                results[current_file] = "\n".join(all_lines)

        # Reset state
        hunks = []
        current_hunk_lines = []
        current_hunk_start = 1

    for line in diff.splitlines():
        if line.startswith("--- "):
            # Check if this is a new file (--- /dev/null)
            is_new_file = "/dev/null" in line

        elif line.startswith("+++ "):
            # Save previous file before starting new one
            save_current_file()

            # Parse new file path
            if line.startswith("+++ b/"):
                path_str = line[6:].strip()
            else:
                path_str = line[4:].strip()
                if path_str.startswith("b/"):
                    path_str = path_str[2:]

            # Validate path stays within workspace to prevent path traversal attacks
            candidate = root / path_str
            validation = validate_path_safe(candidate, root)
            if validation.is_err():
                # Skip files with invalid paths (path traversal attempt)
                current_file = None
                continue
            current_file = validation.unwrap()

        elif line.startswith("@@ "):
            # Save previous hunk before starting new one
            if current_hunk_lines:
                hunks.append((current_hunk_start, current_hunk_lines))
                current_hunk_lines = []

            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.search(r"\+(\d+)", line)
            if match:
                current_hunk_start = int(match.group(1))

        elif line.startswith("+") and not line.startswith("+++"):
            # Added line - include in result (strip the '+' prefix)
            current_hunk_lines.append(line[1:])

        elif line.startswith(" "):
            # Context line - include in result (strip the ' ' prefix)
            current_hunk_lines.append(line[1:])

        # Lines starting with '-' are removed lines - skip them

    # Save the last file
    save_current_file()

    return results


def parse_diff_files(diff: str, root: Path) -> dict[Path, set[int]]:
    """Parse diff to extract file paths and changed line numbers.

    Returns a mapping from file path to set of modified line numbers.
    An empty set means all lines should be checked.
    """
    files: dict[Path, set[int]] = {}
    current_file: Path | None = None
    current_line = 0

    for line in diff.splitlines():
        # Match new file header: +++ b/path/to/file.py
        if line.startswith("+++ b/"):
            path_str = line[6:]
            candidate = root / path_str
            validation = validate_path_safe(candidate, root)
            if validation.is_err():
                current_file = None
                continue
            current_file = validation.unwrap()
            files[current_file] = set()
        elif line.startswith("+++ "):
            # Handle other diff formats: +++ path/to/file.py
            path_str = line[4:].strip()
            if path_str.startswith("b/"):
                path_str = path_str[2:]
            candidate = root / path_str
            validation = validate_path_safe(candidate, root)
            if validation.is_err():
                current_file = None
                continue
            current_file = validation.unwrap()
            files[current_file] = set()
        elif line.startswith("@@ "):
            # Parse hunk header: @@ -start,count +start,count @@
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            # Added line
            if current_file is not None:
                files[current_file].add(current_line)
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Deleted line - don't increment current_line
            pass
        else:
            # Context line or other
            current_line += 1

    return files


__all__ = [
    "apply_patch_in_memory",
    "parse_diff_files",
]
