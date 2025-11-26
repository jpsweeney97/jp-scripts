from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jpscripts.core.context import read_file_context
from jpscripts.core.security import is_git_workspace, validate_path
from jpscripts.mcp import get_config, logger, tool, tool_error_handler


class ToolExecutionError(RuntimeError):
    """Raised when a patch operation cannot be completed."""


@dataclass
class PatchHunk:
    old_start: int
    old_length: int
    new_start: int
    new_length: int
    lines: list[str]


@dataclass
class ParsedPatch:
    hunks: list[PatchHunk]
    delete_file: bool
    new_file: bool


_HUNK_HEADER_RE = re.compile(
    r"@@ -(?P<old_start>\d+)(?:,(?P<old_length>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_length>\d+))? @@"
)
_DEV_NULL_TARGETS = {"/dev/null", "dev/null"}


@tool()
@tool_error_handler
async def read_file(path: str) -> str:
    """
    Read the content of a file (truncated to JP_MAX_FILE_CONTEXT_CHARS).
    Use this to inspect code, config files, or logs.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    root = cfg.workspace_root.expanduser()
    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    target = validate_path(candidate, root)
    if not target.exists():
        return f"Error: File {target} does not exist."
    if not target.is_file():
        return f"Error: {target} is not a file."

    max_chars = getattr(cfg, "max_file_context_chars", 50000)
    total_size = target.stat().st_size
    content = await asyncio.to_thread(read_file_context, target, max_chars)
    if content is None:
        return f"Error: Could not read file {target} (unsupported encoding or IO error)."
    if total_size > max_chars:
        content += (
            f"\n\n[SYSTEM WARNING: File truncated at {max_chars} chars. "
            f"Total size: {total_size}. Use read_file_paged(path, offset={max_chars}) to read more.]"
        )
    return content


@tool()
@tool_error_handler
async def read_file_paged(path: str, offset: int = 0, limit: int = 20000) -> str:
    """
    Read a file segment starting at byte offset. Use this to read large files.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    root = cfg.workspace_root.expanduser()
    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    target = validate_path(candidate, root)

    if not target.exists():
        return f"Error: File {target} does not exist."
    if not target.is_file():
        return f"Error: {target} is not a file."
    if offset < 0:
        return "Error: offset must be non-negative."
    if limit <= 0:
        return "Error: limit must be positive."

    def _read_slice() -> str:
        with target.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(limit)
        return data.decode("utf-8", errors="replace")

    return await asyncio.to_thread(_read_slice)


@tool()
@tool_error_handler
async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Create or overwrite a file with the given content.
    Enforces workspace sandbox. Requires overwrite=True to replace existing files.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    if cfg.dry_run:
        target = Path(path).expanduser()
        return f"Simulated write to {target} (dry-run active). Content length: {len(content)}"

    root = cfg.workspace_root.expanduser()
    target = validate_path(Path(path).expanduser(), root)

    if target.exists() and not overwrite:
        return f"Error: File {target.name} already exists. Pass overwrite=True to replace it."

    target.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> int:
        target.write_text(content, encoding="utf-8")
        return len(content.encode("utf-8"))

    size = await asyncio.to_thread(_write)
    logger.info("Wrote %d bytes to %s", size, target)
    return f"Successfully wrote {target.name} ({size} bytes)."


@tool()
@tool_error_handler
async def list_directory(path: str) -> str:
    """
    List contents of a directory (like ls).
    Returns a list of 'd: dir_name' and 'f: file_name'.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    root = cfg.workspace_root.expanduser()
    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    target = validate_path(candidate, root)
    if not target.exists():
        return f"Error: Path {target} does not exist."
    if not target.is_dir():
        return f"Error: {target} is not a directory."

    def _ls() -> str:
        entries: list[str] = []
        with os.scandir(target) as it:
            for entry in it:
                prefix = "d" if entry.is_dir() else "f"
                entries.append(f"{prefix}: {entry.name}")
        return "\n".join(sorted(entries))

    return await asyncio.to_thread(_ls)


def _normalize_patch_path(raw_path: str) -> str:
    path_str = raw_path.strip()
    if path_str.startswith(("a/", "b/")):
        return path_str[2:]
    return path_str


def _parse_patch(diff_text: str, target: Path, root: Path) -> ParsedPatch:
    if not diff_text.strip():
        raise ToolExecutionError("Patch text is empty.")

    lines = diff_text.splitlines(keepends=True)
    old_path: str | None = None
    new_path: str | None = None
    hunks: list[PatchHunk] = []
    current_header: tuple[int, int, int, int] | None = None
    current_lines: list[str] = []
    target_rel = validate_path(target, root).relative_to(root).as_posix()
    file_section_seen = False

    for line in lines:
        if line.startswith("--- "):
            if file_section_seen:
                raise ToolExecutionError("Patches for multiple files are not supported.")
            old_path = _normalize_patch_path(line[4:].strip())
            file_section_seen = True
            continue

        if line.startswith("+++ "):
            if new_path is not None:
                raise ToolExecutionError("Patches for multiple files are not supported.")
            new_path = _normalize_patch_path(line[4:].strip())
            continue

        if line.startswith("@@"):
            if current_header is not None:
                hunks.append(PatchHunk(*current_header, lines=current_lines))
                current_lines = []

            match = _HUNK_HEADER_RE.match(line.strip())
            if not match:
                raise ToolExecutionError(f"Invalid hunk header: {line.strip()}")

            current_header = (
                int(match.group("old_start")),
                int(match.group("old_length") or "1"),
                int(match.group("new_start")),
                int(match.group("new_length") or "1"),
            )
            continue

        if current_header is None:
            continue

        if line.startswith("\\ No newline at end of file"):
            continue

        if not line or line[0] not in (" ", "+", "-"):
            raise ToolExecutionError(f"Unexpected line in hunk: {line.strip()}")

        current_lines.append(line)

    if current_header is not None:
        hunks.append(PatchHunk(*current_header, lines=current_lines))

    if not hunks:
        raise ToolExecutionError("No hunks found in patch.")

    patch_target = new_path if new_path not in _DEV_NULL_TARGETS else old_path
    if patch_target is None:
        raise ToolExecutionError("Patch does not specify a target file.")

    normalized_patch_target = _normalize_patch_path(patch_target)
    if normalized_patch_target != target_rel:
        raise ToolExecutionError(f"Patch targets {normalized_patch_target} but requested {target_rel}.")

    delete_file = new_path in _DEV_NULL_TARGETS
    new_file = old_path in _DEV_NULL_TARGETS

    return ParsedPatch(hunks=hunks, delete_file=bool(delete_file), new_file=bool(new_file))


def _verify_hunk_lengths(hunk: PatchHunk) -> None:
    expected_old = sum(1 for line in hunk.lines if line.startswith((" ", "-")))
    expected_new = sum(1 for line in hunk.lines if line.startswith((" ", "+")))
    if expected_old != hunk.old_length:
        raise ToolExecutionError(
            f"Hunk length mismatch at line {hunk.old_start}: expected {hunk.old_length} context+removals."
        )
    if expected_new != hunk.new_length:
        raise ToolExecutionError(
            f"Hunk length mismatch at line {hunk.new_start}: expected {hunk.new_length} context+additions."
        )


def _apply_single_hunk(lines: list[str], hunk: PatchHunk, start_index: int) -> tuple[list[str], int]:
    if start_index > len(lines):
        raise ToolExecutionError(f"Hunk starting at line {hunk.old_start} cannot be applied (file too short).")

    output: list[str] = list(lines[:start_index])
    cursor = start_index

    for line in hunk.lines:
        prefix = line[0]
        content = line[1:]

        if prefix == " ":
            if cursor >= len(lines) or lines[cursor] != content:
                failing_line = hunk.old_start + max(0, cursor - start_index)
                raise ToolExecutionError(f"Hunk failed at line {failing_line}: context mismatch.")
            output.append(lines[cursor])
            cursor += 1
        elif prefix == "-":
            if cursor >= len(lines) or lines[cursor] != content:
                failing_line = hunk.old_start + max(0, cursor - start_index)
                raise ToolExecutionError(f"Hunk failed at line {failing_line}: expected removal to match.")
            cursor += 1
        elif prefix == "+":
            output.append(content)
        else:
            raise ToolExecutionError(f"Invalid patch line prefix '{prefix}' near line {hunk.old_start}.")

    output.extend(lines[cursor:])
    delta = len(output) - len(lines)
    return output, delta


def _apply_hunks(lines: list[str], hunks: Sequence[PatchHunk]) -> list[str]:
    updated = lines
    offset = 0

    for hunk in hunks:
        _verify_hunk_lengths(hunk)
        start_index = max(0, hunk.old_start - 1 + offset)
        updated, delta = _apply_single_hunk(updated, hunk, start_index)
        offset += delta

    return updated


def _apply_parsed_patch(target: Path, parsed: ParsedPatch, write_changes: bool) -> None:
    if parsed.delete_file and not target.exists():
        raise ToolExecutionError(f"Cannot delete missing file {target}.")
    if not parsed.new_file and not target.exists():
        raise ToolExecutionError(f"Target file {target} does not exist for modification.")

    original_text = target.read_text(encoding="utf-8") if target.exists() else ""
    original_lines = original_text.splitlines(keepends=True)
    patched_lines = _apply_hunks(original_lines, parsed.hunks)

    if not write_changes:
        return

    target.parent.mkdir(parents=True, exist_ok=True)

    if parsed.delete_file:
        target.unlink(missing_ok=True)
        return

    target.write_text("".join(patched_lines), encoding="utf-8")


async def _apply_patch_with_git(diff_text: str, root: Path) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--whitespace=nowarn",
            cwd=str(root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False

    stdout, stderr = await proc.communicate(diff_text.encode("utf-8"))
    if proc.returncode == 0:
        return True

    logger.error("git apply failed: %s", stderr.decode(errors="replace") or stdout.decode(errors="replace"))
    return False


@tool()
async def apply_patch(path: str, diff: str) -> str:
    """Apply a unified diff to a file within the workspace.

    Args:
        path: Target file path, absolute or relative to the workspace root.
        diff: Unified diff content to apply.

    Returns:
        Status message describing whether the patch was applied.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    root = cfg.workspace_root.expanduser()
    target = validate_path(Path(path).expanduser(), root)
    if target.is_dir():
        raise ToolExecutionError(f"Cannot apply a patch to directory {target}.")

    try:
        parsed = _parse_patch(diff, target, root)
    except ToolExecutionError:
        raise

    write_changes = not cfg.dry_run
    try:
        await asyncio.to_thread(_apply_parsed_patch, target, parsed, write_changes)
        if cfg.dry_run:
            return f"Dry-run: patch validated for {target} (no changes written)."
        return "Patch applied successfully."
    except ToolExecutionError as exc:
        error_reason = str(exc)
    except Exception as exc:
        error_reason = f"Failed to apply patch: {exc}"
    else:
        error_reason = ""

    if not is_git_workspace(root):
        raise ToolExecutionError(error_reason or "Patch apply failed and git apply unavailable.")

    git_applied = await _apply_patch_with_git(diff, root)
    if git_applied:
        return "Patch applied successfully."

    raise ToolExecutionError(error_reason or "Patch apply failed via pure Python and git apply.")
