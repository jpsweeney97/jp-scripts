from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from jpscripts.core.context import smart_read_context
from jpscripts.core.rate_limit import RateLimiter
from jpscripts.core.result import Err
from jpscripts.core.runtime import get_runtime
from jpscripts.core.security import (
    is_git_workspace,
    validate_and_open,
    validate_path,
    validate_path_safe_async,
)
from jpscripts.mcp import logger, tool, tool_error_handler

# Rate limiter for MCP file operations to prevent DoS abuse.
# Allows 100 operations per minute (generous for normal use, limiting for abuse).
_file_rate_limiter = RateLimiter(max_calls=100, window_seconds=60.0)


class ToolExecutionError(RuntimeError):
    """Raised when a patch operation cannot be completed."""


async def _check_rate_limit() -> str | None:
    """Check rate limit and return error message if exceeded, None otherwise."""
    if not await _file_rate_limiter.acquire():
        wait_time = _file_rate_limiter.time_until_available()
        return f"Error: Rate limit exceeded. Too many file operations. Try again in {wait_time:.1f} seconds."
    return None


@tool()
@tool_error_handler
async def read_file(path: str) -> str:
    """
    Read the content of a file (truncated to JP_MAX_FILE_CONTEXT_CHARS).
    Use this to inspect code, config files, or logs.
    """
    if error := await _check_rate_limit():
        return error

    ctx = get_runtime()
    root = ctx.workspace_root

    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    result = await validate_path_safe_async(candidate, root)
    if isinstance(result, Err):
        return f"Error: {result.error.message}"
    target = result.value
    if not target.exists():
        return f"Error: File {target} does not exist."
    if not target.is_file():
        return f"Error: {target} is not a file."

    max_chars = getattr(ctx.config, "max_file_context_chars", 50000)
    total_size = target.stat().st_size
    content = await asyncio.to_thread(smart_read_context, target, max_chars)
    if content == "" and total_size > 0 and max_chars > 0:
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
    if error := await _check_rate_limit():
        return error

    ctx = get_runtime()
    root = ctx.workspace_root

    if offset < 0:
        return "Error: offset must be non-negative."
    if limit <= 0:
        return "Error: limit must be positive."

    base = Path(path)
    candidate = base if base.is_absolute() else root / base

    def _open_and_read() -> str:
        result = validate_and_open(candidate, root, "rb")
        if isinstance(result, Err):
            raise RuntimeError(result.error.message)
        with result.value as fh:
            fh.seek(offset)
            data: bytes = fh.read(limit)
        return data.decode("utf-8", errors="replace")

    try:
        return await asyncio.to_thread(_open_and_read)
    except RuntimeError as exc:
        return f"Error: {exc}"


@tool()
@tool_error_handler
async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """
    Create or overwrite a file with the given content.
    Enforces workspace sandbox. Requires overwrite=True to replace existing files.
    """
    if error := await _check_rate_limit():
        return error

    ctx = get_runtime()

    if ctx.dry_run:
        target = Path(path).expanduser()
        return f"Simulated write to {target} (dry-run active). Content length: {len(content)}"

    root = ctx.workspace_root
    target = Path(path).expanduser()
    candidate = target if target.is_absolute() else root / target

    # Check existence before atomic open (for overwrite protection)
    if candidate.exists() and not overwrite:
        return f"Error: File {candidate.name} already exists. Pass overwrite=True to replace it."

    def _open_and_write() -> int:
        # Create parent directory first
        candidate.parent.mkdir(parents=True, exist_ok=True)

        result = validate_and_open(candidate, root, "w", encoding="utf-8")
        if isinstance(result, Err):
            raise RuntimeError(result.error.message)
        with result.value as fh:
            fh.write(content)
        return len(content.encode("utf-8"))

    try:
        size = await asyncio.to_thread(_open_and_write)
    except RuntimeError as exc:
        return f"Error: {exc}"

    logger.info("Wrote %d bytes to %s", size, candidate)
    return f"Successfully wrote {candidate.name} ({size} bytes)."


@tool()
@tool_error_handler
async def list_directory(path: str) -> str:
    """
    List contents of a directory (like ls).
    Returns a list of 'd: dir_name' and 'f: file_name'.
    """
    if error := await _check_rate_limit():
        return error

    ctx = get_runtime()
    root = ctx.workspace_root

    base = Path(path)
    candidate = base if base.is_absolute() else root / base
    result = await validate_path_safe_async(candidate, root)
    if isinstance(result, Err):
        return f"Error: {result.error.message}"
    target = result.value
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


def _extract_patch_targets(diff_text: str) -> set[str]:
    targets: set[str] = set()
    for line in diff_text.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        candidate = line[4:].split("\t", maxsplit=1)[0].strip()
        if candidate in {"/dev/null", "dev/null"}:
            continue
        normalized = _normalize_patch_path(candidate)
        if normalized:
            targets.add(normalized)
    return targets


def _validate_patch_targets(diff_text: str, target: Path, root: Path) -> None:
    targets = _extract_patch_targets(diff_text)
    if not targets:
        raise ToolExecutionError("Patch missing file headers; include ---/+++ lines.")
    if len(targets) > 1:
        raise ToolExecutionError("Patches for multiple files are not supported.")

    target_rel = validate_path(target, root).relative_to(root).as_posix()
    patch_target = next(iter(targets))
    normalized_target = _normalize_patch_path(patch_target)
    resolved = validate_path(root / normalized_target, root)
    resolved_rel = resolved.relative_to(root).as_posix()
    if resolved_rel != target_rel:
        raise ToolExecutionError(f"Patch targets {resolved_rel} but requested {target_rel}.")


def _detect_strip_level(diff_text: str) -> int:
    for line in diff_text.splitlines():
        if line.startswith(("--- ", "+++ ")):
            path = line[4:].strip()
            if path.startswith(("a/", "b/")):
                return 1
            break
    return 0


def _extract_conflict_lines(stderr_text: str, stdout_text: str = "") -> str:
    combined = [
        *(stderr_text.splitlines() if stderr_text else []),
        *(stdout_text.splitlines() if stdout_text else []),
    ]
    keywords = ("hunk", "failed", "reject", "error", "conflict", "No such file", "file not found")
    interesting: list[str] = []
    for line in combined:
        lower_line = line.lower()
        if any(key.lower() in lower_line for key in keywords):
            interesting.append(line.strip())
    if interesting:
        return "\n".join(interesting)
    trimmed = stderr_text.strip() if stderr_text else stdout_text.strip()
    return trimmed


def _format_patch_error(stderr: bytes, stdout: bytes) -> str:
    stderr_text = stderr.decode(errors="replace")
    stdout_text = stdout.decode(errors="replace")
    details = _extract_conflict_lines(stderr_text, stdout_text)
    return details or "Patch failed with unknown error."


async def _apply_patch_with_git(diff_text: str, root: Path, *, check_only: bool) -> None:
    args = ["git", "apply", "--verbose", "--reject"]
    if check_only:
        args.append("--check")
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ToolExecutionError("git executable not found on PATH.")

    stdout, stderr = await proc.communicate(diff_text.encode("utf-8"))
    if proc.returncode != 0:
        failure = _format_patch_error(stderr, stdout)
        logger.error("git apply failed: %s", failure)
        raise ToolExecutionError(f"git apply failed: {failure}")


async def _apply_patch_with_system_patch(diff_text: str, root: Path, *, check_only: bool) -> None:
    patch_binary = shutil.which("patch")
    if patch_binary is None:
        raise ToolExecutionError(
            "`patch` binary not found on PATH. Install patch to apply diffs outside git workspaces."
        )

    strip_level = _detect_strip_level(diff_text)
    args = [
        patch_binary,
        f"-p{strip_level}",
        "--verbose",
        "--directory",
        str(root),
    ]
    if check_only:
        args.append("--dry-run")

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(diff_text.encode("utf-8"))
    if proc.returncode != 0:
        failure = _format_patch_error(stderr, stdout)
        logger.error("patch failed: %s", failure)
        raise ToolExecutionError(f"patch failed: {failure}")


@tool()
@tool_error_handler
async def apply_patch(path: str, diff: str) -> str:
    """Apply a unified diff to a file within the workspace.

    Args:
        path: Target file path, absolute or relative to the workspace root.
        diff: Unified diff content to apply.

    Returns:
        Status message describing whether the patch was applied.
    """
    if error := await _check_rate_limit():
        return error

    ctx = get_runtime()
    root = ctx.workspace_root

    result = await validate_path_safe_async(Path(path).expanduser(), root)
    if isinstance(result, Err):
        raise ToolExecutionError(result.error.message)
    target = result.value
    if target.is_dir():
        raise ToolExecutionError(f"Cannot apply a patch to directory {target}.")

    if not diff.strip():
        raise ToolExecutionError("Patch text is empty.")

    _validate_patch_targets(diff, target, root)
    check_only = ctx.dry_run

    if is_git_workspace(root):
        await _apply_patch_with_git(diff, root, check_only=check_only)
        if check_only:
            return f"Dry-run: patch validated for {target} (no changes written)."
        return "Patch applied successfully via git."

    await _apply_patch_with_system_patch(diff, root, check_only=check_only)
    if check_only:
        return f"Dry-run: patch validated for {target} (no changes written)."
    return "Patch applied successfully via patch."
