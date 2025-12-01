"""Patch parsing and application utilities.

This module provides functions for parsing unified diffs, applying patches
via git or the system patch command, and handling patch failures.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from jpscripts.core import security
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err

logger = get_logger(__name__)


def compute_patch_hash(patch_text: str) -> str:
    """Compute SHA256 hash of a patch for de-duplication."""
    return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


async def extract_patch_paths(patch_text: str, root: Path) -> list[Path]:
    """Extract and validate file paths from unified diff headers.

    Parses +++ and --- lines, strips a/ b/ prefixes, and validates
    each path against the workspace root.

    Args:
        patch_text: The unified diff patch content
        root: The workspace root directory for path validation

    Returns:
        Sorted list of validated paths found in the patch
    """
    candidates: set[Path] = set()
    for raw_line in patch_text.splitlines():
        if not raw_line.startswith(("+++ ", "--- ")):
            continue
        try:
            _, path_str = raw_line.split(" ", 1)
        except ValueError:
            continue
        path_str = path_str.strip()
        if path_str in {"/dev/null", "dev/null", "a/dev/null", "b/dev/null"}:
            continue
        if path_str.startswith(("a/", "b/")):
            path_str = path_str[2:]
        result = await security.validate_path_safe_async(root / path_str, root)
        if isinstance(result, Err):
            logger.debug("Skipped unsafe patch path %s: %s", path_str, result.error.message)
            continue
        candidates.add(result.value)
    return sorted(candidates)


async def write_failed_patch(patch_text: str, root: Path) -> None:
    """Write a failed patch to agent_failed_patch.diff for inspection.

    Args:
        patch_text: The patch content that failed to apply
        root: The workspace root directory
    """
    result = await security.validate_path_safe_async(root / "agent_failed_patch.diff", root)
    if isinstance(result, Err):
        logger.debug("Unable to persist failed patch: %s", result.error.message)
        return
    try:
        await asyncio.to_thread(result.value.write_text, patch_text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to persist failed patch for inspection: %s", exc)


async def apply_patch_text(patch_text: str, root: Path) -> list[Path]:
    """Apply a unified diff patch to the repository.

    Attempts to apply the patch using git apply first, falling back to
    the standard patch command if git apply fails.

    Args:
        patch_text: The unified diff patch content
        root: The root directory to apply the patch in

    Returns:
        List of paths that were successfully patched, or empty list on failure
    """
    if not patch_text.strip():
        return []

    target_paths = await extract_patch_paths(patch_text, root)

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--whitespace=nowarn",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        proc = None

    if proc:
        _stdout, stderr = await proc.communicate(patch_text.encode())
        if proc.returncode == 0:
            return target_paths
        logger.debug("git apply failed: %s", stderr.decode(errors="replace"))

    try:
        fallback = await asyncio.create_subprocess_exec(
            "patch",
            "-p1",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        await write_failed_patch(patch_text, root)
        return []

    out, err = await fallback.communicate(patch_text.encode())
    if fallback.returncode == 0:
        return target_paths

    logger.error(
        "Patch application failed: %s",
        err.decode(errors="replace") or out.decode(errors="replace"),
    )
    await write_failed_patch(patch_text, root)
    return []


__all__ = [
    "apply_patch_text",
    "compute_patch_hash",
    "extract_patch_paths",
    "write_failed_patch",
]
