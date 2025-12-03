"""
Security utilities for workspace and path validation.

This module provides secure path handling to prevent directory traversal
and workspace escape attacks. All validation functions return Result types
for explicit error handling.

Usage:
    from jpscripts.core.security import validate_path, validate_workspace_root
    from jpscripts.core.result import Err, Ok

    match validate_path(user_input, workspace_root):
        case Ok(safe_path):
            # Proceed with safe_path
        case Err(err):
            # Handle error: err.message
"""

from __future__ import annotations

import asyncio
import functools
import getpass
import os
import subprocess
from pathlib import Path
from typing import IO, Any

from jpscripts.core.result import Err, Ok, Result, SecurityError, WorkspaceError

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

MAX_SYMLINK_DEPTH = 10
"""Maximum symlink chain depth before rejecting as potentially malicious."""

FORBIDDEN_ROOTS = frozenset(
    {
        Path("/etc"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/root"),
        Path("/System"),  # macOS
        Path("/Library"),  # macOS
        Path("/Windows"),  # Windows
        Path("/Program Files"),  # Windows
    }
)
"""System directories that should never be accessed even via symlink.

Note: /var is intentionally excluded because macOS temp directories
resolve to /private/var. The workspace validation already ensures
paths stay within the workspace root.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_git_repo(path: Path) -> bool:
    """Check if path is a git repository."""
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False

    try:
        proc = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False

    if proc.returncode != 0:
        return False
    return proc.stdout.strip().lower() == "true"


def _is_owned_by_current_user(path: Path) -> bool:
    """Check if path is owned by the current user."""
    try:
        stat_result = path.stat()
    except OSError:
        return False

    # Unix: use getuid()
    if hasattr(os, "getuid"):
        try:
            return stat_result.st_uid == os.getuid()
        except Exception:
            return False

    # Windows/fallback: compare owner name
    try:
        return path.owner() == getpass.getuser()
    except Exception:
        return False


async def _is_git_repo_async(path: Path) -> bool:
    """Async check if path is a git repository.

    Uses asyncio.create_subprocess_exec to avoid blocking the event loop.
    Path resolution is offloaded to a thread to avoid blocking.
    """
    try:
        resolved = await asyncio.to_thread(lambda: path.expanduser().resolve())
    except OSError:
        return False

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(resolved),
            "rev-parse",
            "--is-inside-work-tree",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except OSError:
        return False

    return proc.returncode == 0 and stdout.decode().strip().lower() == "true"


def _is_forbidden_path(candidate: Path) -> bool:
    """Check if path resolves to a forbidden system location.

    This provides defense-in-depth against symlink attacks that might
    escape the workspace and target system directories.
    """
    try:
        for forbidden in FORBIDDEN_ROOTS:
            if candidate == forbidden or forbidden in candidate.parents:
                return True
    except (ValueError, OSError):
        pass
    return False


def _resolve_with_limit(
    path: Path, max_depth: int = MAX_SYMLINK_DEPTH
) -> Result[Path, SecurityError]:
    """Resolve path with symlink depth limit to prevent abuse.

    Deep symlink chains are often indicative of attack attempts or
    filesystem misconfigurations. This function limits the depth
    to prevent resource exhaustion and aid auditability.

    Args:
        path: The path to resolve
        max_depth: Maximum symlink hops allowed (default: MAX_SYMLINK_DEPTH)

    Returns:
        Ok(resolved_path) if successful, Err(SecurityError) if depth exceeded
    """
    current = path.expanduser()
    visited: set[Path] = set()

    for _ in range(max_depth):
        if current in visited:
            return Err(
                SecurityError(
                    f"Circular symlink detected: {path}",
                    context={"path": str(path), "current": str(current)},
                )
            )
        visited.add(current)

        if not current.is_symlink():
            return Ok(current.resolve())

        try:
            target = current.readlink()
        except OSError as exc:
            return Err(
                SecurityError(
                    f"Failed to read symlink: {path}: {exc}",
                    context={"path": str(path), "current": str(current)},
                )
            )

        current = target if target.is_absolute() else current.parent / target

    return Err(
        SecurityError(
            f"Symlink chain too deep (>{max_depth}): {path}",
            context={"path": str(path), "max_depth": max_depth},
        )
    )


async def _resolve_with_limit_async(
    path: Path, max_depth: int = MAX_SYMLINK_DEPTH
) -> Result[Path, SecurityError]:
    """Async resolve path with symlink depth limit to prevent abuse.

    This is the non-blocking version that offloads filesystem operations
    to a thread pool to avoid blocking the event loop.

    Args:
        path: The path to resolve
        max_depth: Maximum symlink hops allowed (default: MAX_SYMLINK_DEPTH)

    Returns:
        Ok(resolved_path) if successful, Err(SecurityError) if depth exceeded
    """
    # Offload the entire blocking operation to a thread
    return await asyncio.to_thread(_resolve_with_limit, path, max_depth)


# ---------------------------------------------------------------------------
# Result-based API (recommended for new code)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _validate_workspace_root_cached(resolved: Path) -> Result[Path, WorkspaceError]:
    """Cached workspace validation for normalized paths."""
    if not resolved.exists():
        return Err(
            WorkspaceError(
                f"Workspace root does not exist: {resolved}",
                context={"path": str(resolved)},
            )
        )

    if not resolved.is_dir():
        return Err(
            WorkspaceError(
                f"Workspace root is not a directory: {resolved}",
                context={"path": str(resolved)},
            )
        )

    if not (_is_owned_by_current_user(resolved) or _is_git_repo(resolved)):
        return Err(
            WorkspaceError(
                f"Workspace root must be a git repository or owned by current user: {resolved}",
                context={"path": str(resolved)},
            )
        )

    return Ok(resolved)


def validate_workspace_root(root: Path | str) -> Result[Path, WorkspaceError]:
    """Validate and resolve a workspace root path.

    A valid workspace root must:
    - Exist
    - Be a directory
    - Be either a git repository OR owned by the current user

    Args:
        root: The workspace root path to validate

    Returns:
        Ok(resolved_path) if valid, Err(WorkspaceError) otherwise
    """
    try:
        resolved = Path(root).expanduser().resolve()
    except (RuntimeError, OSError) as exc:
        # RuntimeError: expanduser fails for invalid ~username (e.g., ~0, ~nonexistent)
        # OSError: various path resolution failures
        return Err(
            WorkspaceError(
                f"Invalid workspace root path: {exc}",
                context={"path": str(root), "error": str(exc)},
            )
        )
    return _validate_workspace_root_cached(resolved)


def validate_path(
    path: str | Path,
    root: Path | str,
) -> Result[Path, SecurityError]:
    """Validate that a path stays within the workspace root.

    Resolves the path (following symlinks with depth limiting) and ensures
    it does not escape the validated workspace root or target forbidden
    system directories.

    TOCTOU Warning:
        This validation happens at a point in time. For security-critical
        operations, consider using validate_and_open() which performs
        atomic validation and open with O_NOFOLLOW.

    Args:
        path: The path to validate
        root: The workspace root to validate against

    Returns:
        Ok(resolved_path) if safe, Err(SecurityError) otherwise
    """
    # First validate the workspace root
    root_result = validate_workspace_root(root)
    if isinstance(root_result, Err):
        # Convert WorkspaceError to SecurityError for consistent typing
        workspace_err = root_result.error
        return Err(
            SecurityError(
                f"Invalid workspace root: {workspace_err.message}",
                context=workspace_err.context,
            )
        )

    base_root = root_result.value

    # Resolve path with symlink depth limiting (TOCTOU mitigation)
    resolve_result = _resolve_with_limit(Path(path))
    if isinstance(resolve_result, Err):
        return resolve_result

    candidate = resolve_result.value

    # Check for forbidden system paths (defense-in-depth)
    if _is_forbidden_path(candidate):
        return Err(
            SecurityError(
                f"Path targets forbidden system location: {candidate}",
                context={
                    "path": str(candidate),
                    "workspace_root": str(base_root),
                },
            )
        )

    try:
        candidate.relative_to(base_root)
    except ValueError:
        return Err(
            SecurityError(
                f"Path escapes workspace: {candidate}",
                context={
                    "path": str(candidate),
                    "workspace_root": str(base_root),
                },
            )
        )

    return Ok(candidate)


# ---------------------------------------------------------------------------
# Async Result-based API (for async contexts)
# ---------------------------------------------------------------------------


async def validate_workspace_root_async(root: Path | str) -> Result[Path, WorkspaceError]:
    """Async validate and resolve a workspace root path.

    A valid workspace root must:
    - Exist
    - Be a directory
    - Be either a git repository OR owned by the current user

    This async version uses non-blocking operations for all filesystem I/O
    to avoid blocking the event loop in high-concurrency scenarios.

    Args:
        root: The workspace root path to validate

    Returns:
        Ok(resolved_path) if valid, Err(WorkspaceError) otherwise
    """
    # Offload blocking path resolution to thread
    try:
        resolved = await asyncio.to_thread(lambda: Path(root).expanduser().resolve())
    except (RuntimeError, OSError) as exc:
        # RuntimeError: expanduser fails for invalid ~username (e.g., ~0, ~nonexistent)
        # OSError: various path resolution failures
        return Err(
            WorkspaceError(
                f"Invalid workspace root path: {exc}",
                context={"path": str(root), "error": str(exc)},
            )
        )

    # Offload blocking filesystem checks to thread
    exists = await asyncio.to_thread(resolved.exists)
    if not exists:
        return Err(
            WorkspaceError(
                f"Workspace root does not exist: {resolved}",
                context={"path": str(resolved)},
            )
        )

    is_dir = await asyncio.to_thread(resolved.is_dir)
    if not is_dir:
        return Err(
            WorkspaceError(
                f"Workspace root is not a directory: {resolved}",
                context={"path": str(resolved)},
            )
        )

    # Offload ownership check to thread
    is_owned = await asyncio.to_thread(_is_owned_by_current_user, resolved)
    is_git = await _is_git_repo_async(resolved) if not is_owned else False

    if not (is_owned or is_git):
        return Err(
            WorkspaceError(
                f"Workspace root must be a git repository or owned by current user: {resolved}",
                context={"path": str(resolved)},
            )
        )

    return Ok(resolved)


async def validate_path_async(
    path: str | Path,
    root: Path | str,
) -> Result[Path, SecurityError]:
    """Async validate that a path stays within the workspace root.

    Resolves the path (following symlinks with depth limiting) and ensures
    it does not escape the validated workspace root or target forbidden
    system directories.

    This async version uses non-blocking operations for all filesystem I/O
    to avoid blocking the event loop in high-concurrency scenarios.

    TOCTOU Warning:
        This validation happens at a point in time. For security-critical
        operations, consider using validate_and_open() which performs
        atomic validation and open with O_NOFOLLOW.

    Args:
        path: The path to validate
        root: The workspace root to validate against

    Returns:
        Ok(resolved_path) if safe, Err(SecurityError) otherwise
    """
    # First validate the workspace root (async, non-blocking)
    root_result = await validate_workspace_root_async(root)
    if isinstance(root_result, Err):
        # Convert WorkspaceError to SecurityError for consistent typing
        workspace_err = root_result.error
        return Err(
            SecurityError(
                f"Invalid workspace root: {workspace_err.message}",
                context=workspace_err.context,
            )
        )

    base_root = root_result.value

    # Resolve path with symlink depth limiting (async, non-blocking)
    resolve_result = await _resolve_with_limit_async(Path(path))
    if isinstance(resolve_result, Err):
        return resolve_result

    candidate = resolve_result.value

    # Check for forbidden system paths (defense-in-depth)
    # Note: _is_forbidden_path is a fast in-memory check on path.parents,
    # no filesystem I/O involved, so no need for to_thread
    if _is_forbidden_path(candidate):
        return Err(
            SecurityError(
                f"Path targets forbidden system location: {candidate}",
                context={
                    "path": str(candidate),
                    "workspace_root": str(base_root),
                },
            )
        )

    try:
        candidate.relative_to(base_root)
    except ValueError:
        return Err(
            SecurityError(
                f"Path escapes workspace: {candidate}",
                context={
                    "path": str(candidate),
                    "workspace_root": str(base_root),
                },
            )
        )

    return Ok(candidate)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def is_git_workspace(root: Path | str) -> bool:
    """Check if the path is a validated workspace and a git repository.

    Args:
        root: The workspace root to check

    Returns:
        True if the path is a valid workspace and a git repository
    """
    result = validate_workspace_root(root)
    if isinstance(result, Err):
        return False
    return _is_git_repo(result.value)


def is_path_safe(path: str | Path, root: Path | str) -> bool:
    """Check if a path is safe (within workspace) without raising exceptions.

    Args:
        path: The path to check
        root: The workspace root

    Returns:
        True if the path is safe, False otherwise
    """
    return validate_path(path, root).is_ok()


def validate_and_open(
    path: str | Path,
    root: Path | str,
    mode: str = "r",
    **kwargs: Any,
) -> Result[IO[Any], SecurityError]:
    """Validate path and open file atomically to prevent TOCTOU attacks.

    This function combines path validation with file opening using O_NOFOLLOW
    to prevent time-of-check-time-of-use vulnerabilities where an attacker
    might modify a symlink between validation and use.

    Args:
        path: The path to validate and open
        root: The workspace root to validate against
        mode: File mode ('r', 'rb', 'w', 'wb', etc.)
        **kwargs: Additional arguments passed to os.fdopen()

    Returns:
        Ok(file_handle) if successful, Err(SecurityError) otherwise

    Example:
        result = validate_and_open("config.json", workspace_root, "r")
        if result.is_ok():
            with result.value as f:
                data = json.load(f)
    """
    result = validate_path(path, root)
    if isinstance(result, Err):
        return result

    safe_path = result.value

    # Determine open flags based on mode
    flags = os.O_NOFOLLOW  # Prevent symlink following at open time
    if "w" in mode or "a" in mode:
        flags |= os.O_WRONLY | os.O_CREAT
        if "w" in mode:
            flags |= os.O_TRUNC
        elif "a" in mode:
            flags |= os.O_APPEND
    else:
        flags |= os.O_RDONLY

    try:
        fd = os.open(str(safe_path), flags)
        # Convert binary mode to text mode for fdopen
        fdopen_mode = mode.replace("t", "")  # Remove explicit text marker if present
        return Ok(os.fdopen(fd, fdopen_mode, **kwargs))
    except OSError as exc:
        return Err(
            SecurityError(
                f"Failed to open {path}: {exc}",
                context={
                    "path": str(safe_path),
                    "mode": mode,
                    "error": str(exc),
                },
            )
        )


__all__ = [
    "FORBIDDEN_ROOTS",
    "MAX_SYMLINK_DEPTH",
    "is_git_workspace",
    "is_path_safe",
    "validate_and_open",
    "validate_path",
    "validate_path_async",
    "validate_workspace_root",
    "validate_workspace_root_async",
]
