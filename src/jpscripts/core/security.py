"""
Security utilities for workspace and path validation.

This module provides secure path handling to prevent directory traversal
and workspace escape attacks. It offers both exception-based and Result-based
APIs for flexibility in error handling.

Usage:
    # Exception-based (backward compatible)
    from jpscripts.core.security import validate_path, validate_workspace_root

    try:
        safe_path = validate_path(user_input, workspace_root)
    except PermissionError:
        handle_error()

    # Result-based (recommended for new code)
    from jpscripts.core.security import validate_path_safe, validate_workspace_root_safe

    result = validate_path_safe(user_input, workspace_root)
    if result.is_ok():
        safe_path = result.value
    else:
        error = result.error
"""

from __future__ import annotations

import asyncio
import getpass
import os
from pathlib import Path
from typing import Any

from jpscripts.core import git as git_core
from jpscripts.core.result import Err, Ok, Result, SecurityError, WorkspaceError


# ---------------------------------------------------------------------------
# Backward-compatible exception (alias)
# ---------------------------------------------------------------------------


class WorkspaceValidationError(PermissionError, WorkspaceError):
    """Raised when the workspace root fails validation.

    This class inherits from both PermissionError (for backward compatibility)
    and WorkspaceError (for integration with the new error hierarchy).

    DEPRECATED: New code should catch WorkspaceError or SecurityError instead.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        PermissionError.__init__(self, message)
        WorkspaceError.__init__(self, message, context=context)


class PathValidationError(PermissionError, SecurityError):
    """Raised when path validation fails (escape attempt detected).

    This class inherits from both PermissionError (for backward compatibility)
    and SecurityError (for integration with the new error hierarchy).

    DEPRECATED: New code should catch SecurityError instead.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        PermissionError.__init__(self, message)
        SecurityError.__init__(self, message, context=context)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_git_repo(path: Path) -> bool:
    """Check if path is a git repository."""
    async def _probe() -> bool:
        match await git_core.is_repo(path):
            case Ok(flag):
                return flag
            case Err(_):
                return False

    try:
        return asyncio.run(_probe())
    except RuntimeError:
        # Fallback when running inside an event loop; treat as not a repo to avoid blocking.
        return False


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


# ---------------------------------------------------------------------------
# Result-based API (recommended for new code)
# ---------------------------------------------------------------------------


def validate_workspace_root_safe(root: Path | str) -> Result[Path, WorkspaceError]:
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
    resolved = Path(root).expanduser().resolve()

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


def validate_path_safe(
    path: str | Path,
    root: Path | str,
) -> Result[Path, SecurityError]:
    """Validate that a path stays within the workspace root.

    Resolves the path (following symlinks) and ensures it does not
    escape the validated workspace root.

    Args:
        path: The path to validate
        root: The workspace root to validate against

    Returns:
        Ok(resolved_path) if safe, Err(SecurityError) otherwise
    """
    # First validate the workspace root
    root_result = validate_workspace_root_safe(root)
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
    candidate = Path(path).expanduser().resolve()

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
# Exception-based API (backward compatible)
# ---------------------------------------------------------------------------


def validate_workspace_root(root: Path | str) -> Path:
    """Validate and resolve a workspace root path.

    A valid workspace root must:
    - Exist
    - Be a directory
    - Be either a git repository OR owned by the current user

    Args:
        root: The workspace root path to validate

    Returns:
        The resolved, validated path

    Raises:
        WorkspaceValidationError: If validation fails
    """
    result = validate_workspace_root_safe(root)
    if isinstance(result, Err):
        raise WorkspaceValidationError(
            result.error.message,
            context=result.error.context,
        )
    return result.value


def validate_path(path: str | Path, root: Path | str) -> Path:
    """Validate that a path stays within the workspace root.

    Resolves the path (following symlinks) and ensures it does not
    escape the validated workspace root.

    Args:
        path: The path to validate
        root: The workspace root to validate against

    Returns:
        The resolved, validated path

    Raises:
        PermissionError: If the path escapes the workspace root
        WorkspaceValidationError: If the workspace root is invalid
    """
    result = validate_path_safe(path, root)
    if isinstance(result, Err):
        err = result.error
        # For backward compatibility, raise PermissionError for path escape
        # and WorkspaceValidationError for workspace issues
        if "workspace root" in err.message.lower():
            raise WorkspaceValidationError(err.message, context=err.context)
        raise PathValidationError(err.message, context=err.context)
    return result.value


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
    result = validate_workspace_root_safe(root)
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
    return validate_path_safe(path, root).is_ok()


__all__ = [
    # Exception-based API (backward compatible)
    "validate_workspace_root",
    "validate_path",
    "WorkspaceValidationError",
    "PathValidationError",
    # Result-based API (recommended)
    "validate_workspace_root_safe",
    "validate_path_safe",
    # Utilities
    "is_git_workspace",
    "is_path_safe",
]
