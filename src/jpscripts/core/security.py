from __future__ import annotations

import getpass
import os
from pathlib import Path

from git.exc import InvalidGitRepositoryError, NoSuchPathError

from jpscripts.core import git as git_core


class WorkspaceValidationError(PermissionError):
    """Raised when the workspace root fails validation."""


def validate_workspace_root(root: Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    if not resolved.exists():
        raise WorkspaceValidationError(f"Workspace root {resolved} does not exist.")
    if not resolved.is_dir():
        raise WorkspaceValidationError(f"Workspace root {resolved} is not a directory.")
    if not (_is_owned_by_current_user(resolved) or _is_git_repo(resolved)):
        raise WorkspaceValidationError(
            f"Workspace root {resolved} must be a git repository or owned by the current user."
        )

    return resolved


def validate_path(path: str | Path, root: Path) -> Path:
    """
    Resolve a user-supplied path and ensure it stays within the provided workspace root.
    Raises PermissionError if the path escapes the root (including via symlinks).
    """
    base_root = validate_workspace_root(root)
    candidate = Path(path).expanduser().resolve()

    try:
        candidate.relative_to(base_root)
    except ValueError:
        raise PermissionError(f"Access to {candidate} is outside allowed root {base_root}")

    return candidate


def is_git_workspace(root: Path) -> bool:
    """Return True if the path is a validated workspace and a git repository."""
    try:
        validated = validate_workspace_root(root)
    except WorkspaceValidationError:
        return False

    return _is_git_repo(validated)


def _is_git_repo(path: Path) -> bool:
    try:
        return git_core.is_repo(path)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return False
    except Exception:
        return False


def _is_owned_by_current_user(path: Path) -> bool:
    try:
        stat_result = path.stat()
    except OSError:
        return False

    if hasattr(os, "getuid"):
        try:
            return stat_result.st_uid == os.getuid()
        except Exception:
            return False

    try:
        return path.owner() == getpass.getuser()
    except Exception:
        return False
