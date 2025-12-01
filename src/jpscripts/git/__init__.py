"""Git operations and repository management.

This package provides async git operations:
    - AsyncRepo: Non-blocking git commands
    - Branch status and tracking
    - Worktree management
    - Commit and diff operations
"""

from __future__ import annotations

from .client import (
    AsyncRepo,
    BranchStatus,
    GitCommit,
    GitOperationError,
    WorktreeInfo,
    is_repo,
    iter_git_repos,
)
from .ops import (
    BranchSummary,
    branch_statuses,
    commit_all,
    format_status,
    undo_last_commit,
)

__all__ = [
    "AsyncRepo",
    "BranchStatus",
    "BranchSummary",
    "GitCommit",
    "GitOperationError",
    "WorktreeInfo",
    "branch_statuses",
    "commit_all",
    "format_status",
    "is_repo",
    "iter_git_repos",
    "undo_last_commit",
]
