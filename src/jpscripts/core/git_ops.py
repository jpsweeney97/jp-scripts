from __future__ import annotations

from git import GitCommandError, Repo

from . import git as git_core


class GitOperationError(RuntimeError):
    """Raised when a git operation fails."""


def format_status(status: git_core.BranchStatus) -> str:
    """Return a compact, multiline representation of BranchStatus."""
    upstream = status.upstream or "none"
    lines = [
        f"path: {status.path}",
        f"branch: {status.branch}",
        f"upstream: {upstream}",
        f"ahead: {status.ahead}",
        f"behind: {status.behind}",
        f"staged: {status.staged}",
        f"unstaged: {status.unstaged}",
        f"untracked: {status.untracked}",
        f"dirty: {status.dirty}",
    ]
    if status.error:
        lines.append(f"error: {status.error}")
    return "\n".join(lines)


def commit_all(repo: Repo, message: str) -> str:
    """
    Stage all changes and create a commit. Returns the commit SHA.
    Raises GitOperationError if there is nothing to commit or git fails.
    """
    try:
        repo.git.add(all=True)
        if not repo.is_dirty(untracked_files=True):
            raise GitOperationError("No changes to commit.")
        commit = repo.index.commit(message)
        return commit.hexsha
    except GitCommandError as exc:
        raise GitOperationError(str(exc)) from exc
