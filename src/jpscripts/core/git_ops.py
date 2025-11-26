from __future__ import annotations

from dataclasses import dataclass

from git import Repo
from git.objects import Commit

from . import git as git_core

GitOperationError = git_core.GitOperationError


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


async def commit_all(repo: git_core.AsyncRepo, message: str) -> str:
    """
    Stage all changes and create a commit. Returns the commit SHA.
    Raises GitOperationError if there is nothing to commit or git fails.
    """
    await repo.add(all=True, paths=[])
    status = await repo.status()
    if not status.dirty:
        raise GitOperationError("No changes to commit.")
    return await repo.commit(message)


async def undo_last_commit(repo: git_core.AsyncRepo, hard: bool = False) -> str:
    """
    Reset the current branch back one commit.
    Refuses to operate if the branch is behind its upstream to avoid history rewrites.
    """
    commits = await repo.get_commits("HEAD", 1)
    if not commits:
        raise GitOperationError("Repo has no commits to undo.")

    status = await repo.status()
    if status.upstream and status.behind > 0:
        raise GitOperationError("Refusing to undo: branch is behind upstream. Pull first.")

    mode = "--hard" if hard else "--soft"
    await repo.reset(mode, "HEAD~1")

    return f"Reset {status.branch} one commit back ({mode})."


@dataclass
class BranchSummary:
    name: str
    upstream: str | None
    ahead: int
    behind: int
    error: str | None = None


def branch_statuses(repo: Repo) -> list[BranchSummary]:
    """Return ahead/behind information for all branches in a repo."""
    summaries: list[BranchSummary] = []
    for branch in repo.branches:
        try:
            tracking = branch.tracking_branch()
            upstream = str(tracking) if tracking else None
            ahead = behind = 0
            if tracking:
                ahead = sum(1 for _ in repo.iter_commits(f"{tracking}..{branch.name}"))
                behind = sum(1 for _ in repo.iter_commits(f"{branch.name}..{tracking}"))
            summaries.append(BranchSummary(branch.name, upstream, ahead, behind, None))
        except Exception as exc:
            summaries.append(BranchSummary(branch.name, None, 0, 0, str(exc)))
    return summaries
