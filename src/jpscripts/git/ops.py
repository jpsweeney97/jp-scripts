"""High-level git operations.

Provides convenience functions for common git workflows:
    - Status formatting
    - Branch summaries
    - Commit operations
    - Undo functionality
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

from jpscripts.core.result import Err, GitError, Ok, Result

from . import client as git_core

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


async def commit_all(repo: git_core.AsyncRepo, message: str) -> Result[str, GitError]:
    """
    Stage all changes and create a commit. Returns the commit SHA.
    """
    match await repo.add(all=True, paths=[]):
        case Err(err):
            return Err(err)
        case Ok(_):
            pass

    match await repo.status():
        case Err(err):
            return Err(err)
        case Ok(status):
            if not status.dirty:
                return Err(GitError("No changes to commit."))

    return await repo.commit(message)


async def undo_last_commit(repo: git_core.AsyncRepo, hard: bool = False) -> Result[str, GitError]:
    """
    Reset the current branch back one commit.
    Refuses to operate if the branch is behind its upstream to avoid history rewrites.
    """
    match await repo.get_commits("HEAD", 1):
        case Err(err):
            return Err(err)
        case Ok(commits):
            if not commits:
                return Err(GitError("Repo has no commits to undo."))

    match await repo.status():
        case Err(err):
            return Err(err)
        case Ok(status):
            if status.upstream and status.behind > 0:
                return Err(GitError("Refusing to undo: branch is behind upstream. Pull first."))

    mode = "--hard" if hard else "--soft"
    match await repo.reset(mode, "HEAD~1"):
        case Err(err):
            return Err(err)
        case Ok(_):
            pass

    return Ok(f"Reset {status.branch} one commit back ({mode}).")


@dataclass
class BranchSummary:
    name: str
    upstream: str | None
    ahead: int
    behind: int
    error: str | None = None


def _parse_ahead_behind(track: str) -> tuple[int, int]:
    ahead = 0
    behind = 0
    if not track:
        return ahead, behind

    cleaned = track.strip().strip("[]")
    ahead_match = re.search(r"ahead\s+(\d+)", cleaned)
    behind_match = re.search(r"behind\s+(\d+)", cleaned)
    if ahead_match:
        ahead = int(ahead_match.group(1))
    if behind_match:
        behind = int(behind_match.group(1))
    return ahead, behind


def _parse_ref_line(line: str) -> tuple[str, str | None, int, int]:
    parts = line.split(" ", 2)
    name = parts[0].strip()
    upstream = parts[1].strip() if len(parts) > 1 else ""
    track = parts[2].strip() if len(parts) > 2 else ""
    ahead, behind = _parse_ahead_behind(track)
    return name, upstream or None, ahead, behind


async def branch_statuses(repo: git_core.AsyncRepo) -> Result[list[BranchSummary], GitError]:
    """Return ahead/behind information for all branches in a repo using async plumbing."""
    runner = getattr(repo, "run_git", None)
    legacy_runner = getattr(repo, "_run_git", None)
    git_call: Callable[..., Awaitable[Result[str, GitError]]] | None = None
    if callable(runner):
        git_call = cast(Callable[..., Awaitable[Result[str, GitError]]], runner)
    elif callable(legacy_runner):
        git_call = cast(Callable[..., Awaitable[Result[str, GitError]]], legacy_runner)
    else:
        return Err(GitError("Repository does not support git execution"))

    if git_call is None:
        return Err(GitError("Repository does not support git execution"))

    match await git_call(
        "for-each-ref",
        "--format=%(refname:short) %(upstream:short) %(upstream:track)",
        "refs/heads",
    ):
        case Err(err):
            return Err(err)
        case Ok(output):
            pass

    summaries: list[BranchSummary] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            name, upstream, ahead, behind = _parse_ref_line(line)
            summaries.append(BranchSummary(name, upstream, ahead, behind, None))
        except Exception as exc:
            summaries.append(BranchSummary(line.strip(), None, 0, 0, str(exc)))
    return Ok(summaries)
