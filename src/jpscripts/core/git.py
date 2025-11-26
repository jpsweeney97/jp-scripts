from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo


@dataclass
class BranchStatus:
    path: Path
    branch: str
    upstream: str | None
    ahead: int
    behind: int
    staged: int
    unstaged: int
    untracked: int
    dirty: bool
    error: str | None = None


def open_repo(path: Path | str = ".") -> Repo:
    """Open a git repository, searching parent directories by default."""
    return Repo(path, search_parent_directories=True)


def is_repo(path: Path | str = ".") -> bool:
    try:
        open_repo(path)
        return True
    except (InvalidGitRepositoryError, NoSuchPathError):
        return False


def get_upstream(repo: Repo) -> str | None:
    try:
        if repo.head.is_detached:
            return None
        tracking = repo.active_branch.tracking_branch()
        return str(tracking) if tracking else None
    except (TypeError, GitCommandError):
        return None


def _count_commits(repo: Repo, ref_range: str) -> int:
    try:
        output = repo.git.rev_list("--count", ref_range)
        return int(output.strip())
    except GitCommandError:
        return 0


def describe_status(repo: Repo) -> BranchStatus:
    path = Path(repo.working_tree_dir or ".")
    try:
        status_output = repo.git.status("--porcelain=v2", "--branch", "-z")
    except GitCommandError as exc:
        return BranchStatus(
            path=path,
            branch="(unknown)",
            upstream=None,
            ahead=0,
            behind=0,
            staged=0,
            unstaged=0,
            untracked=0,
            dirty=repo.is_dirty(untracked_files=True),
            error=str(exc),
        )

    branch = "(unknown)"
    upstream: str | None = None
    ahead = 0
    behind = 0
    staged = 0
    unstaged = 0
    untracked = 0

    entries = [line for line in status_output.split("\0") if line]
    for entry in entries:
        line = entry.strip()
        if not line:
            continue

        if line.startswith("#"):
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "branch.head":
                branch = parts[2]
            elif len(parts) >= 3 and parts[1] == "branch.upstream":
                upstream = parts[2]
            elif len(parts) >= 4 and parts[1] == "branch.ab":
                try:
                    ahead = int(parts[2].lstrip("+"))
                    behind = int(parts[3].lstrip("-"))
                except ValueError:
                    ahead = behind = 0
            continue

        kind = line[0]
        if kind in {"1", "2", "u"}:
            parts = line.split()
            if len(parts) < 2:
                continue
            xy = parts[1]
            if len(xy) >= 1 and xy[0] != ".":
                staged += 1
            if len(xy) >= 2 and xy[1] != ".":
                unstaged += 1
        elif kind == "?":
            untracked += 1
        elif kind == "!":
            continue

    dirty = bool(staged or unstaged or untracked)

    return BranchStatus(
        path=path,
        branch=branch,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        dirty=dirty,
        error=None,
    )


def iter_git_repos(root: Path, max_depth: int = 2) -> Iterator[Path]:
    """Yield git working directories under root up to the requested depth."""
    root = root.expanduser()
    seen: set[Path] = set()

    for git_dir in root.rglob(".git"):
        repo_root = git_dir.parent
        depth = len(repo_root.relative_to(root).parts)
        if depth > max_depth:
            continue
        if repo_root in seen:
            continue
        seen.add(repo_root)
        yield repo_root


def open_many(paths: Sequence[Path]) -> list[Repo]:
    repos: list[Repo] = []
    for path in paths:
        try:
            repos.append(open_repo(path))
        except InvalidGitRepositoryError:
            continue
    return repos
