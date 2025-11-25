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
    branch = "(detached)"
    try:
        if repo.head.is_detached:
            branch = repo.git.rev_parse("--short", "HEAD")
        else:
            branch = repo.active_branch.name
    except GitCommandError:
        pass

    upstream = get_upstream(repo)
    ahead = _count_commits(repo, f"{upstream}..HEAD") if upstream else 0
    behind = _count_commits(repo, f"HEAD..{upstream}") if upstream else 0

    staged = len(repo.index.diff("HEAD"))
    unstaged = len(repo.index.diff(None))
    untracked = len(repo.untracked_files)

    return BranchStatus(
        path=Path(repo.working_tree_dir or "."),
        branch=branch,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        dirty=repo.is_dirty(untracked_files=True),
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
