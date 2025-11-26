from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Iterator, Sequence


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


@dataclass
class GitCommit:
    hexsha: str
    summary: str
    author_name: str
    author_email: str
    committed_date: int

    @property
    def committed_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.committed_date)


class GitOperationError(RuntimeError):
    """Raised when git operations fail."""


async def _run_git(cwd: Path, *args: str) -> str:
    """Run git with asyncio and return stdout as text, raising on failure."""
    if not cwd.exists():
        raise GitOperationError(f"Repository path does not exist: {cwd}")

    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise GitOperationError("git executable not found on PATH") from exc
    except Exception as exc:
        raise GitOperationError(f"Failed to start git: {exc}") from exc

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        detail = message or stdout_text or f"git {' '.join(args)} failed"
        raise GitOperationError(detail)

    return stdout.decode("utf-8", errors="replace")


def _parse_status_output(raw: str, repo_path: Path) -> BranchStatus:
    branch = "(unknown)"
    upstream: str | None = None
    ahead = behind = staged = unstaged = untracked = 0

    entries = [item for item in raw.split("\0") if item]
    for entry in entries:
        if entry.startswith("#"):
            parts = entry.split()
            if len(parts) >= 3 and parts[1] == "branch.head":
                branch = parts[2]
            elif len(parts) >= 3 and parts[1] == "branch.upstream":
                upstream = parts[2]
            elif len(parts) >= 4 and parts[1] == "branch.ab":
                ahead = _safe_int(parts[2].lstrip("+"))
                behind = _safe_int(parts[3].lstrip("-"))
            continue

        kind = entry[0]
        if kind in {"1", "2", "u"}:
            parts = entry.split()
            if len(parts) < 2:
                continue
            xy = parts[1]
            if len(xy) >= 1 and xy[0] != ".":
                staged += 1
            if len(xy) >= 2 and xy[1] != ".":
                unstaged += 1
        elif kind == "?":
            untracked += 1

    dirty = bool(staged or unstaged or untracked)
    return BranchStatus(
        path=repo_path,
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


def _parse_commits(raw: str) -> list[GitCommit]:
    commits: list[GitCommit] = []
    records = [rec for rec in raw.split("\x1e") if rec.strip()]
    for record in records:
        fields = record.strip().split("\x00")
        if len(fields) < 5:
            continue
        sha, author_name, author_email, ts, summary = fields[:5]
        commits.append(
            GitCommit(
                hexsha=sha,
                summary=summary,
                author_name=author_name,
                author_email=author_email,
                committed_date=_safe_int(ts),
            )
        )
    return commits


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def _resolve_worktree(path: Path) -> Path:
    raw = await _run_git(path, "rev-parse", "--show-toplevel")
    resolved = Path(raw.strip()).resolve()
    return resolved


class AsyncRepo:
    """Async git wrapper built on subprocess plumbing."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def path(self) -> Path:
        return self._root

    @classmethod
    async def open(cls, path: Path | str = ".") -> AsyncRepo:
        root = Path(path).expanduser()
        resolved_root = await _resolve_worktree(root)
        return cls(resolved_root)

    async def status(self) -> BranchStatus:
        output = await _run_git(self._root, "status", "--porcelain=v2", "--branch", "-z")
        return _parse_status_output(output, self._root)

    async def add(self, *, all: bool = False, paths: Sequence[Path] | None = None) -> None:
        args: list[str] = ["add"]
        if all:
            args.append("--all")
        elif paths:
            args.extend(str(path) for path in paths)
        else:
            return
        await _run_git(self._root, *args)

    async def commit(self, message: str) -> str:
        await _run_git(self._root, "commit", "-m", message)
        sha = await _run_git(self._root, "rev-parse", "HEAD")
        return sha.strip()

    async def reset(self, mode: str, ref: str) -> None:
        await _run_git(self._root, "reset", mode, ref)

    async def fetch(self) -> None:
        await _run_git(self._root, "fetch", "--all", "--prune")

    async def get_commits(self, ref_range: str, limit: int) -> list[GitCommit]:
        format_str = "%H%x00%an%x00%ae%x00%at%x00%s%x1e"
        output = await _run_git(
            self._root,
            "log",
            f"--max-count={limit}",
            "--date=unix",
            f"--format={format_str}",
            ref_range,
        )
        return _parse_commits(output)

    async def diff_stat(self, ref_range: str) -> str:
        output = await _run_git(self._root, "diff", "--stat", ref_range)
        return output

    async def head(self, short: bool = True) -> str:
        args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
        output = await _run_git(self._root, *args)
        return output.strip()

    async def _run_git(self, *args: str) -> str:
        """Internal helper to run git commands for higher-level ops."""
        return await _run_git(self._root, *args)


def is_repo(path: Path | str = ".") -> bool:
    target = Path(path).expanduser()
    if not target.exists():
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


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
