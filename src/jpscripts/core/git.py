from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from jpscripts.core.result import Err, GitError, Ok, Result


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


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""

    path: Path
    branch: str
    commit: str
    is_locked: bool
    prunable: bool


class GitOperationError(GitError):
    """Raised when git operations fail."""


async def _run_git(cwd: Path, *args: str) -> Result[str, GitError]:
    """Run git with asyncio and return stdout as text, wrapping failures."""
    if not cwd.exists():
        return Err(GitError("Repository path does not exist", context={"cwd": str(cwd)}))

    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
    except FileNotFoundError:
        return Err(GitError("git executable not found on PATH", context={"cwd": str(cwd)}))
    except Exception as exc:
        return Err(
            GitError(
                "Failed to start git",
                context={"cwd": str(cwd), "args": list(args), "error": str(exc)},
            )
        )

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        detail = message or stdout_text or f"git {' '.join(args)} failed"
        return Err(
            GitError(
                detail,
                context={"cwd": str(cwd), "args": list(args), "returncode": process.returncode},
            )
        )

    return Ok(stdout.decode("utf-8", errors="replace"))


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


def _parse_worktree_list(output: str) -> list[WorktreeInfo]:
    """Parse `git worktree list --porcelain` output."""
    worktrees: list[WorktreeInfo] = []
    current: dict[str, str] = {}

    for line in output.splitlines():
        if not line.strip():
            if current:
                worktrees.append(
                    WorktreeInfo(
                        path=Path(current.get("worktree", "")),
                        branch=current.get("branch", "").replace("refs/heads/", ""),
                        commit=current.get("HEAD", ""),
                        is_locked="locked" in current,
                        prunable="prunable" in current,
                    )
                )
                current = {}
            continue

        if line.startswith("worktree "):
            current["worktree"] = line[9:]
        elif line.startswith("HEAD "):
            current["HEAD"] = line[5:]
        elif line.startswith("branch "):
            current["branch"] = line[7:]
        elif line == "locked":
            current["locked"] = "true"
        elif line == "prunable":
            current["prunable"] = "true"

    # Handle last entry if no trailing newline
    if current:
        worktrees.append(
            WorktreeInfo(
                path=Path(current.get("worktree", "")),
                branch=current.get("branch", "").replace("refs/heads/", ""),
                commit=current.get("HEAD", ""),
                is_locked="locked" in current,
                prunable="prunable" in current,
            )
        )

    return worktrees


async def _resolve_worktree(path: Path) -> Result[Path, GitError]:
    match await _run_git(path, "rev-parse", "--show-toplevel"):
        case Ok(raw):
            resolved = Path(raw.strip()).resolve()
            return Ok(resolved)
        case Err(err):
            return Err(err)


class AsyncRepo:
    """Async git wrapper built on subprocess plumbing."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def path(self) -> Path:
        return self._root

    @classmethod
    async def open(cls, path: Path | str = ".") -> Result[AsyncRepo, GitError]:
        root = Path(path).expanduser()
        match await _resolve_worktree(root):
            case Ok(resolved_root):
                return Ok(cls(resolved_root))
            case Err(err):
                return Err(err)

    async def status(self) -> Result[BranchStatus, GitError]:
        match await _run_git(self._root, "status", "--porcelain=v2", "--branch", "-z"):
            case Ok(output):
                return Ok(_parse_status_output(output, self._root))
            case Err(err):
                return Err(err)

    async def status_short(self) -> Result[list[tuple[str, str]], GitError]:
        """Return short status entries as (status_code, path)."""
        match await _run_git(self._root, "status", "--porcelain"):
            case Ok(output):
                entries: list[tuple[str, str]] = []
                for line in output.splitlines():
                    if not line:
                        continue
                    status_code = line[:2].strip()
                    path = line[3:] if len(line) > 3 else ""
                    entries.append((status_code, path))
                return Ok(entries)
            case Err(err):
                return Err(err)

    async def add(
        self, *, all: bool = False, paths: Sequence[Path] | None = None
    ) -> Result[None, GitError]:
        args: list[str] = ["add"]
        if all:
            args.append("--all")
        elif paths:
            args.extend(str(path) for path in paths)
        else:
            return Ok(None)
        result = await _run_git(self._root, *args)
        return result.map(lambda _: None)

    async def commit(self, message: str) -> Result[str, GitError]:
        match await _run_git(self._root, "commit", "-m", message):
            case Err(err):
                return Err(err)
            case Ok(_):
                pass

        match await _run_git(self._root, "rev-parse", "HEAD"):
            case Ok(sha):
                return Ok(sha.strip())
            case Err(err):
                return Err(err)

    async def get_remote_url(self, remote: str = "origin") -> Result[str, GitError]:
        match await _run_git(self._root, "remote", "get-url", remote):
            case Ok(output):
                return Ok(output.strip())
            case Err(err):
                return Err(
                    GitError(
                        f"Failed to get remote '{remote}'",
                        context={"cwd": str(self._root), "error": err.message},
                    )
                )

    async def stash_list(self) -> Result[list[str], GitError]:
        match await _run_git(self._root, "stash", "list"):
            case Ok(output):
                lines = [line for line in output.splitlines() if line.strip()]
                return Ok(lines)
            case Err(err):
                return Err(err)

    async def stash_apply(self, ref: str) -> Result[None, GitError]:
        result = await _run_git(self._root, "stash", "apply", ref)
        return result.map(lambda _: None)

    async def stash_pop(self, ref: str) -> Result[None, GitError]:
        result = await _run_git(self._root, "stash", "pop", ref)
        return result.map(lambda _: None)

    async def stash_drop(self, ref: str) -> Result[None, GitError]:
        result = await _run_git(self._root, "stash", "drop", ref)
        return result.map(lambda _: None)

    async def reset(self, mode: str, ref: str) -> Result[None, GitError]:
        result = await _run_git(self._root, "reset", mode, ref)
        return result.map(lambda _: None)

    async def fetch(self) -> Result[None, GitError]:
        result = await _run_git(self._root, "fetch", "--all", "--prune")
        return result.map(lambda _: None)

    async def get_commits(self, ref_range: str, limit: int) -> Result[list[GitCommit], GitError]:
        format_str = "%H%x00%an%x00%ae%x00%at%x00%s%x1e"
        match await _run_git(
            self._root,
            "log",
            f"--max-count={limit}",
            "--date=unix",
            f"--format={format_str}",
            ref_range,
        ):
            case Ok(output):
                return Ok(_parse_commits(output))
            case Err(err):
                return Err(err)

    async def diff_stat(self, ref_range: str) -> Result[str, GitError]:
        result = await _run_git(self._root, "diff", "--stat", ref_range)
        return result.map(lambda out: out)

    async def head(self, short: bool = True) -> Result[str, GitError]:
        args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
        match await _run_git(self._root, *args):
            case Ok(output):
                return Ok(output.strip())
            case Err(err):
                return Err(err)

    async def get_file_churn(self, path: Path) -> Result[int, GitError]:
        """Return the number of commits touching a path using git log --follow."""
        target = path
        if path.is_absolute():
            try:
                target = path.resolve().relative_to(self._root)
            except ValueError:
                target = path

        match await _run_git(self._root, "log", "--oneline", "--follow", "--", str(target)):
            case Err(err):
                return Err(err)
            case Ok(output):
                churn = sum(1 for line in output.splitlines() if line.strip())
                return Ok(churn)

    async def _run_git(self, *args: str) -> Result[str, GitError]:
        """Internal helper to run git commands for higher-level ops."""
        return await _run_git(self._root, *args)

    async def run_git(self, *args: str) -> Result[str, GitError]:
        """Public wrapper around git subprocess execution."""
        return await self._run_git(*args)

    # -------------------------------------------------------------------------
    # Worktree operations
    # -------------------------------------------------------------------------

    async def worktree_add(
        self,
        path: Path,
        branch: str,
        *,
        new_branch: bool = True,
        start_point: str | None = None,
    ) -> Result[Path, GitError]:
        """Create a new worktree.

        Args:
            path: Directory for the new worktree
            branch: Branch name (created if new_branch=True)
            new_branch: If True, create branch with -b flag
            start_point: Base commit/branch (default: HEAD)

        Returns:
            Ok(worktree_path) on success, Err(GitError) on failure

        [invariant:async-io] Uses asyncio subprocess
        """
        args: list[str] = ["worktree", "add"]
        if new_branch:
            args.extend(["-b", branch])
        args.append(str(path))
        if not new_branch:
            args.append(branch)
        if start_point:
            args.append(start_point)

        match await _run_git(self._root, *args):
            case Ok(_):
                return Ok(path.resolve())
            case Err(err):
                return Err(err)

    async def worktree_remove(
        self,
        path: Path,
        *,
        force: bool = False,
    ) -> Result[None, GitError]:
        """Remove a worktree.

        Args:
            path: Worktree directory to remove
            force: If True, remove even if dirty

        Returns:
            Ok(None) on success, Err(GitError) on failure

        [invariant:async-io] Uses asyncio subprocess
        """
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))

        result = await _run_git(self._root, *args)
        return result.map(lambda _: None)

    async def worktree_list(self) -> Result[list[WorktreeInfo], GitError]:
        """List all worktrees.

        Returns:
            Ok(list[WorktreeInfo]) on success

        [invariant:async-io] Uses asyncio subprocess
        """
        match await _run_git(self._root, "worktree", "list", "--porcelain"):
            case Ok(output):
                return Ok(_parse_worktree_list(output))
            case Err(err):
                return Err(err)

    async def worktree_prune(self) -> Result[None, GitError]:
        """Prune stale worktree references.

        [invariant:async-io] Uses asyncio subprocess
        """
        result = await _run_git(self._root, "worktree", "prune")
        return result.map(lambda _: None)

    # -------------------------------------------------------------------------
    # Merge operations
    # -------------------------------------------------------------------------

    async def merge(
        self,
        branch: str,
        *,
        no_ff: bool = False,
        message: str | None = None,
    ) -> Result[str, GitError]:
        """Merge a branch into current HEAD.

        Args:
            branch: Branch to merge
            no_ff: Force merge commit even if fast-forward possible
            message: Custom merge commit message

        Returns:
            Ok(commit_sha) on success, Err(GitError) on conflict/failure

        [invariant:async-io] Uses asyncio subprocess
        """
        args = ["merge"]
        if no_ff:
            args.append("--no-ff")
        if message:
            args.extend(["-m", message])
        args.append(branch)

        match await _run_git(self._root, *args):
            case Ok(_):
                return await self.head(short=False)
            case Err(err):
                return Err(err)

    async def merge_abort(self) -> Result[None, GitError]:
        """Abort an in-progress merge.

        [invariant:async-io] Uses asyncio subprocess
        """
        result = await _run_git(self._root, "merge", "--abort")
        return result.map(lambda _: None)

    async def get_conflict_files(self) -> Result[list[Path], GitError]:
        """Get list of files with merge conflicts.

        Returns:
            Ok(list[Path]) of conflicted files

        [invariant:async-io] Uses asyncio subprocess
        """
        match await _run_git(self._root, "diff", "--name-only", "--diff-filter=U"):
            case Ok(output):
                files = [self._root / line.strip() for line in output.splitlines() if line.strip()]
                return Ok(files)
            case Err(err):
                return Err(err)

    # -------------------------------------------------------------------------
    # Branch operations
    # -------------------------------------------------------------------------

    async def checkout_branch(
        self,
        branch: str,
        *,
        create: bool = False,
    ) -> Result[None, GitError]:
        """Checkout a branch.

        [invariant:async-io] Uses asyncio subprocess
        """
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(branch)
        result = await _run_git(self._root, *args)
        return result.map(lambda _: None)

    async def delete_branch(
        self,
        branch: str,
        *,
        force: bool = False,
    ) -> Result[None, GitError]:
        """Delete a local branch.

        [invariant:async-io] Uses asyncio subprocess
        """
        flag = "-D" if force else "-d"
        result = await _run_git(self._root, "branch", flag, branch)
        return result.map(lambda _: None)


async def is_repo(path: Path | str = ".") -> Result[bool, GitError]:
    target = Path(path).expanduser()
    if not target.exists():
        return Err(GitError("Path does not exist", context={"path": str(target)}))
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(target),
            "rev-parse",
            "--is-inside-work-tree",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return Err(GitError("git executable not found on PATH", context={"path": str(target)}))
    except Exception as exc:
        return Err(
            GitError("Failed to check repository", context={"path": str(target), "error": str(exc)})
        )

    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = (
            stderr.decode("utf-8", errors="replace").strip()
            or stdout.decode("utf-8", errors="replace").strip()
        )
        return Err(GitError(message or "Not a git repository", context={"path": str(target)}))
    return Ok(stdout.decode("utf-8", errors="replace").strip() == "true")


async def iter_git_repos(root: Path, max_depth: int = 2) -> Result[list[Path], GitError]:
    """Return git working directories under root up to the requested depth."""
    root = root.expanduser()
    if not root.exists():
        return Err(GitError("Root path does not exist", context={"root": str(root)}))

    def _scan() -> list[Path]:
        seen: set[Path] = set()
        repos: list[Path] = []
        for git_dir in root.rglob(".git"):
            repo_root = git_dir.parent
            depth = len(repo_root.relative_to(root).parts)
            if depth > max_depth:
                continue
            if repo_root in seen:
                continue
            seen.add(repo_root)
            repos.append(repo_root)
        return repos

    try:
        repos = await asyncio.to_thread(_scan)
    except OSError as exc:
        return Err(
            GitError(
                "Failed to scan for git repositories",
                context={"root": str(root), "error": str(exc)},
            )
        )

    return Ok(repos)
