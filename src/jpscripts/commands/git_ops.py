from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import typer
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core import git as git_core
from jpscripts.core.decorators import handle_exceptions


@dataclass
class _StatusContext:
    root: Path
    total: int


async def _describe_repo(path: Path) -> git_core.BranchStatus:
    try:
        repo = await git_core.AsyncRepo.open(path)
        return await repo.status()
    except Exception as exc:  # Git errors are reported per-repo
        return git_core.BranchStatus(
            path=path,
            branch="(error)",
            upstream=None,
            ahead=0,
            behind=0,
            staged=0,
            unstaged=0,
            untracked=0,
            dirty=False,
            error=str(exc),
        )


def _render_status_table(ctx: _StatusContext, statuses: list[git_core.BranchStatus]) -> Table:
    table = Table(
        title=f"Git status in {ctx.root} ({len(statuses)}/{ctx.total})",
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    table.add_column("Repo", style="cyan", no_wrap=True)
    table.add_column("Branch", style="white", no_wrap=True)
    table.add_column("Dirty", style="white", no_wrap=True)
    table.add_column("Upstream", style="white", no_wrap=True)
    table.add_column("Ahead/Behind", style="white", no_wrap=True)
    table.add_column("Changes", style="white")

    for status in sorted(statuses, key=lambda s: s.path.name.lower()):
        repo_name = status.path.name
        branch = status.branch
        dirty = "[yellow]dirty[/]" if status.dirty else "[green]clean[/]"
        upstream = status.upstream or "none"
        ahead_behind = f"{status.ahead}/{status.behind}"
        changes = f"{status.staged} staged, {status.unstaged} unstaged, {status.untracked} untracked"

        row_style = "red" if status.error else None
        branch_display = branch if not status.error else f"{branch} ({status.error})"
        table.add_row(repo_name, branch_display, dirty, upstream, ahead_behind, changes, style=row_style)

    return table


async def _collect_statuses(repo_paths: list[Path], root: Path) -> list[git_core.BranchStatus]:
    ctx = _StatusContext(root=root, total=len(repo_paths))
    results: list[git_core.BranchStatus] = []

    with Live(_render_status_table(ctx, results), console=console, refresh_per_second=4) as live:
        async def worker(path: Path) -> None:
            status = await _describe_repo(path)
            results.append(status)
            live.update(_render_status_table(ctx, results))

        tasks = [asyncio.create_task(worker(path)) for path in repo_paths]
        await asyncio.gather(*tasks)

    return results


def status_all(
    ctx: typer.Context,
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Root directory to scan for git repositories (defaults to worktree_root or workspace_root).",
    ),
    max_depth: int = typer.Option(2, "--max-depth", help="Maximum depth to search for repositories."),
) -> None:
    """Summarize git status across repositories with a live-updating table."""
    state = ctx.obj
    base_root = root or state.config.worktree_root or state.config.workspace_root
    base_root = base_root.expanduser()

    if not base_root.exists():
        console.print(f"[red]Root {base_root} does not exist.[/red]")
        raise typer.Exit(code=1)

    repo_paths = list(git_core.iter_git_repos(base_root, max_depth=max_depth))
    if not repo_paths:
        console.print(f"[yellow]No git repositories found under {base_root} (max depth {max_depth}).[/yellow]")
        return

    state.logger.debug("Scanning %s repositories under %s", len(repo_paths), base_root)
    asyncio.run(_collect_statuses(repo_paths, base_root))


@handle_exceptions
def whatpush(
    ctx: typer.Context,
    repo_path: Path = typer.Option(
        Path("."),
        "--repo",
        "-r",
        help="Path to a git repository (defaults to current directory).",
    ),
    max_commits: int = typer.Option(50, help="Maximum number of commits to display."),
) -> None:
    """Show what will be pushed to the upstream branch."""
    repo_path = repo_path.expanduser()

    async def _collect() -> tuple[git_core.BranchStatus, list[git_core.GitCommit], str]:
        repo = await git_core.AsyncRepo.open(repo_path)
        status = await repo.status()
        upstream = status.upstream
        if not upstream:
            raise git_core.GitOperationError(
                "No upstream branch is configured. Set one with git push --set-upstream origin <branch>"
            )
        commits = await repo.get_commits(f"{upstream}..HEAD", max_commits)
        diffstat = await repo.diff_stat(f"{upstream}..HEAD") if commits else ""
        return status, commits, diffstat

    status, commits, diffstat = asyncio.run(_collect())

    upstream = status.upstream or "none"

    summary = Table(title="Push summary", box=box.SIMPLE)
    summary.add_column("Field", style="cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("Repository", status.path.as_posix())
    summary.add_row("Branch", status.branch)
    summary.add_row("Upstream", upstream)
    summary.add_row("Ahead/Behind", f"{status.ahead}/{status.behind}")
    summary.add_row("Dirty", "yes" if status.dirty else "no")
    console.print(summary)

    if status.behind:
        console.print(Panel.fit(f"Behind upstream by {status.behind} commits. Pull or rebase recommended.", style="yellow"))

    if not commits:
        console.print("[green]Nothing to push.[/green]")
        return

    commits_table = Table(title=f"Commits to push (showing up to {max_commits})", box=box.SIMPLE_HEAVY, expand=True)
    commits_table.add_column("SHA", style="cyan", no_wrap=True)
    commits_table.add_column("Summary", style="white")
    commits_table.add_column("Author", style="white", no_wrap=True)
    commits_table.add_column("Date", style="white", no_wrap=True)

    for commit in commits:
        commits_table.add_row(
            commit.hexsha[:8],
            commit.summary,
            commit.author_name,
            commit.committed_datetime.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(commits_table)

    if diffstat.strip():
        console.print(Panel(diffstat, title="Diffstat", box=box.SIMPLE))

async def _fetch_repo(path: Path) -> str:
    """Run git fetch on all remotes and return a status string."""
    try:
        has_remote = await asyncio.to_thread(_has_remotes, path)
        if not has_remote:
            return "[green]fetched (no remotes)[/]"
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(path),
            "fetch",
            "--all",
            "--prune",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return "[red]fetched (timeout)[/]"

        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
            return f"[red]failed: {message or 'git fetch failed'}[/]"

        return "[green]fetched[/]"
    except Exception as exc:
        return f"[red]error: {exc}[/]"


def _resolve_git_dir(path: Path) -> Path:
    git_dir = path / ".git"
    if git_dir.is_file():
        try:
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                target = content.partition(":")[2].strip()
                if target:
                    return (path / target).resolve()
        except OSError:
            return git_dir
    return git_dir


def _has_remotes(path: Path) -> bool:
    config_path = _resolve_git_dir(path) / "config"
    try:
        data = config_path.read_text(encoding="utf-8")
    except OSError:
        return True
    return "[remote " in data

def sync(
    ctx: typer.Context,
    root: Path | None = typer.Option(None, "--root", "-r"),
    max_depth: int = typer.Option(2, "--max-depth"),
) -> None:
    """Parallel git fetch across all repositories."""
    state = ctx.obj
    base_root = root or state.config.worktree_root or state.config.workspace_root
    base_root = base_root.expanduser()

    repo_paths = list(git_core.iter_git_repos(base_root, max_depth=max_depth))

    async def runner() -> list[tuple[Path, str]]:
        sem = asyncio.Semaphore(10)
        results: list[tuple[Path, str]] = []

        async def bounded_fetch(path: Path) -> None:
            async with sem:
                try:
                    res = await asyncio.wait_for(_fetch_repo(path), timeout=10)
                except asyncio.TimeoutError:
                    res = "[red]fetched (timeout)[/]"
                results.append((path, res))

        tasks = [bounded_fetch(p) for p in repo_paths]
        await asyncio.gather(*tasks)
        return results

    results = asyncio.run(runner())

    summary = Table(title=f"Syncing {len(results)} Repositories", box=box.SIMPLE)
    summary.add_column("Repo", style="cyan")
    summary.add_column("Status", style="white")
    for path, status in results:
        summary.add_row(path.name, status)
    console.print(summary)
