from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core import git as git_core


@dataclass
class _StatusContext:
    root: Path
    total: int


async def _describe_repo(path: Path) -> git_core.BranchStatus:
    try:
        repo = await asyncio.to_thread(git_core.open_repo, path)
        return await asyncio.to_thread(git_core.describe_status, repo)
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

    try:
        repo = git_core.open_repo(repo_path)
    except Exception as exc:
        console.print(f"[red]Failed to open git repo at {repo_path}: {exc}[/red]")
        raise typer.Exit(code=1)

    status = git_core.describe_status(repo)
    upstream = status.upstream
    if not upstream:
        console.print("[yellow]No upstream branch is configured. Set one with[/yellow] git push --set-upstream origin <branch>")
        raise typer.Exit(code=1)

    commits = list(repo.iter_commits(f"{upstream}..HEAD"))[:max_commits]
    diffstat = repo.git.diff("--stat", f"{upstream}..HEAD") if commits else ""

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
            commit.author.name,
            datetime.fromtimestamp(commit.committed_date).strftime("%Y-%m-%d %H:%M"),
        )

    console.print(commits_table)

    if diffstat.strip():
        console.print(Panel(diffstat, title="Diffstat", box=box.SIMPLE))
