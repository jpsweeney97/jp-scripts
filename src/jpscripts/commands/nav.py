"""Directory navigation and bookmark commands.

Provides CLI commands for quick directory navigation:
    - Fuzzy directory search
    - Bookmark management (add, list, remove)
    - Recent directory history
    - Project root detection
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.commands.ui import fzf_select_async

# Import core logic
from jpscripts.core import nav as nav_core
from jpscripts.core.console import console
from jpscripts.core.result import Err, NavigationError, Ok, Result


def _human_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _fzf_select(
    lines: list[str], prompt: str, extra_args: list[str] | None = None
) -> str | list[str] | None:
    """Run fzf selection without blocking the main thread."""
    return asyncio.run(fzf_select_async(lines, prompt=prompt, extra_args=extra_args))


def recent(
    ctx: typer.Context,
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Root directory to scan (defaults to workspace_root or current directory).",
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum number of entries to consider."),
    max_depth: int = typer.Option(4, "--max-depth", help="Maximum depth to traverse."),
    include_dirs: bool = typer.Option(
        True, "--include-dirs", help="Include directories in the results."
    ),
    files_only: bool = typer.Option(
        False, "--files-only", help="Only include files (no directories)."
    ),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-jump to recently modified files or directories."""
    state = ctx.obj
    base_root = root or state.config.workspace_root or Path.cwd()
    base_root = base_root.expanduser()

    if not base_root.exists():
        console.print(f"[red]Root {base_root} does not exist.[/red]")
        raise typer.Exit(code=1)

    include_dirs = include_dirs and not files_only
    state = ctx.obj
    ignore_dirs = set(state.config.ignore_dirs)

    async def run_scan() -> Result[list[nav_core.RecentEntry], NavigationError]:
        with console.status(f"Scanning {base_root}...", spinner="dots"):
            return await nav_core.scan_recent(
                base_root, max_depth=max_depth, include_dirs=include_dirs, ignore_dirs=ignore_dirs
            )

    # LOGIC: Delegate to async core
    match asyncio.run(run_scan()):
        case Err(err):
            console.print(f"[red]Error: {err.message}[/red]")
            raise typer.Exit(code=1)
        case Ok(entries):
            entries = entries[:limit]

    if not entries:
        console.print(f"[yellow]No recent files found under {base_root}.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    lines = [str(entry.path) for entry in entries]

    if use_fzf:
        selection = _fzf_select(lines, prompt="recent> ", extra_args=["--no-sort"])
        if isinstance(selection, str) and selection:
            typer.echo(selection)
        return

    table = Table(title=f"Recent items in {base_root}", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("When", style="cyan", no_wrap=True)
    table.add_column("Type", style="white", no_wrap=True)
    table.add_column("Path", style="white")

    for entry in entries:
        table.add_row(
            _human_time(entry.mtime),
            "dir" if entry.is_dir else "file",
            str(entry.path),
        )

    console.print(table)


def proj(
    _ctx: typer.Context,
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-pick a project using zoxide + fzf and print the path."""

    async def run_query() -> Result[list[str], NavigationError]:
        return await nav_core.get_zoxide_projects()

    match asyncio.run(run_query()):
        case Err(err):
            console.print(f"[red]{err.message}[/red]")
            if "not found" in err.message:
                console.print("[red]Install with `brew install zoxide`.[/red]")
            raise typer.Exit(code=1)
        case Ok(paths):
            pass

    if not paths:
        console.print("[yellow]No zoxide entries found.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection: str | None = None

    if use_fzf:
        fzf_selection = _fzf_select(paths, prompt="proj> ", extra_args=["--no-sort"])
        selection = fzf_selection if isinstance(fzf_selection, str) else None
    else:
        table = Table(title="Projects (zoxide)", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Path", style="white")
        for idx, path in enumerate(paths, start=1):
            table.add_row(str(idx), path)
        console.print(table)
        console.print(
            Panel("fzf not available; re-run with fzf for interactive selection.", style="yellow")
        )
        return

    if selection:
        typer.echo(selection)
