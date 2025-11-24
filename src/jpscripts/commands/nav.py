from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console

IGNORE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}


@dataclass
class RecentEntry:
    path: Path
    mtime: float
    is_dir: bool


def _human_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _scan_recent(root: Path, max_depth: int, include_dirs: bool) -> list[RecentEntry]:
    entries: list[RecentEntry] = []
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.name in IGNORE_DIRS:
                        continue
                    try:
                        is_dir = entry.is_dir(follow_symlinks=False)
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                    except OSError:
                        continue

                    if include_dirs or not is_dir:
                        entries.append(RecentEntry(path=Path(entry.path), mtime=mtime, is_dir=is_dir))

                    if is_dir and depth < max_depth:
                        stack.append((Path(entry.path), depth + 1))
        except OSError:
            continue

    return sorted(entries, key=lambda e: e.mtime, reverse=True)


def _run_fzf(lines: list[str], prompt: str) -> str | None:
    proc = subprocess.run(
        ["fzf", "--no-sort", "--prompt", prompt],
        input="\n".join(lines),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


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
    include_dirs: bool = typer.Option(True, "--include-dirs/--files-only", help="Include directories in the results."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-jump to recently modified files or directories."""
    state = ctx.obj
    base_root = root or state.config.workspace_root or Path.cwd()
    base_root = base_root.expanduser()

    if not base_root.exists():
        console.print(f"[red]Root {base_root} does not exist.[/red]")
        raise typer.Exit(code=1)

    entries = _scan_recent(base_root, max_depth=max_depth, include_dirs=include_dirs)[:limit]
    if not entries:
        console.print(f"[yellow]No recent files found under {base_root}.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    lines = [str(entry.path) for entry in entries]

    if use_fzf:
        selection = _run_fzf(lines, prompt="recent> ")
        if selection:
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
    zoxide = shutil.which("zoxide")
    if not zoxide:
        console.print("[red]zoxide is required for jp proj. Install with `brew install zoxide`.[/red]")
        raise typer.Exit(code=1)

    use_fzf = shutil.which("fzf") and not no_fzf

    try:
        proc = subprocess.run(
            [zoxide, "query", "-l"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]zoxide query failed: {exc.stderr or exc}[/red]")
        raise typer.Exit(code=1)

    paths = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not paths:
        console.print("[yellow]No zoxide entries found.[/yellow]")
        return

    selection: str | None = None
    if use_fzf:
        selection = _run_fzf(paths, prompt="proj> ")
    else:
        table = Table(title="Projects (zoxide)", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Path", style="white")
        for idx, path in enumerate(paths, start=1):
            table.add_row(str(idx), path)
        console.print(table)
        console.print(Panel("fzf not available; re-run with fzf for interactive selection.", style="yellow"))
        return

    if selection:
        typer.echo(selection)
