from __future__ import annotations

import datetime as dt
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pyperclip
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core import git as git_core

CLIPHIST_DIR = Path.home() / ".local" / "share" / "jpscripts" / "cliphist"
CLIPHIST_FILE = CLIPHIST_DIR / "history.txt"


def _ensure_notes_dir(notes_dir: Path) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)


def _today_path(notes_dir: Path) -> Path:
    today = dt.date.today().isoformat()
    return notes_dir / f"{today}.md"


def note(
    ctx: typer.Context,
    message: str = typer.Option("", "--message", "-m", help="Message to append. If empty, opens editor."),
) -> None:
    """Append to today's note or open it in the configured editor."""
    state = ctx.obj
    notes_dir = state.config.notes_dir.expanduser()
    _ensure_notes_dir(notes_dir)
    note_path = _today_path(notes_dir)

    if message:
        timestamp = dt.datetime.now().strftime("%H:%M")
        with note_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] {message}\n")
        console.print(f"[green]Appended to[/green] {note_path}")
        return

    editor_cmd = shlex.split(state.config.editor)
    try:
        subprocess.run([*editor_cmd, str(note_path)], check=False)
    except FileNotFoundError:
        console.print(f"[red]Editor not found:[/red] {state.config.editor}")
        raise typer.Exit(code=1)


def note_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search pattern for ripgrep."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Search notes with ripgrep and optionally fzf."""
    state = ctx.obj
    notes_dir = state.config.notes_dir.expanduser()
    if not notes_dir.exists():
        console.print(f"[yellow]Notes directory {notes_dir} does not exist.[/yellow]")
        raise typer.Exit(code=1)

    if not shutil.which("rg"):
        console.print("[red]ripgrep (rg) is required for note-search.[/red]")
        raise typer.Exit(code=1)

    rg_cmd = ["rg", "--line-number", query, str(notes_dir)]
    use_fzf = shutil.which("fzf") and not no_fzf

    if use_fzf:
        proc_rg = subprocess.Popen(rg_cmd, stdout=subprocess.PIPE)
        proc_fzf = subprocess.run(["fzf", "--delimiter", ":", "--nth", "3.."], stdin=proc_rg.stdout, text=True)
        if proc_fzf.returncode == 0:
            console.print(proc_fzf.stdout.strip())
        return

    proc = subprocess.run(rg_cmd, text=True, capture_output=True)
    if proc.stdout:
        console.print(proc.stdout)
    else:
        console.print("[yellow]No matches found.[/yellow]")


def _repo_commits_since(repo: git_core.Repo, since: dt.datetime, author: str | None) -> list:
    args = {}
    if author:
        args["author"] = author
    commits = list(repo.iter_commits(since=since.isoformat(), **args))
    return commits


def standup(
    ctx: typer.Context,
    days: int = typer.Option(3, "--days", "-d", help="Look back this many days."),
    max_depth: int = typer.Option(2, "--max-depth", help="Max depth when scanning repos."),
) -> None:
    """Summarize recent commits across repos."""
    state = ctx.obj
    root = state.config.worktree_root or state.config.workspace_root
    root = root.expanduser()
    since = dt.datetime.now() - dt.timedelta(days=days)

    user_email = None
    try:
        import git

        repo = git.Repo(root, search_parent_directories=True)
        with repo.config_reader() as cfg:
            user_email = cfg.get_value("user", "email", fallback=None)
    except Exception:
        pass

    repos = list(git_core.iter_git_repos(root, max_depth=max_depth))
    if not repos:
        console.print(f"[yellow]No git repositories found under {root}.[/yellow]")
        return

    table = Table(title=f"Commits since {since.date()}", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Repo", style="cyan", no_wrap=True)
    table.add_column("Count", style="white", no_wrap=True)
    table.add_column("Latest", style="white")

    total = 0
    for repo_path in repos:
        try:
            repo = git_core.open_repo(repo_path)
            commits = _repo_commits_since(repo, since, user_email)
            if not commits:
                continue
            latest = commits[0]
            table.add_row(
                repo_path.name,
                str(len(commits)),
                latest.summary,
            )
            total += len(commits)
        except Exception as exc:
            table.add_row(repo_path.name, "0", f"error: {exc}", style="red")

    if total == 0:
        console.print(f"[yellow]No commits in the last {days} days.[/yellow]")
    else:
        console.print(table)


def standup_note(ctx: typer.Context, days: int = typer.Option(3, "--days", "-d", help="Look back this many days.")) -> None:
    """Run standup and append its output to today's note."""
    state = ctx.obj
    notes_dir = state.config.notes_dir.expanduser()
    _ensure_notes_dir(notes_dir)
    note_path = _today_path(notes_dir)

    with console.capture() as capture:
        standup(ctx, days=days)
    captured = capture.get()

    if not captured.strip():
        console.print("[yellow]No standup output to append.[/yellow]")
        return

    heading = f"## Standup {dt.date.today().isoformat()}\n"
    with note_path.open("a", encoding="utf-8") as f:
        f.write("\n" + heading + captured + "\n")
    console.print(f"[green]Appended standup to[/green] {note_path}")
    console.print(captured)


def cliphist(
    action: str = typer.Option("add", "--action", "-a", help="add (save current clipboard), pick (fzf select), show"),
    limit: int = typer.Option(50, "--limit", "-l", help="Max entries to show when picking."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Simple clipboard history backed by a text file."""
    CLIPHIST_DIR.mkdir(parents=True, exist_ok=True)
    CLIPHIST_FILE.touch(exist_ok=True)

    use_fzf = shutil.which("fzf") and not no_fzf

    if action == "add":
        content = pyperclip.paste()
        if not content:
            console.print("[yellow]Clipboard is empty.[/yellow]")
            return
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        with CLIPHIST_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{content}\n")
        console.print("[green]Saved clipboard entry.[/green]")
        return

    entries = [line.rstrip("\n") for line in CLIPHIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        console.print("[yellow]No clipboard history yet.[/yellow]")
        return

    entries = entries[-limit:]

    if action == "show":
        table = Table(title="Clipboard history", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("When", style="cyan", no_wrap=True)
        table.add_column("Text", style="white")
        for line in reversed(entries):
            when, _, text = line.partition("\t")
            table.add_row(when, text)
        console.print(table)
        return

    if action == "pick":
        lines = [line for line in reversed(entries)]
        selection = None
        if use_fzf:
            selection = subprocess.run(
                ["fzf", "--prompt", "clip> "],
                input="\n".join(lines),
                text=True,
                capture_output=True,
            ).stdout.strip()
        else:
            selection = lines[0]

        if selection:
            _, _, text = selection.partition("\t")
            pyperclip.copy(text)
            console.print("[green]Copied selection to clipboard.[/green]")
        return

    console.print("[red]Unknown action. Use add, pick, or show.[/red]")
