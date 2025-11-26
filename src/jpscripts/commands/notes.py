from __future__ import annotations

import datetime as dt
import shlex
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pyperclip  # type: ignore[import-untyped]
from git import Repo
import typer
from rich import box
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core import git as git_core
from jpscripts.core import notes_impl

CLIPHIST_DIR = Path.home() / ".local" / "share" / "jpscripts" / "cliphist"
CLIPHIST_FILE = CLIPHIST_DIR / "history.txt"
CLIPHIST_DB = CLIPHIST_DIR / "history.db"

def note(
    ctx: typer.Context,
    message: str = typer.Option("", "--message", "-m", help="Message to append. If empty, opens editor."),
) -> None:
    """Append to today's note or open it in the configured editor."""
    state = ctx.obj
    notes_dir = state.config.notes_dir.expanduser()

    # Use core logic to get the path
    notes_impl.ensure_notes_dir(notes_dir)
    note_path = notes_impl.get_today_path(notes_dir)

    if message:
        # Use core logic to write
        notes_impl.append_to_daily_note(notes_dir, message)
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


def _repo_commits_since(repo: Repo, since: dt.datetime, author: str | None) -> list:
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

    user_email: str | None = None
    try:
        import git

        repo = git.Repo(root, search_parent_directories=True)
        with repo.config_reader() as cfg:
            try:
                value = cfg.get_value("user", "email")
            except Exception:
                value = None
            if isinstance(value, str):
                user_email = value
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
            repo = Repo(repo_path)
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

    # FIX: Use the core module instead of the deleted local helpers
    notes_impl.ensure_notes_dir(notes_dir)     # Was _ensure_notes_dir(notes_dir)
    note_path = notes_impl.get_today_path(notes_dir) # Was _today_path(notes_dir)

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

def _init_db() -> sqlite3.Connection:
    """Initialize the clipboard history database and return a connection."""
    CLIPHIST_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CLIPHIST_DB)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """
    )
    _migrate_legacy_history(conn)
    return conn


def _migrate_legacy_history(conn: sqlite3.Connection) -> None:
    """One-time import from the legacy text history if present."""
    if not CLIPHIST_FILE.exists():
        return

    try:
        has_rows = conn.execute("SELECT 1 FROM history LIMIT 1").fetchone()
    except sqlite3.Error:
        return

    if has_rows:
        return

    try:
        lines = [line for line in CLIPHIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return

    records: list[tuple[str, str]] = []
    for line in lines:
        when, _, text = line.partition("\t")
        if text:
            records.append((when, text))

    if records:
        conn.executemany("INSERT INTO history (timestamp, content) VALUES (?, ?)", records)
        console.print(f"[green]Imported {len(records)} clipboard entries from legacy history.[/green]")


def cliphist(
    ctx: typer.Context,
    action: str = typer.Option("add", "--action", "-a", help="add (save current clipboard), pick (fzf select), show"),
    limit: int = typer.Option(50, "--limit", "-l", help="Max entries to show when picking."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Simple clipboard history backed by SQLite."""
    try:
        with _init_db() as conn:
            use_fzf = shutil.which("fzf") and not no_fzf

            if action == "add":
                content = pyperclip.paste()
                if not content:
                    console.print("[yellow]Clipboard is empty.[/yellow]")
                    return
                timestamp = dt.datetime.now().isoformat(timespec="seconds")
                conn.execute(
                    "INSERT INTO history (timestamp, content) VALUES (?, ?)",
                    (timestamp, content),
                )
                console.print("[green]Saved clipboard entry.[/green]")
                return

            cursor = conn.execute(
                "SELECT timestamp, content FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            entries = cursor.fetchall()

            if not entries:
                console.print("[yellow]No clipboard history yet.[/yellow]")
                return

            if action == "show":
                table = Table(title="Clipboard history", box=box.SIMPLE_HEAVY, expand=True)
                table.add_column("When", style="cyan", no_wrap=True)
                table.add_column("Text", style="white")
                for when, text in entries:
                    table.add_row(when, text)
                console.print(table)
                return

            if action == "pick":
                lines = [f"{when}\t{text}" for when, text in entries]
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
    except sqlite3.Error as exc:
        console.print(f"[red]Clipboard history error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print("[red]Unknown action. Use add, pick, or show.[/red]")
