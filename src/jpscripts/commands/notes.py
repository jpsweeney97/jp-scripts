"""Note-taking and clipboard history commands.

Provides CLI commands for:
    - Creating and searching notes
    - Generating standup summaries from git commits
    - Clipboard history management
    - Editor integration for note editing
"""

from __future__ import annotations

import asyncio
import datetime as dt
import shlex
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pyperclip
import typer
from rich import box
from rich.table import Table

from jpscripts.commands.ui import fzf_select_async, fzf_stream_with_command
from jpscripts.core import notes_impl
from jpscripts.core import search as search_core
from jpscripts.core.console import console
from jpscripts.core.result import Err, Ok
from jpscripts.git import client as git_core

CLIPHIST_DIR = Path.home() / ".local" / "share" / "jpscripts" / "cliphist"
CLIPHIST_FILE = CLIPHIST_DIR / "history.txt"
CLIPHIST_DB = CLIPHIST_DIR / "history.db"


def note(
    ctx: typer.Context,
    message: str = typer.Option(
        "", "--message", "-m", help="Message to append. If empty, opens editor."
    ),
) -> None:
    """Append to today's note or open it in the configured editor."""
    state = ctx.obj
    notes_dir = state.config.user.notes_dir.expanduser()

    # Use core logic to get the path
    notes_impl.ensure_notes_dir(notes_dir)
    note_path = notes_impl.get_today_path(notes_dir)

    async def _run() -> None:
        if message:
            # Use core logic to write
            await notes_impl.append_to_daily_note(notes_dir, message)
            console.print(f"[green]Appended to[/green] {note_path}")
            return

        editor_cmd = shlex.split(state.config.user.editor)
        try:
            exit_code = await _launch_editor(editor_cmd, note_path)
            if exit_code != 0:
                console.print(f"[red]Editor exited with code {exit_code}[/red]")
        except FileNotFoundError:
            console.print(f"[red]Editor not found:[/red] {state.config.user.editor}")
            raise typer.Exit(code=1)

    asyncio.run(_run())


def note_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search pattern for ripgrep."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Search notes with ripgrep and optionally fzf."""
    state = ctx.obj
    notes_dir = state.config.user.notes_dir.expanduser()
    if not notes_dir.exists():
        console.print(f"[yellow]Notes directory {notes_dir} does not exist.[/yellow]")
        raise typer.Exit(code=1)

    async def _run() -> None:
        use_fzf = shutil.which("fzf") and not no_fzf

        if use_fzf:
            cmd = search_core.get_ripgrep_cmd(query, notes_dir, line_number=True)
            try:
                selection = await fzf_stream_with_command(
                    cmd,
                    prompt="note-search> ",
                    ansi=True,
                    extra_args=["--delimiter", ":", "--nth", "3.."],
                )
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(code=1)

            if selection:
                if isinstance(selection, list):
                    for line in selection:
                        console.print(line)
                else:
                    console.print(selection)
            return

        try:
            result = await asyncio.to_thread(
                search_core.run_ripgrep, query, notes_dir, line_number=True
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

        if result:
            console.print(result)
        else:
            console.print("[yellow]No matches found.[/yellow]")

    asyncio.run(_run())


@dataclass
class RepoSummary:
    path: Path
    commits: list[git_core.GitCommit]
    error: str | None = None


async def _collect_repo_commits(
    repo_path: Path,
    since: dt.datetime,
    author_email: str | None,
    limit: int,
) -> RepoSummary:
    cutoff = int(since.timestamp())
    match await git_core.AsyncRepo.open(repo_path):
        case Err(err):
            return RepoSummary(path=repo_path, commits=[], error=err.message)
        case Ok(repo):
            match await repo.get_commits("HEAD", limit):
                case Err(err):
                    return RepoSummary(path=repo_path, commits=[], error=err.message)
                case Ok(commits):
                    filtered = [
                        commit
                        for commit in commits
                        if commit.committed_date >= cutoff
                        and (author_email is None or commit.author_email == author_email)
                    ]
                    return RepoSummary(path=repo_path, commits=filtered)


async def _detect_user_email(root: Path) -> str | None:
    match await git_core.AsyncRepo.open(root):
        case Err(_):
            return None
        case Ok(repo):
            match await repo.run_git("config", "user.email"):
                case Err(_):
                    return None
                case Ok(email_raw):
                    email = email_raw.strip()
                    return email or None


def standup(
    ctx: typer.Context,
    days: int = typer.Option(3, "--days", "-d", help="Look back this many days."),
    max_depth: int = typer.Option(2, "--max-depth", help="Max depth when scanning repos."),
) -> None:
    """Summarize recent commits across repos."""
    state = ctx.obj
    root = state.config.infra.worktree_root or state.config.user.workspace_root
    root = root.expanduser()
    since = dt.datetime.now() - dt.timedelta(days=days)
    commit_limit = min(1000, max(100, days * 100))

    async def _run() -> None:
        match await git_core.iter_git_repos(root, max_depth=max_depth):
            case Err(err):
                console.print(f"[red]Error scanning git repositories: {err.message}[/red]")
                raise typer.Exit(code=1)
            case Ok(repos):
                pass

        if not repos:
            console.print(f"[yellow]No git repositories found under {root}.[/yellow]")
            return

        user_email = await _detect_user_email(root)
        summaries = await asyncio.gather(
            *(
                _collect_repo_commits(repo_path, since, user_email, commit_limit)
                for repo_path in repos
            )
        )

        table = Table(title=f"Commits since {since.date()}", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Repo", style="cyan", no_wrap=True)
        table.add_column("Count", style="white", no_wrap=True)
        table.add_column("Latest", style="white")

        total = 0
        for summary in summaries:
            repo_label = summary.path.name or summary.path.as_posix()
            if repo_label in {"", "."}:
                resolved = summary.path.resolve()
                repo_label = resolved.name or resolved.as_posix()
            if summary.error:
                table.add_row(repo_label, "0", f"error: {summary.error}", style="red")
                continue
            if not summary.commits:
                continue
            latest = summary.commits[0]
            table.add_row(repo_label, str(len(summary.commits)), latest.summary)
            total += len(summary.commits)

        if total == 0:
            console.print(f"[yellow]No commits in the last {days} days.[/yellow]")
        else:
            console.print(table)

    asyncio.run(_run())


def standup_note(
    ctx: typer.Context,
    days: int = typer.Option(3, "--days", "-d", help="Look back this many days."),
) -> None:
    """Run standup and append its output to today's note."""
    state = ctx.obj
    notes_dir = state.config.user.notes_dir.expanduser()

    notes_impl.ensure_notes_dir(notes_dir)
    note_path = notes_impl.get_today_path(notes_dir)

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
        lines = [
            line for line in CLIPHIST_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except OSError:
        return

    records: list[tuple[str, str]] = []
    for line in lines:
        when, _, text = line.partition("\t")
        if text:
            records.append((when, text))

    if records:
        conn.executemany("INSERT INTO history (timestamp, content) VALUES (?, ?)", records)
        console.print(
            f"[green]Imported {len(records)} clipboard entries from legacy history.[/green]"
        )


def cliphist(
    ctx: typer.Context,
    action: str = typer.Option(
        "add", "--action", "-a", help="add (save current clipboard), pick (fzf select), show"
    ),
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
                    selection = asyncio.run(
                        fzf_select_async(lines, prompt="clip> ", extra_args=["--with-nth", "2.."])
                    )
                else:
                    selection = lines[0]

                if selection and isinstance(selection, str):
                    _, _, text = selection.partition("\t")
                    pyperclip.copy(text)
                    console.print("[green]Copied selection to clipboard.[/green]")
                return
    except sqlite3.Error as exc:
        console.print(f"[red]Clipboard history error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print("[red]Unknown action. Use add, pick, or show.[/red]")


async def _launch_editor(editor_cmd: list[str], note_path: Path) -> int:
    """Launch the configured editor asynchronously."""
    proc = await asyncio.create_subprocess_exec(*editor_cmd, str(note_path))
    return await proc.wait()
