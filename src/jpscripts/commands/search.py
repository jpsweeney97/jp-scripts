from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer
from rich import box
from rich.table import Table
from rich.panel import Panel

from jpscripts.core import search as search_core
from jpscripts.core.console import console


def _run_interactive(cmd: list[str], prompt: str) -> None:
    """Run rg piped into fzf."""
    if not shutil.which("fzf"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        console.print(proc.stdout or "[yellow]No matches.[/yellow]")
        return

    proc_rg = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    subprocess.run(["fzf", "--ansi", "--prompt", prompt], stdin=proc_rg.stdout)


def ripper(
    pattern: str = typer.Argument(..., help="Search pattern for ripgrep."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Root path to search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    context: int = typer.Option(2, "--context", "-C", help="Lines of context to include."),
) -> None:
    """Interactive code search using ripgrep + fzf."""
    use_fzf = shutil.which("fzf") and not no_fzf

    if use_fzf:
        cmd = search_core.get_ripgrep_cmd(pattern, path, context=context)
        _run_interactive(cmd, prompt="ripper> ")
    else:
        try:
            result = search_core.run_ripgrep(pattern, path, context=context)
            console.print(result or "[yellow]No matches.[/yellow]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)


_BAT_WARNED = False

def _print_with_bat(content: str, language: str = "md") -> None:
    """Safely try to use bat for syntax highlighting, fallback to rich."""
    global _BAT_WARNED

    if shutil.which("bat"):
        try:
            subprocess.run(
                ["bat", "--language", language, "--style", "plain", "--paging", "never"],
                input=content,
                text=True,
                check=False
            )
            return
        except Exception:
            pass # Fallback on error

    # Fallback
    console.print(content)
    if not shutil.which("bat") and not _BAT_WARNED:
        console.print("[dim]Tip: Install `bat` for syntax highlighting.[/dim]")
        _BAT_WARNED = True


def todo_scan(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to scan."),
    types: str = typer.Option("TODO|FIXME|HACK|BUG", "--types", help="Patterns to search for."),
) -> None:
    """Scan for TODO items and display a structured table."""

    try:
        todos = search_core.scan_todos(path, types=types)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if not todos:
        console.print("[green]No TODOs found.[/green]")
        return

    table = Table(title=f"Found {len(todos)} Items", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("File", style="white")
    table.add_column("Line", style="dim")
    table.add_column("Content", style="white")

    for todo in todos:
        # Simple truncation for display
        content = todo.text.replace(todo.type, "").strip()
        content = (content[:75] + "...") if len(content) > 75 else content

        style = "red" if "FIXME" in todo.type or "BUG" in todo.type else "yellow"
        table.add_row(f"[{style}]{todo.type}[/]", todo.file, str(todo.line), content)

    console.print(table)

def loggrep(
    pattern: str = typer.Argument(..., help="Pattern to search for."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new matches (rg --follow --pcre2)."),
) -> None:
    """Friendly log search with optional follow mode."""
    use_fzf = shutil.which("fzf") and not no_fzf

    if use_fzf:
        cmd = search_core.get_ripgrep_cmd(pattern, path, line_number=True, follow=follow, pcre2=True)
        _run_interactive(cmd, prompt="loggrep> ")
    else:
        try:
            result = search_core.run_ripgrep(pattern, path, line_number=True, follow=follow, pcre2=True)
            console.print(result or "[yellow]No matches.[/yellow]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)
