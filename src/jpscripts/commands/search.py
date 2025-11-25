from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from jpscripts.core import search as search_core
from jpscripts.core.console import console


def _run_interactive(cmd: list[str], prompt: str) -> None:
    """Run rg piped into fzf."""
    if not shutil.which("fzf"):
        # Fallback if fzf is missing but logic demanded interactive
        # In theory, we shouldn't get here if we checked before, but safe to fallback
        proc = subprocess.run(cmd, capture_output=True, text=True)
        console.print(proc.stdout or "[yellow]No matches.[/yellow]")
        return

    # Popen logic for streaming to fzf
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
        # UI Logic: Construct command for piping
        cmd = search_core.get_ripgrep_cmd(pattern, path, context=context)
        _run_interactive(cmd, prompt="ripper> ")
    else:
        # Core Logic: Run and print
        try:
            result = search_core.run_ripgrep(pattern, path, context=context)
            console.print(result or "[yellow]No matches.[/yellow]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)


def todo_scan(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to scan."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    types: str = typer.Option("TODO|FIXME|HACK|BUG", "--types", help="Patterns to search for."),
) -> None:
    """Scan for TODO/FIXME/HACK/BUG markers."""
    use_fzf = shutil.which("fzf") and not no_fzf

    if use_fzf:
        cmd = search_core.get_ripgrep_cmd(types, path, line_number=True)
        _run_interactive(cmd, prompt="todo> ")
    else:
        try:
            result = search_core.run_ripgrep(types, path, line_number=True)
            console.print(result or "[green]No markers found.[/green]")
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)


def loggrep(
    pattern: str = typer.Argument(..., help="Pattern to search for."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new matches (rg --follow --pcre2)."),
) -> None:
    """Friendly log search with optional follow mode."""
    use_fzf = shutil.which("fzf") and not no_fzf

    # Loggrep specific logic: enable PCRE2 and Line Numbers by default
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
