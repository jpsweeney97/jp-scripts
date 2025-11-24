from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from jpscripts.core.console import console


def _ensure_tool(binary: str, friendly: str) -> None:
    if not shutil.which(binary):
        console.print(f"[red]{friendly} is required ({binary} not found).[/red]")
        raise typer.Exit(code=1)


def _run_rg(args: list[str], use_fzf: bool, prompt: str) -> None:
    proc_rg = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    if use_fzf:
        subprocess.run(["fzf", "--ansi", "--prompt", prompt], stdin=proc_rg.stdout)
    else:
        out, _ = proc_rg.communicate()
        console.print(out or "[yellow]No matches.[/yellow]")


def ripper(
    pattern: str = typer.Argument(..., help="Search pattern for ripgrep."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Root path to search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    context: int = typer.Option(2, "--context", "-C", help="Lines of context to include."),
) -> None:
    """Interactive code search using ripgrep + fzf."""
    _ensure_tool("rg", "ripgrep")
    use_fzf = shutil.which("fzf") and not no_fzf
    args = ["rg", "--color=always", f"-C{context}", pattern, str(path)]
    _run_rg(args, use_fzf, prompt="ripper> ")


def todo_scan(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to scan."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    types: str = typer.Option("TODO|FIXME|HACK|BUG", "--types", help="Patterns to search for."),
) -> None:
    """Scan for TODO/FIXME/HACK/BUG markers."""
    _ensure_tool("rg", "ripgrep")
    use_fzf = shutil.which("fzf") and not no_fzf
    args = ["rg", "--color=always", "--line-number", types, str(path)]
    _run_rg(args, use_fzf, prompt="todo> ")


def loggrep(
    pattern: str = typer.Argument(..., help="Pattern to search for."),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new matches (rg --follow --pcre2)."),
) -> None:
    """Friendly log search with optional follow mode."""
    _ensure_tool("rg", "ripgrep")
    use_fzf = shutil.which("fzf") and not no_fzf
    args = ["rg", "--color=always", "--line-number"]
    if follow:
        args += ["--follow", "--pcre2"]
    args += [pattern, str(path)]
    _run_rg(args, use_fzf, prompt="loggrep> ")
