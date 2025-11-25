from __future__ import annotations

import shutil
import subprocess
import sys

import typer
from rich import box
from rich.panel import Panel

from jpscripts.core.console import console
from jpscripts.core.nav import scan_recent

def _ensure_codex() -> str:
    binary = shutil.which("codex")
    if not binary:
        console.print("[red]Codex CLI not found. Install it via `npm install -g @openai/codex` or `brew install codex`.[/red]")
        raise typer.Exit(code=1)
    return binary

def codex_exec(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Instruction for Codex."),
    attach_recent: bool = typer.Option(False, "--recent", "-r", help="Attach top 5 recently modified files to context."),
    full_auto: bool = typer.Option(False, "--full-auto", "-y", help="Run without asking for confirmation (dangerous)."),
    model: str = typer.Option("gpt-5.1-codex-max", "--model", "-m", help="Model to use."),
) -> None:
    """Delegate a task to the Codex agent."""
    codex_bin = _ensure_codex()

    # 1. Build the command
    cmd = [codex_bin, "exec", prompt, "--model", model]

    if full_auto:
        cmd.append("--full-auto")

    # 2. Context Injection (The God-Mode Feature)
    # We use our own core logic to find what's relevant, so we don't rely on Codex guessing.
    if attach_recent:
        state = ctx.obj
        root = state.config.workspace_root or state.config.notes_dir
        # Scan for recent files using our pure core function
        recents = scan_recent(
            root.expanduser(),
            max_depth=3,
            include_dirs=False,
            ignore_dirs=set(state.config.ignore_dirs)
        )

        # Attach top 5 files
        for entry in recents[:5]:
            cmd.extend(["--file", str(entry.path)])
            console.print(f"[dim]Attaching context: {entry.path.name}[/dim]")

    # 3. Handoff
    console.print(Panel(f"Handing off to [bold magenta]Codex[/bold magenta]...", box=box.SIMPLE))

    try:
        # We use shell=False for safety, and allow it to take over stdout/stderr
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        console.print("[yellow]Codex session cancelled.[/yellow]")
