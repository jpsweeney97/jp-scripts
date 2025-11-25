from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

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
    cmd = [codex_bin, "exec", prompt, "--model", model, "--json"]

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
    console.print(Panel("Handing off to [bold magenta]Codex[/bold magenta]...", box=box.SIMPLE))

    assistant_parts: list[str] = []
    status = None

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except KeyboardInterrupt:
        console.print("[yellow]Codex session cancelled.[/yellow]")
        return
    except Exception as exc:
        console.print(f"[red]Failed to start Codex:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        if proc.stdout is None:
            console.print("[red]Codex did not provide output.[/red]")
            return

        status = console.status("Connecting to Codex...", spinner="dots")
        status.start()

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                console.print(f"[red]Failed to parse Codex output:[/red] {line}")
                continue

            data: dict[str, Any] = event.get("data") or {}
            event_type = event.get("event") or event.get("type")

            if event_type == "item.started":
                action = data.get("action") or data.get("command") or data.get("name") or "working"
                status.update(f"[cyan]Running[/cyan] {action}")
            elif event_type == "turn.failed":
                error_msg = data.get("error") or event.get("error") or "Codex execution failed."
                console.print(Panel(f"[red]{error_msg}[/red]", title="Codex error", box=box.SIMPLE))
            elif event_type == "item.completed":
                action = data.get("action") or data.get("command") or data.get("name") or "task"
                status.update(f"[green]Completed[/green] {action}")

            message = (
                data.get("assistant_message")
                or event.get("assistant_message")
                or data.get("message")
            )
            if isinstance(message, str) and message.strip():
                assistant_parts.append(message.strip())

        proc.wait()
        if proc.stderr:
            stderr = proc.stderr.read().strip()
            if stderr:
                console.print(Panel(f"[red]{stderr}[/red]", title="Codex stderr", box=box.SIMPLE))
    finally:
        if status:
            status.stop()

    if assistant_parts:
        final_message = "\n\n".join(assistant_parts)
        console.print(Panel(final_message, title="Codex response", box=box.SIMPLE))
