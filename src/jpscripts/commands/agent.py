from __future__ import annotations

import json
import shutil
import subprocess
import asyncio
from typing import Any
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel

from jpscripts.core.console import console
from jpscripts.core.nav import scan_recent
# NEW IMPORT
from jpscripts.core.context import gather_context

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
    # NEW FLAG
    run_command: str = typer.Option(None, "--run", "-x", help="Run this shell command first and attach referenced files from output (RAG)."),
    full_auto: bool = typer.Option(False, "--full-auto", "-y", help="Run without asking for confirmation (dangerous)."),
    model: str = typer.Option(None, "--model", "-m", help="Model to use. Defaults to config."),
) -> None:
    """Delegate a task to the Codex agent."""
    state = ctx.obj
    root = state.config.workspace_root or state.config.notes_dir
    target_model = model or getattr(state.config, "default_model", "gpt-5.1-codex-max")
    codex_bin = _ensure_codex()

    # 1. Build the base command
    cmd = [codex_bin, "exec", prompt, "--model", target_model, "--json"]

    if full_auto:
        cmd.append("--full-auto")

    # 2. Context Injection Strategy

    # Strategy A: JIT Context (Run & Scrape)
    if run_command:
        with console.status(f"Diagnosing with `{run_command}`...", spinner="dots"):
            output, detected_files = asyncio.run(gather_context(run_command, root.expanduser()))

        if not detected_files:
            console.print("[yellow]No files detected in command output. Proceeding without file context.[/yellow]")
        else:
            console.print(f"[green]Detected relevant files:[/green] {', '.join(f.name for f in detected_files)}")
            for path in detected_files:
                cmd.extend(["--file", str(path)])

        # We also attach the output log as a temporary file context or preamble?
        # Simpler: Just append it to the prompt so Codex sees the error.
        # This is "God Mode" - we modify the user prompt transparently.
        prompt += f"\n\nCommand `{run_command}` Output:\n```\n{output[-2000:]}\n```" # limit to last 2k chars to avoid token limit
        # Update the command list prompt argument (index 2)
        cmd[2] = prompt

    # Strategy B: Recent Files (Heuristic)
    elif attach_recent:
        with console.status("Scanning for recent context...", spinner="dots"):
            recents = asyncio.run(scan_recent(
                root.expanduser(),
                max_depth=3,
                include_dirs=False,
                ignore_dirs=set(state.config.ignore_dirs)
            ))

        # Attach top 5 files
        for entry in recents[:5]:
            cmd.extend(["--file", str(entry.path)])
            console.print(f"[dim]Attaching context: {entry.path.name}[/dim]")

    # 3. Handoff (Existing Logic)
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
