"""Trace inspection commands for agentic workflow observability."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from jpscripts.core.console import console
from jpscripts.core.decorators import handle_exceptions
from jpscripts.core.engine import AgentTraceStep

app = typer.Typer(help="Inspect agentic execution traces.")


def _get_trace_dir() -> Path:
    """Return the default trace directory."""
    return Path.home() / ".jpscripts" / "traces"


def _parse_trace_line(line: str) -> AgentTraceStep | None:
    """Parse a single JSONL line into an AgentTraceStep."""
    try:
        data = json.loads(line.strip())
        return AgentTraceStep.model_validate(data)
    except (json.JSONDecodeError, Exception):
        return None


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_timestamp(ts: str) -> str:
    """Format ISO timestamp to a more readable form."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts[:19] if len(ts) >= 19 else ts


def _get_persona_color(persona: str) -> str:
    """Return a color for a given persona."""
    colors = {
        "architect": "cyan",
        "engineer": "green",
        "qa": "yellow",
    }
    return colors.get(persona.lower(), "white")


@app.callback()
def _trace_callback(ctx: typer.Context) -> None:
    """Trace inspection commands."""


@app.command("list")
@handle_exceptions
def list_traces(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", "-n", help="Number of traces to show"),
) -> None:
    """List recent execution traces."""
    _ = ctx
    trace_dir = _get_trace_dir()

    if not trace_dir.exists():
        console.print("[yellow]No traces found. Run an agent workflow first.[/yellow]")
        return

    # Glob trace files and sort by modification time (newest first)
    trace_files = sorted(
        trace_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not trace_files:
        console.print("[yellow]No trace files found.[/yellow]")
        return

    table = Table(
        title="Recent Execution Traces",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Timestamp", style="dim")
    table.add_column("ID", style="cyan")
    table.add_column("Persona", style="green")
    table.add_column("Thought", style="white")

    for trace_file in trace_files[:limit]:
        trace_id = trace_file.stem[:12]  # Short ID

        # Read first and last lines
        try:
            lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
            if not lines:
                continue

            first_step = _parse_trace_line(lines[0])
            last_step = _parse_trace_line(lines[-1]) if len(lines) > 1 else first_step

            if not first_step:
                continue

            timestamp = _format_timestamp(first_step.timestamp)
            persona = first_step.agent_persona

            # Extract thought snippet from last response
            thought = ""
            if last_step and last_step.response:
                thought = last_step.response.get("thought_process", "")
                if not thought:
                    thought = last_step.response.get("final_message", "")
            thought = _truncate(thought, 60)

            table.add_row(timestamp, trace_id, persona, thought)
        except Exception:
            continue

    console.print(table)
    console.print(f"\n[dim]Use `jp trace show <ID>` to view details.[/dim]")


@app.command("show")
@handle_exceptions
def show_trace(
    ctx: typer.Context,
    trace_id: str = typer.Argument(..., help="Trace ID (full or partial)"),
) -> None:
    """Display detailed trace for a specific execution."""
    _ = ctx
    trace_dir = _get_trace_dir()

    if not trace_dir.exists():
        console.print("[red]Trace directory not found.[/red]")
        raise typer.Exit(1)

    # Find matching trace file (support partial ID matching)
    matching_files = list(trace_dir.glob(f"{trace_id}*.jsonl"))
    if not matching_files:
        # Try broader match
        matching_files = [f for f in trace_dir.glob("*.jsonl") if trace_id in f.stem]

    if not matching_files:
        console.print(f"[red]No trace found matching '{trace_id}'[/red]")
        raise typer.Exit(1)

    if len(matching_files) > 1:
        console.print(f"[yellow]Multiple traces match '{trace_id}':[/yellow]")
        for f in matching_files[:5]:
            console.print(f"  - {f.stem}")
        console.print("[dim]Please provide a more specific ID.[/dim]")
        raise typer.Exit(1)

    trace_file = matching_files[0]
    console.print(f"[dim]Trace: {trace_file.stem}[/dim]\n")

    # Parse and display all steps
    try:
        lines = trace_file.read_text(encoding="utf-8").strip().split("\n")
    except Exception as exc:
        console.print(f"[red]Error reading trace: {exc}[/red]")
        raise typer.Exit(1)

    for i, line in enumerate(lines, 1):
        step = _parse_trace_line(line)
        if not step:
            continue

        persona = step.agent_persona
        color = _get_persona_color(persona)
        timestamp = _format_timestamp(step.timestamp)

        # Header panel
        header = Text()
        header.append(f"Step {i}: ", style="bold")
        header.append(persona, style=f"bold {color}")
        header.append(f" @ {timestamp}", style="dim")

        console.print(Panel(header, box=box.HEAVY, style=color))

        # Thought process
        thought = step.response.get("thought_process", "")
        if thought:
            console.print(Panel(
                thought,
                title="[bold]Thought Process[/bold]",
                border_style="dim",
                padding=(0, 1),
            ))

        # Tool call
        tool_call = step.response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("tool", "unknown")
            tool_args = json.dumps(tool_call.get("arguments", {}), indent=2)
            console.print(Panel(
                f"[cyan]{tool_name}[/cyan]\n{tool_args}",
                title="[bold]Tool Call[/bold]",
                border_style="blue",
                padding=(0, 1),
            ))

        # Tool output
        if step.tool_output:
            console.print(Panel(
                step.tool_output[:500] + ("..." if len(step.tool_output) > 500 else ""),
                title="[bold]Tool Output[/bold]",
                border_style="dim",
                padding=(0, 1),
            ))

        # File patch
        patch = step.response.get("file_patch")
        if patch:
            syntax = Syntax(patch, "diff", theme="monokai", line_numbers=True)
            console.print(Panel(
                syntax,
                title="[bold]File Patch[/bold]",
                border_style="green",
                padding=(0, 1),
            ))

        # Final message
        final_msg = step.response.get("final_message")
        if final_msg:
            console.print(Panel(
                final_msg,
                title="[bold]Final Message[/bold]",
                border_style="magenta",
                padding=(0, 1),
            ))

        console.print()  # Spacing between steps
