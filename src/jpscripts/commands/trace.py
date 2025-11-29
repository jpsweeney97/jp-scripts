"""Trace inspection commands for agentic workflow observability."""

from __future__ import annotations

import json
from datetime import datetime
import time
from pathlib import Path

import typer
from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

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


def _build_trace_tree(trace_id: str, steps: list[AgentTraceStep]) -> Group:
    if not steps:
        return Group(Panel("No steps recorded.", title="Trace", border_style="red"))

    first = steps[0]
    root_label = f"{trace_id} / {first.timestamp} / {first.agent_persona}"
    tree = Tree(root_label, guide_style="dim")

    context_branch = tree.add("Context")
    for msg in first.input_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        context_branch.add(f"{role}: {_truncate(content, 160)}")

    execution_branch = tree.add("Execution Flow")
    total_tokens = 0
    tool_calls = 0
    patches = 0

    for idx, step in enumerate(steps, 1):
        persona_color = _get_persona_color(step.agent_persona)
        step_node = execution_branch.add(f"[{persona_color}]Step {idx}: {step.agent_persona}[/{persona_color}]")

        thought = step.response.get("thought_process") or ""
        if thought:
            step_node.add(Panel(thought, title="Thought", border_style="dim", padding=(0, 1)))

        usage = step.response.get("usage")
        if usage:
            tokens = usage.get("total_tokens") or usage.get("tokens") or 0
            try:
                total_tokens += int(tokens)
            except (TypeError, ValueError):
                pass

        tool_call = step.response.get("tool_call")
        if tool_call:
            tool_calls += 1
            tool_name = tool_call.get("tool", "unknown")
            tool_args = json.dumps(tool_call.get("arguments", {}), indent=2)
            tool_node = step_node.add(f"[cyan]Tool Call: {tool_name}[/cyan]")
            tool_node.add(Panel(tool_args, border_style="blue", padding=(0, 1)))
            if step.tool_output:
                tool_node.add(Panel(step.tool_output, title="Output", border_style="dim", padding=(0, 1)))

        patch = step.response.get("file_patch")
        if patch:
            patches += 1
            step_node.add(Panel(Syntax(patch, "diff", line_numbers=True), title="[green]File Patch[/green]", border_style="green"))

        final_msg = step.response.get("final_message")
        if final_msg:
            step_node.add(Panel(final_msg, title="Final Message", border_style="magenta", padding=(0, 1)))

    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Steps", str(len(steps)))
    summary.add_row("Tool Calls", str(tool_calls))
    summary.add_row("Patches", str(patches))
    summary.add_row("Tokens", str(total_tokens) if total_tokens else "unavailable")

    return Group(tree, Panel(summary, title="Trace Summary", box=box.SIMPLE))


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
    watch: bool = typer.Option(False, "--watch", "-w", help="Stream trace updates in real-time."),
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

    def _load_steps() -> list[AgentTraceStep]:
        try:
            raw = trace_file.read_text(encoding="utf-8").strip().split("\n")
        except Exception:
            return []
        steps: list[AgentTraceStep] = []
        for line in raw:
            step = _parse_trace_line(line)
            if step:
                steps.append(step)
        return steps

    def _render_current() -> Group:
        steps = _load_steps()
        return _build_trace_tree(trace_file.stem, steps)

    if watch:
        with Live(_render_current(), console=console, refresh_per_second=2) as live:
            try:
                last_size = trace_file.stat().st_size
                while True:
                    time.sleep(1)
                    current_size = trace_file.stat().st_size
                    if current_size != last_size:
                        live.update(_render_current())
                        last_size = current_size
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped watching trace.[/yellow]")
        return

    console.print(_render_current())
