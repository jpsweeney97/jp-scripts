from __future__ import annotations

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core.memory import query_memory, save_memory

app = typer.Typer(help="Persistent memory store for ADRs and lessons learned.")


@app.command("add")
def add(
    ctx: typer.Context,
    content: str = typer.Argument(..., help="Memory content or ADR/lesson learned."),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Tags to associate (repeatable)."),
) -> None:
    """Add a memory entry."""
    state = ctx.obj
    entry = save_memory(content, tags=tag, config=state.config)
    console.print(Panel(f"[green]Saved[/green] at {entry.ts}\nTags: {', '.join(entry.tags) if entry.tags else '—'}", title="Memory"))


@app.command("search")
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search text."),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum results to show."),
) -> None:
    """Search memory for relevant entries."""
    state = ctx.obj
    results = query_memory(query, limit=limit, config=state.config)

    if not results:
        console.print(Panel("No matching memories.", style="yellow"))
        return

    table = Table(title=f"Top {min(limit, len(results))} memories", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Entry", style="white")
    for line in results:
        table.add_row(line)

    console.print(table)
