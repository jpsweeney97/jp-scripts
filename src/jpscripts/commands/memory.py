from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core.result import JPScriptsError
from jpscripts.core.memory import prune_memory, query_memory, reindex_memory, save_memory

app = typer.Typer(help="Persistent memory store for ADRs and lessons learned.")


@app.command("add")
def add(
    ctx: typer.Context,
    content: str = typer.Argument(..., help="Memory content or ADR/lesson learned."),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Tags to associate (repeatable)."),
) -> None:
    """Add a memory entry."""
    state = ctx.obj
    try:
        entry = save_memory(content, tags=tag, config=state.config)
    except JPScriptsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(Panel(f"[green]Saved[/green] at {entry.ts}\nTags: {', '.join(entry.tags) if entry.tags else '—'}", title="Memory"))


@app.command("search")
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search text."),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum results to show."),
) -> None:
    """Search memory for relevant entries."""
    state = ctx.obj
    try:
        results = query_memory(query, limit=limit, config=state.config)
    except JPScriptsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if not results:
        console.print(Panel("No matching memories.", style="yellow"))
        return

    table = Table(title=f"Top {min(limit, len(results))} memories", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Entry", style="white")
    for line in results:
        table.add_row(line)

    console.print(table)


@app.command("reindex")
def reindex(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
) -> None:
    state = ctx.obj
    store_path = Path(state.config.memory_store).expanduser()
    if force and store_path.exists():
        if store_path.is_dir():
            shutil.rmtree(store_path, ignore_errors=True)
        else:
            store_path.unlink(missing_ok=True)

    try:
        rebuilt_path = reindex_memory(config=state.config, target_path=store_path)
    except JPScriptsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(Panel(f"[green]Memory reindexed.[/green]\nStore: {rebuilt_path}", title="Memory"))


@app.command("vacuum")
def vacuum(ctx: typer.Context) -> None:
    """Remove memory entries related to deleted files to maintain vector store hygiene."""
    state = ctx.obj
    try:
        count = prune_memory(state.config)
    except JPScriptsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Pruned {count} stale memory entries.[/green]")
