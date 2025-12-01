"""Memory store management commands.

Provides CLI commands for managing the vector-based memory store:
    - Adding and querying memories
    - Clustering similar memories
    - Pruning old or redundant entries
    - Exporting and importing memory data
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core.memory import (
    HybridMemoryStore,
    MemoryEntry,
    _write_entries,  # pyright: ignore[reportPrivateUsage]
    cluster_memories,
    get_memory_store,
    prune_memory,
    query_memory,
    reindex_memory,
    save_memory,
    synthesize_cluster,
)
from jpscripts.core.result import CapabilityMissingError, Err, JPScriptsError, Ok

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
    console.print(
        Panel(
            f"[green]Saved[/green] at {entry.ts}\nTags: {', '.join(entry.tags) if entry.tags else '—'}",
            title="Memory",
        )
    )


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

    table = Table(
        title=f"Top {min(limit, len(results))} memories", box=box.SIMPLE_HEAVY, expand=True
    )
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
            shutil.rmtree(store_path, ignore_errors=True)  # safety: checked
        else:
            store_path.unlink(missing_ok=True)  # safety: checked

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


@app.command("consolidate")
def consolidate(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model used to synthesize canonical memories."
    ),
    threshold: float = typer.Option(
        0.85, "--threshold", help="Cosine similarity threshold for clustering."
    ),
) -> None:
    """Cluster similar memories and synthesize canonical truth entries."""
    state = ctx.obj

    clusters_result = asyncio.run(cluster_memories(state.config, similarity_threshold=threshold))
    match clusters_result:
        case Err(err):
            if isinstance(err, CapabilityMissingError):
                console.print(f"[red]{err}[/red]")
            else:
                console.print(f"[red]{err}[/red]")
            raise typer.Exit(code=1)
        case Ok(clusters):
            pass

    if not clusters:
        console.print("[yellow]No clusters found for consolidation.[/yellow]")
        return

    store_result = get_memory_store(state.config)
    if isinstance(store_result, Err):
        console.print(f"[red]{store_result.error}[/red]")
        raise typer.Exit(code=1)

    store = store_result.value
    if not isinstance(store, HybridMemoryStore):
        console.print(
            "[red]Consolidation requires the hybrid memory store with LanceDB enabled.[/red]"
        )
        raise typer.Exit(code=1)

    archived_ids: set[str] = set()
    synthesized_entries: list[MemoryEntry] = []

    for cluster in clusters:
        synth_result = asyncio.run(synthesize_cluster(cluster, state.config, model=model))
        if isinstance(synth_result, Err):
            console.print(f"[red]Synthesis failed: {synth_result.error}[/red]")
            continue
        synthesized_entries.append(synth_result.value)
        archived_ids.update(entry.id for entry in cluster)

    if not synthesized_entries:
        console.print("[yellow]No synthesized entries were created.[/yellow]")
        return

    existing_entries: list[MemoryEntry] = asyncio.run(
        asyncio.to_thread(store.archiver.load_entries)
    )
    updated_entries: list[MemoryEntry] = []
    for entry in existing_entries:
        if entry.id in archived_ids and "archived" not in entry.tags:
            entry.tags.append("archived")
        updated_entries.append(entry)

    updated_entries.extend(synthesized_entries)
    asyncio.run(asyncio.to_thread(_write_entries, store.archiver.path, updated_entries))

    if store.vector_store:
        for entry in synthesized_entries:
            if entry.embedding is not None:
                add_result = store.vector_store.add(entry)
                if isinstance(add_result, Err):
                    console.print(
                        f"[yellow]Vector insert failed for {entry.id}: {add_result.error}[/yellow]"
                    )

    console.print(
        Panel(
            f"[green]Consolidated {len(clusters)} clusters[/green]\n"
            f"Created {len(synthesized_entries)} canonical entries.\n"
            f"Archived {len(archived_ids)} originals.",
            title="Memory Consolidation",
            box=box.SIMPLE_HEAVY,
        )
    )
