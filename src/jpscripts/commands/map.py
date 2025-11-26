from __future__ import annotations

from pathlib import Path

import typer

from jpscripts.core.console import console
from jpscripts.core.structure import generate_map


def map_cmd(
    ctx: typer.Context,
    root: Path = typer.Option(Path("."), "--root", "-r", help="Project root to map."),
    depth: int = typer.Option(5, "--depth", "-d", help="Maximum directory depth to traverse."),
) -> None:
    """Generate a concise project structure map with top-level symbols."""
    _ = ctx  # Typer requires ctx for shared state; unused here.
    try:
        project_map = generate_map(root, max_depth=depth)
    except Exception as exc:
        console.print(f"[red]Failed to generate map:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if project_map.strip():
        console.print(project_map)
    else:
        console.print("[yellow]No files found within the specified depth.[/yellow]")
