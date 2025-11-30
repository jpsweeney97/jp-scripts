from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from jpscripts.core.console import console
from jpscripts.core.result import Err, JPScriptsError, Ok, Result
from jpscripts.core.serializer import AsyncSerializer, RepoManifest, write_manifest_yaml

if TYPE_CHECKING:
    from jpscripts.main import AppState

app = typer.Typer(help="Serialize the current workspace into a lossless manifest.")


async def _run_snapshot(
    workspace_root: Path,
    output: Path,
    dry_run: bool,
) -> Result[tuple[RepoManifest, Path | None], JPScriptsError]:
    serializer = AsyncSerializer()
    manifest_result = await serializer.serialize(workspace_root)
    if isinstance(manifest_result, Err):
        return Err(manifest_result.error)

    manifest = manifest_result.value
    if dry_run:
        return Ok((manifest, None))

    write_result = await write_manifest_yaml(
        manifest,
        output,
        workspace_root=workspace_root,
    )
    if isinstance(write_result, Err):
        return Err(write_result.error)
    return Ok((manifest, write_result.value))


@app.command("snapshot")
def snapshot(
    ctx: typer.Context,
    target: Path | None = typer.Argument(
        None,
        help="Directory to serialize. Defaults to configured workspace root.",
    ),
    output: Path = typer.Option(
        Path("manifest.yaml"), "--output", "-o", help="Path to write the manifest."
    ),
    format: str = typer.Option(
        "yaml", "--format", "-f", help="Output format (only 'yaml' is supported)."
    ),
) -> None:
    state: AppState = ctx.obj
    fmt = format.lower()
    if fmt != "yaml":
        console.print("[red]Only 'yaml' format is supported for serialization.[/red]")
        raise typer.Exit(code=1)

    runtime = state.runtime_ctx
    resolved_root = (target or runtime.workspace_root).expanduser().resolve()
    result = asyncio.run(
        _run_snapshot(
            workspace_root=resolved_root,
            output=output,
            dry_run=runtime.dry_run,
        )
    )

    if isinstance(result, Err):
        console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(code=1)

    manifest, written_path = result.value
    if written_path is None:
        console.print(
            f"[yellow]Dry run:[/yellow] would serialize {manifest.file_count} files "
            f"({manifest.total_size_bytes} bytes) to {output}"
        )
        return

    console.print(
        f"[green]Serialized[/green] {manifest.file_count} files "
        f"({manifest.total_size_bytes} bytes) to {written_path}"
    )
