from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import click
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from typer.main import get_command

from . import __version__
from .commands import agent, git_extra, git_ops, handbook, init, map, memory, nav, notes, search, system, team, web
from .core.config import AppConfig, ConfigLoadResult, load_config
from .core.console import console, setup_logging
from .core.diagnostics import ExternalTool, ToolCheck, run_diagnostics_suite

app = typer.Typer(help="jp: the modern Python CLI for the jp-scripts toolbox.")


@dataclass
class AppState:
    config: AppConfig
    config_meta: ConfigLoadResult
    logger: logging.Logger


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to a jp config file (TOML or JSON)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Enable dry-run mode (no side effects)."),
) -> None:
    # We no longer try/except here because load_config is safe
    loaded_config, meta = load_config(config_path=config)
    if dry_run:
        loaded_config = loaded_config.model_copy(update={"dry_run": True})

    logger = setup_logging(level=loaded_config.log_level, verbose=verbose)
    ctx.obj = AppState(config=loaded_config, config_meta=meta, logger=logger)

    if meta.error:
        # Display "Safe Mode" Warning
        console.print(
            Panel(
                f"[bold red]Configuration Error - Safe Mode Active[/bold red]\n\n"
                f"Failed to load {meta.path}:\n{meta.error}\n\n"
                f"[yellow]Using default settings. Run `jp config-fix` to repair.[/yellow]",
                border_style="red"
            )
        )
    else:
        logger.debug("Loaded configuration from %s (env overrides: %s)", meta.path, sorted(meta.env_overrides))

@app.command("com")
def command_catalog() -> None:
    """Display the available jp commands and their descriptions."""
    click_app = get_command(app)
    table = Table(title="jp commands", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Summary", style="white")

    commands: Mapping[str, click.Command] = {}
    if isinstance(click_app, click.Group):
        commands = click_app.commands
    for name, command in sorted(commands.items()):
        if name == "help":
            continue
        summary = (command.help or command.short_help or "").strip()
        table.add_row(name, summary or "—")

    console.print(table)


@app.command("doctor")
def doctor(
    ctx: typer.Context,
    tool: list[str] | None = typer.Option(None, "--tool", "-t", help="Check only specific tools (name or binary)."),
) -> None:
    """Inspect external dependencies in parallel."""
    state: AppState = ctx.obj
    state.logger.debug("Running doctor for tools: %s", tool or "all")

    diag_results, tool_results = asyncio.run(
        run_diagnostics_suite(
            config=state.config,
            config_path=state.config_meta.path,
            tool_names=tool,
        )
    )

    tree = Tree("System Health")
    style_map = {"ok": "green", "warn": "yellow", "error": "red", "missing": "red"}
    diag_branch = tree.add("Deep Checks")
    for name, status, message in diag_results:
        style = style_map.get(status, "white")
        diag_branch.add(f"[{style}]{status}[/{style}] {name}: {message}")

    tools_branch = tree.add("Binaries")
    for result in tool_results:
        style = style_map.get(result.status, "white")
        message = result.version or result.message or result.tool.install_hint or ""
        tools_branch.add(f"[{style}]{result.status}[/{style}] {result.tool.name} ({result.tool.binary}) {message}".strip())

    console.print(tree)


@app.command("config")
def show_config(ctx: typer.Context) -> None:
    """Show the active configuration and where it came from."""
    state: AppState = ctx.obj
    config = state.config
    meta = state.config_meta

    table = Table(title="Config", box=box.SIMPLE, expand=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in config.model_dump().items():
        table.add_row(key, str(value))

    console.print(table)

    meta_lines = [
        f"Path: {meta.path}",
        "File loaded: yes" if meta.file_loaded else "File loaded: no (using defaults + env)",
    ]

    if meta.env_overrides:
        meta_lines.append("Env overrides: " + ", ".join(sorted(meta.env_overrides)))

    console.print(Panel("\n".join(meta_lines), title="Config source", box=box.SIMPLE))


@app.command("version")
def show_version() -> None:
    """Print the jpscripts version."""
    console.print(__version__)


app.command("status-all")(git_ops.status_all)
app.command("whatpush")(git_ops.whatpush)
app.command("sync")(git_ops.sync)
app.command("recent")(nav.recent)
app.command("proj")(nav.proj)
app.command("init")(init.init)
app.add_typer(team.app, name="team")
app.add_typer(memory.app, name="memory")
app.command("web-snap")(web.web_snap)
app.command("process-kill")(system.process_kill)
app.command("port-kill")(system.port_kill)
app.command("brew-explorer")(system.brew_explorer)
app.command("audioswap")(system.audioswap)
app.command("ssh-open")(system.ssh_open)
app.command("tmpserver")(system.tmpserver)
app.command("update")(system.update)
app.command("note")(notes.note)
app.command("note-search")(notes.note_search)
app.command("standup")(notes.standup)
app.command("standup-note")(notes.standup_note)
app.command("cliphist")(notes.cliphist)
app.command("map")(map.map_cmd)
app.command("repo-map")(map.map_cmd)
app.command("ripper")(search.ripper)
app.command("todo-scan")(search.todo_scan)
app.command("loggrep")(search.loggrep)
app.command("gundo-last")(git_extra.gundo_last)
app.command("gstage")(git_extra.gstage)
app.command("gpr")(git_extra.gpr)
app.command("gbrowse")(git_extra.gbrowse)
app.command("git-branchcheck")(git_extra.git_branchcheck)
app.command("stashview")(git_extra.stashview)
app.command("fix")(agent.codex_exec)  # "jp fix" is faster to type than "jp agent"
app.command("agent")(agent.codex_exec) # Alias
app.command("config-fix")(init.config_fix)
app.command("handbook")(handbook.handbook)

def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
