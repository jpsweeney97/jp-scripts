from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Mapping
from uuid import uuid4

import click
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from typer.main import get_command

from . import __version__
from .core.config import AppConfig, ConfigLoadResult, load_config
from .core.console import console, setup_logging
from .core.diagnostics import run_diagnostics_suite
from .core.registry import discover_commands
from .core.runtime import RuntimeContext, _runtime_ctx

app = typer.Typer(help="jp: the modern Python CLI for the jp-scripts toolbox.")
logger = logging.getLogger(__name__)

# Token for CLI runtime context (established in callback, persists through command)
_cli_runtime_token: contextvars.Token[RuntimeContext | None] | None = None


@dataclass
class AppState:
    config: AppConfig
    config_meta: ConfigLoadResult
    logger: logging.Logger
    runtime_ctx: RuntimeContext = field(default=None)  # type: ignore[assignment]


def _establish_cli_runtime(config: AppConfig, dry_run: bool) -> RuntimeContext:
    """Establish runtime context for CLI commands.

    Since CLI commands run synchronously after the callback returns,
    we establish the context and don't reset it during command execution.
    """
    global _cli_runtime_token

    ctx = RuntimeContext(
        config=config,
        workspace_root=config.workspace_root.expanduser().resolve(),
        trace_id=f"cli-{uuid4().hex[:8]}",
        dry_run=dry_run,
    )
    _cli_runtime_token = _runtime_ctx.set(ctx)
    return ctx


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

    # Establish runtime context for all subsequent operations
    runtime = _establish_cli_runtime(loaded_config, dry_run)

    ctx.obj = AppState(
        config=loaded_config,
        config_meta=meta,
        logger=logger,
        runtime_ctx=runtime,
    )

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
        logger.debug(
            "Loaded configuration from %s (env overrides: %s, trace: %s)",
            meta.path,
            sorted(meta.env_overrides),
            runtime.trace_id,
        )

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

def _register_commands() -> None:
    commands_path = Path(__file__).resolve().parent / "commands"
    typer_modules, function_commands = discover_commands(commands_path)

    for name, module in typer_modules:
        app.add_typer(module.app, name=name)

    for spec in function_commands:
        app.command(spec.name)(spec.handler)


def _register_commands_with_timing() -> None:
    start = perf_counter()
    _register_commands()
    elapsed = perf_counter() - start
    logger.debug("Command registry initialized in %.3f seconds", elapsed)


_register_commands_with_timing()

def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
