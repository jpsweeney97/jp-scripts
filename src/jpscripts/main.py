from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
from pydantic import BaseModel, Field
from rich import box
from rich.panel import Panel
from rich.table import Table
from typer.main import get_command

from . import __version__
from .commands import agent, git_extra, git_ops, init, nav, notes, search, system, web
from .core.config import AppConfig, ConfigError, ConfigLoadResult, load_config
from .core.console import console, setup_logging

app = typer.Typer(help="jp: the modern Python CLI for the jp-scripts toolbox.")


@dataclass
class AppState:
    config: AppConfig
    config_meta: ConfigLoadResult
    logger: logging.Logger


class ExternalTool(BaseModel):
    name: str
    binary: str
    version_args: list[str] = Field(default_factory=lambda: ["--version"])
    required: bool = True
    install_hint: str | None = None


@dataclass
class ToolCheck:
    tool: ExternalTool
    status: str
    version: str | None
    message: str | None = None



DEFAULT_TOOLS: list[ExternalTool] = [
    ExternalTool(name="Git", binary="git", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="ripgrep", binary="rg", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="fzf", binary="fzf", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="GitHub CLI", binary="gh", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="Python", binary="python3", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="Homebrew", binary="brew", install_hint="macOS: https://brew.sh"),
    ExternalTool(name="System Clipboard", binary="pbcopy", install_hint="macOS: Built-in. Linux: Install xclip/xsel.", required=False),
    ExternalTool(name="SwitchAudioSource", binary="SwitchAudioSource", required=False),
    ExternalTool(name="zoxide", binary="zoxide", install_hint="Install via your package manager (brew, apt, etc.)", required=False),
]

@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to a jp config file (TOML or JSON)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    # We no longer try/except here because load_config is safe
    loaded_config, meta = load_config(config_path=config)

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

    for name, command in sorted(click_app.commands.items()):
        if name == "help":
            continue
        summary = (command.help or command.short_help or "").strip()
        table.add_row(name, summary or "—")

    console.print(table)


def _select_tools(names: Iterable[str] | None) -> list[ExternalTool]:
    if not names:
        return DEFAULT_TOOLS

    requested = {name.lower() for name in names}
    selected = [
        tool for tool in DEFAULT_TOOLS if tool.name.lower() in requested or tool.binary.lower() in requested
    ]
    return selected or DEFAULT_TOOLS


async def _check_tool(tool: ExternalTool) -> ToolCheck:
    resolved = shutil.which(tool.binary)
    if not resolved:
        return ToolCheck(tool=tool, status="missing", version=None, message=tool.install_hint)

    try:
        # FIX: Redirect stdin to DEVNULL to prevent interactive tools (like pbcopy) from hanging
        process = await asyncio.create_subprocess_exec(
            resolved,
            *tool.version_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,  # <--- CRITICAL FIX
        )
    except FileNotFoundError:
        return ToolCheck(tool=tool, status="missing", version=None, message=tool.install_hint)

    stdout, stderr = await process.communicate()
    output = (stdout or b"").decode().strip() or (stderr or b"").decode().strip()
    version = output.splitlines()[0] if output else None

    if process.returncode != 0:
        return ToolCheck(tool=tool, status="error", version=version, message=output or "version command failed")

    return ToolCheck(tool=tool, status="ok", version=version, message=None)


async def _run_doctor(tools: list[ExternalTool]) -> list[ToolCheck]:
    tasks = [asyncio.create_task(_check_tool(tool)) for tool in tools]
    return await asyncio.gather(*tasks)


def _render_tool_checks(results: list[ToolCheck]) -> None:
    table = Table(title="External tools", box=box.SIMPLE, expand=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Binary", style="white", no_wrap=True)
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Version / Message", style="white")
    table.add_column("Install hint", style="white")

    for result in results:
        status_text = {
            "ok": "[green]ok[/]",
            "missing": "[red]missing[/]",
            "error": "[yellow]error[/]",
        }.get(result.status, result.status)

        table.add_row(
            result.tool.name,
            result.tool.binary,
            status_text,
            result.version or result.message or "",
            result.tool.install_hint or "",
        )

    console.print(table)

    missing = [r.tool.name for r in results if r.status == "missing" and r.tool.required]
    if missing:
        console.print(Panel.fit("Missing required tools: " + ", ".join(missing), style="red"))


@app.command("doctor")
def doctor(
    ctx: typer.Context,
    tool: list[str] | None = typer.Option(None, "--tool", "-t", help="Check only specific tools (name or binary)."),
) -> None:
    """Inspect external dependencies in parallel."""
    state: AppState = ctx.obj
    state.logger.debug("Running doctor for tools: %s", tool or "all")

    tools = _select_tools(tool)
    results = asyncio.run(_run_doctor(tools))
    _render_tool_checks(results)


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
app.command("web-snap")(web.web_snap)
app.command("process-kill")(system.process_kill)
app.command("port-kill")(system.port_kill)
app.command("brew-explorer")(system.brew_explorer)
app.command("audioswap")(system.audioswap)
app.command("ssh-open")(system.ssh_open)
app.command("tmpserver")(system.tmpserver)
app.command("note")(notes.note)
app.command("note-search")(notes.note_search)
app.command("standup")(notes.standup)
app.command("standup-note")(notes.standup_note)
app.command("cliphist")(notes.cliphist)
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

def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
