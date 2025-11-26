from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import click
import typer
from pydantic import BaseModel, Field
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from typer.main import get_command

from . import __version__
from .commands import agent, git_extra, git_ops, handbook, init, map, memory, nav, notes, search, system, team, web
from .core.config import AppConfig, ConfigError, ConfigLoadResult, load_config
from .core.console import console, setup_logging
from .core.memory import LanceDBStore
from .core.security import WorkspaceValidationError, validate_workspace_root

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


class DiagnosticCheck(ABC):
    name: str

    @abstractmethod
    async def run(self) -> tuple[str, str]:
        """Run the diagnostic and return (status, message)."""


class ConfigCheck(DiagnosticCheck):
    def __init__(self, config: AppConfig, config_path: Path | None) -> None:
        self.config = config
        self.config_path = config_path
        self.name = "Config"

    async def run(self) -> tuple[str, str]:
        issues: list[str] = []
        if self.config_path and self.config_path.exists():
            try:
                _ = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"Invalid config TOML: {exc}")
        elif self.config_path:
            issues.append(f"Config file missing: {self.config_path}")

        for label, path in (("workspace_root", self.config.workspace_root), ("notes_dir", self.config.notes_dir)):
            expanded = path.expanduser()
            if not expanded.exists():
                issues.append(f"{label} missing: {expanded}")
            elif not os.access(expanded, os.W_OK):
                issues.append(f"{label} not writable: {expanded}")
            else:
                try:
                    validate_workspace_root(expanded) if label == "workspace_root" else None
                except WorkspaceValidationError as exc:
                    issues.append(str(exc))

        if issues:
            return "error", "; ".join(issues)
        return "ok", "Configuration valid."


class AuthCheck(DiagnosticCheck):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.name = "Auth"

    async def run(self) -> tuple[str, str]:
        model = (self.config.default_model or "").lower()
        if "local" in model or "offline" in model:
            return "ok", "Local model in use; API key not required."
        if os.environ.get("OPENAI_API_KEY"):
            return "ok", "OPENAI_API_KEY present."
        return "warn", "OPENAI_API_KEY missing for remote models."


class VectorDBCheck(DiagnosticCheck):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.name = "VectorDB"

    async def run(self) -> tuple[str, str]:
        store_path = Path(self.config.memory_store).expanduser()
        try:
            store = LanceDBStore(store_path, embedding_dim=1)
            _ = store.search([0.0], limit=1)
            return "ok", f"LanceDB ready at {store_path}"
        except ImportError:
            return "warn", "lancedb not installed; vector memory unavailable."
        except Exception as exc:
            return "error", f"Vector DB check failed: {exc}"


class MCPCheck(DiagnosticCheck):
    def __init__(self) -> None:
        self.name = "MCP"
        self.config_path = Path.home() / ".codex" / "config.toml"

    async def run(self) -> tuple[str, str]:
        if not self.config_path.exists():
            return "warn", f"MCP config missing at {self.config_path}"
        try:
            data = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return "warn", f"MCP config unreadable: {exc}"

        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if isinstance(servers, dict) and "jpscripts" in servers:
            return "ok", "jpscripts MCP server registered."
        return "warn", "jpscripts MCP server not registered."



DEFAULT_TOOLS: list[ExternalTool] = [
    ExternalTool(name="Git", binary="git", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="ripgrep", binary="rg", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="fzf", binary="fzf", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="GitHub CLI", binary="gh", install_hint="Install via your package manager (brew, apt, etc.)"),
    ExternalTool(name="Codex", binary="codex", install_hint="npm install -g @openai/codex"),
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
    tree = Tree("Binaries")
    style_map = {"ok": "green", "missing": "red", "error": "yellow"}
    for result in results:
        status_style = style_map.get(result.status, "white")
        status_text = f"[{status_style}]{result.status}[/{status_style}]"
        message = result.version or result.message or result.tool.install_hint or ""
        tree.add(f"{status_text} {result.tool.name} ({result.tool.binary}) {message}".strip())
    console.print(tree)


@app.command("doctor")
def doctor(
    ctx: typer.Context,
    tool: list[str] | None = typer.Option(None, "--tool", "-t", help="Check only specific tools (name or binary)."),
) -> None:
    """Inspect external dependencies in parallel."""
    state: AppState = ctx.obj
    state.logger.debug("Running doctor for tools: %s", tool or "all")

    tools = _select_tools(tool)

    async def _run_checks() -> tuple[list[tuple[str, str, str]], list[ToolCheck]]:
        diag_checks: list[DiagnosticCheck] = [
            ConfigCheck(state.config, state.config_meta.path),
            AuthCheck(state.config),
            VectorDBCheck(state.config),
            MCPCheck(),
        ]
        diag_results: list[tuple[str, str, str]] = []
        for check in diag_checks:
            status, message = await check.run()
            diag_results.append((check.name, status, message))
        tool_results = await _run_doctor(tools)
        return diag_results, tool_results

    diag_results, tool_results = asyncio.run(_run_checks())

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
