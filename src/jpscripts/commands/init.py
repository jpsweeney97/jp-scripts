from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt

from jpscripts.core import security
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console


def _write_config(path: Path, config: AppConfig) -> None:
    ignore_dirs_literal = ", ".join(json.dumps(item) for item in config.ignore_dirs)
    content = dedent(
        f"""
        # jpscripts configuration (TOML)
        editor = "{config.editor}"
        notes_dir = "{config.notes_dir}"
        workspace_root = "{config.workspace_root}"
        ignore_dirs = [{ignore_dirs_literal}]
        snapshots_dir = "{config.snapshots_dir}"
        log_level = "{config.log_level}"
        worktree_root = "{config.worktree_root or ""}"
        focus_audio_device = "{config.focus_audio_device or ""}"
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")


def init(
    ctx: typer.Context,
    config_path: Path | None = typer.Option(None, help="Where to write config."),
    install_hooks: bool = typer.Option(
        False, "--install-hooks", help="Install git hooks (pre-commit) to enforce protocols."
    ),
) -> None:
    """Interactive initializer that writes the active config file."""
    state = ctx.obj
    defaults: AppConfig = state.config
    target_path = (config_path or state.config_meta.path).expanduser()

    notes_dir = Path(Prompt.ask("Notes directory", default=str(defaults.notes_dir)))
    workspace_root = Path(Prompt.ask("Workspace root", default=str(defaults.workspace_root)))
    worktree_root_input = Prompt.ask(
        "Worktree root (optional)", default=str(defaults.worktree_root or "")
    )
    worktree_root = Path(worktree_root_input).expanduser() if worktree_root_input else None
    editor = Prompt.ask("Editor command", default=defaults.editor)
    log_level = Prompt.ask("Log level", default=defaults.log_level)
    snapshots_dir = Path(Prompt.ask("Snapshots directory", default=str(defaults.snapshots_dir)))
    focus_audio_device = (
        Prompt.ask(
            "Preferred audio device (optional)", default=defaults.focus_audio_device or ""
        ).strip()
        or None
    )
    ignore_dirs_input = Prompt.ask(
        "Ignore directories (comma separated)",
        default=",".join(defaults.ignore_dirs),
    )
    ignore_dirs = [item.strip() for item in ignore_dirs_input.split(",") if item.strip()]
    if not ignore_dirs:
        ignore_dirs = defaults.ignore_dirs

    config = AppConfig(
        editor=editor,
        notes_dir=notes_dir,
        workspace_root=workspace_root,
        snapshots_dir=snapshots_dir,
        log_level=log_level,
        worktree_root=worktree_root,
        focus_audio_device=focus_audio_device,
        ignore_dirs=ignore_dirs,
    )

    for target in [
        config.notes_dir,
        config.workspace_root,
        config.snapshots_dir,
        config.worktree_root,
    ]:
        if target:
            Path(target).expanduser().mkdir(parents=True, exist_ok=True)

    _write_config(target_path, config)
    console.print(f"[green]Wrote config to[/green] {target_path}")
    console.print("You can rerun `jp init` anytime to update these values.")

    if install_hooks:
        _install_precommit_hook(config.workspace_root)


def config_fix(ctx: typer.Context) -> None:
    """Attempt to fix a broken configuration file using Codex."""
    state = ctx.obj
    path = state.config_meta.path

    if not path.exists():
        console.print(f"[red]Config file {path} does not exist. Run `jp init` to create one.[/red]")
        raise typer.Exit(code=1)

    if not state.config_meta.error:
        console.print(f"[green]Config file {path} is valid. No fix needed.[/green]")
        return

    # Read the broken content
    content = path.read_text(encoding="utf-8")

    console.print(Panel(f"Attempting to fix {path}...", title="Self-Healing", box=box.SIMPLE))

    # Check for Codex
    codex_bin = shutil.which("codex")
    if not codex_bin:
        console.print("[red]Codex CLI not found. Cannot auto-fix.[/red]")
        console.print("Please fix the file manually or run `jp init` to overwrite it.")
        raise typer.Exit(code=1)

    # Construct the prompt
    prompt = (
        f"The following TOML configuration file is invalid.\n"
        f"Error: {state.config_meta.error}\n\n"
        f"Content:\n```toml\n{content}\n```\n\n"
        f"Fix the syntax errors and overwrite the file at {path} with the corrected TOML."
    )

    # Delegate to Codex
    # We use --full-auto (YOLO mode) because we are fixing a broken config
    cmd = [codex_bin, "exec", prompt, "--full-auto", "--model", "gpt-5.1-codex-max"]

    try:
        exit_code = asyncio.run(_run_codex_command(cmd))
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, cmd)
        console.print(f"[green]Repaired[/green] {path}")
    except subprocess.CalledProcessError:
        console.print("[red]Codex failed to fix the configuration.[/red]")
        raise typer.Exit(code=1)


async def _run_codex_command(cmd: list[str]) -> int:
    """Run Codex CLI asynchronously while preserving terminal IO."""
    proc = await asyncio.create_subprocess_exec(*cmd)
    return await proc.wait()


def _install_precommit_hook(workspace_root: Path) -> None:
    try:
        root = security.validate_workspace_root(workspace_root)
    except Exception as exc:
        console.print(f"[red]Cannot install hooks: {exc}[/red]")
        return

    git_dir = root / ".git"
    hooks_dir = git_dir / "hooks"
    precommit = hooks_dir / "pre-commit"

    if not git_dir.exists():
        console.print(f"[yellow]Skipping hook install: {git_dir} not found.[/yellow]")
        return

    try:
        hooks_dir.mkdir(parents=True, exist_ok=True)
        script = "#!/bin/sh\njp verify-protocol --name pre-commit\n"
        precommit.write_text(script, encoding="utf-8")
        precommit.chmod(0o755)
        console.print(f"[green]Installed pre-commit hook at {precommit}[/green]")
    except OSError as exc:
        console.print(f"[red]Failed to install pre-commit hook: {exc}[/red]")
