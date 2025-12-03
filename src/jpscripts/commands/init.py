"""Project initialization and configuration setup.

Provides CLI commands for:
    - Initializing new jpscripts configuration
    - Setting up workspace defaults
    - Configuring editor and notes directory
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from textwrap import dedent

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Prompt

from jpscripts.core import security
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console
from jpscripts.core.result import Err, Ok
from jpscripts.providers import CompletionOptions, LLMProvider, Message, ProviderError
from jpscripts.providers.factory import get_provider


def _write_config(path: Path, config: AppConfig) -> None:
    ignore_dirs_literal = ", ".join(json.dumps(item) for item in config.user.ignore_dirs)
    content = dedent(
        f"""
        # jpscripts configuration (TOML)
        editor = "{config.user.editor}"
        notes_dir = "{config.user.notes_dir}"
        workspace_root = "{config.user.workspace_root}"
        ignore_dirs = [{ignore_dirs_literal}]
        snapshots_dir = "{config.user.snapshots_dir}"
        log_level = "{config.user.log_level}"
        worktree_root = "{config.infra.worktree_root or ""}"
        focus_audio_device = "{config.user.focus_audio_device or ""}"
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

    notes_dir = Path(Prompt.ask("Notes directory", default=str(defaults.user.notes_dir)))
    workspace_root = Path(Prompt.ask("Workspace root", default=str(defaults.user.workspace_root)))
    worktree_root_input = Prompt.ask(
        "Worktree root (optional)", default=str(defaults.infra.worktree_root or "")
    )
    worktree_root = Path(worktree_root_input).expanduser() if worktree_root_input else None
    editor = Prompt.ask("Editor command", default=defaults.user.editor)
    log_level = Prompt.ask("Log level", default=defaults.user.log_level)
    snapshots_dir = Path(
        Prompt.ask("Snapshots directory", default=str(defaults.user.snapshots_dir))
    )
    focus_audio_device = (
        Prompt.ask(
            "Preferred audio device (optional)", default=defaults.user.focus_audio_device or ""
        ).strip()
        or None
    )
    ignore_dirs_input = Prompt.ask(
        "Ignore directories (comma separated)",
        default=",".join(defaults.user.ignore_dirs),
    )
    ignore_dirs = [item.strip() for item in ignore_dirs_input.split(",") if item.strip()]
    if not ignore_dirs:
        ignore_dirs = defaults.user.ignore_dirs

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
        config.user.notes_dir,
        config.user.workspace_root,
        config.user.snapshots_dir,
        config.infra.worktree_root,
    ]:
        if target:
            Path(target).expanduser().mkdir(parents=True, exist_ok=True)

    _write_config(target_path, config)
    console.print(f"[green]Wrote config to[/green] {target_path}")
    console.print("You can rerun `jp init` anytime to update these values.")

    if install_hooks:
        _install_precommit_hook(config.user.workspace_root)


def config_fix(ctx: typer.Context) -> None:
    """Attempt to fix a broken configuration file using the default LLM provider."""
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

    # Construct the prompt
    prompt = (
        f"The following TOML configuration file is invalid.\n"
        f"Error: {state.config_meta.error}\n\n"
        f"Content:\n```toml\n{content}\n```\n\n"
        f"Please fix the syntax errors and respond with ONLY the corrected TOML content. "
        f"Do not include any explanation or markdown formatting, just the raw TOML."
    )

    # Get the default provider
    try:
        provider = get_provider(state.config)
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        console.print("Please fix the file manually or run `jp init` to overwrite it.")
        raise typer.Exit(code=1)

    console.print(f"[dim]Using {provider.provider_type.name.lower()} provider...[/dim]")

    try:
        fixed_content = asyncio.run(_fix_config_with_provider(provider, prompt))
        path.write_text(fixed_content + "\n", encoding="utf-8")
        console.print(f"[green]Repaired[/green] {path}")
    except ProviderError as exc:
        console.print(f"[red]Provider failed to fix the configuration: {exc}[/red]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Could not extract valid TOML from response: {exc}[/red]")
        raise typer.Exit(code=1)


async def _fix_config_with_provider(provider: LLMProvider, prompt: str) -> str:
    """Use the LLM provider to fix a broken config and extract the TOML."""
    messages = [Message(role="user", content=prompt)]
    options = CompletionOptions(temperature=0.0, max_tokens=2048)

    response = await provider.complete(messages, options=options)
    raw = response.content.strip()

    # Try to extract TOML from markdown code block if present
    toml_match = re.search(r"```(?:toml)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if toml_match:
        return toml_match.group(1).strip()

    # If no code block, assume the entire response is TOML
    # Basic validation: should have at least one key = value pair
    if "=" in raw and not raw.startswith("{"):
        return raw

    raise ValueError("Response does not appear to contain valid TOML")


def _install_precommit_hook(workspace_root: Path) -> None:
    match security.validate_workspace_root(workspace_root):
        case Err(err):
            console.print(f"[red]Cannot install hooks: {err.message}[/red]")
            return
        case Ok(root):
            pass

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
