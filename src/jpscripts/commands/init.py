from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import typer
from rich.prompt import Confirm, Prompt

from jpscripts.core.console import console
from jpscripts.core.config import AppConfig


def _write_config(path: Path, config: AppConfig) -> None:
    content = dedent(
        f"""
        # jpscripts configuration (TOML)
        editor = "{config.editor}"
        notes_dir = "{config.notes_dir}"
        workspace_root = "{config.workspace_root}"
        snapshots_dir = "{config.snapshots_dir}"
        log_level = "{config.log_level}"
        worktree_root = "{config.worktree_root or ''}"
        focus_audio_device = "{config.focus_audio_device or ''}"
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")


def init(ctx: typer.Context, config_path: Path | None = typer.Option(None, help="Where to write config.")) -> None:
    """Interactive initializer that writes the active config file."""
    state = ctx.obj
    defaults: AppConfig = state.config
    target_path = (config_path or state.config_meta.path).expanduser()

    notes_dir = Path(Prompt.ask("Notes directory", default=str(defaults.notes_dir)))
    workspace_root = Path(Prompt.ask("Workspace root", default=str(defaults.workspace_root)))
    worktree_root_input = Prompt.ask("Worktree root (optional)", default=str(defaults.worktree_root or ""))
    worktree_root = Path(worktree_root_input).expanduser() if worktree_root_input else None
    editor = Prompt.ask("Editor command", default=defaults.editor)
    log_level = Prompt.ask("Log level", default=defaults.log_level)
    snapshots_dir = Path(Prompt.ask("Snapshots directory", default=str(defaults.snapshots_dir)))
    focus_audio_device = Prompt.ask(
        "Preferred audio device (optional)", default=defaults.focus_audio_device or ""
    ).strip() or None

    config = AppConfig(
        editor=editor,
        notes_dir=notes_dir,
        workspace_root=workspace_root,
        snapshots_dir=snapshots_dir,
        log_level=log_level,
        worktree_root=worktree_root,
        focus_audio_device=focus_audio_device,
    )

    for target in [config.notes_dir, config.workspace_root, config.snapshots_dir, config.worktree_root]:
        if target:
            Path(target).expanduser().mkdir(parents=True, exist_ok=True)

    _write_config(target_path, config)
    console.print(f"[green]Wrote config to[/green] {target_path}")
    console.print("You can rerun `jp init` anytime to update these values.")
