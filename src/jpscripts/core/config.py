from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_ENV_VAR = "JPSCRIPTS_CONFIG"

ENV_VAR_MAP: dict[str, str] = {
    "editor": "JP_EDITOR",
    "notes_dir": "JP_NOTES_DIR",
    "workspace_root": "JP_WORKSPACE_ROOT",
    "snapshots_dir": "JP_SNAPSHOTS_DIR",
    "log_level": "JP_LOG_LEVEL",
    "worktree_root": "JP_WORKTREE_ROOT",
    "focus_audio_device": "JP_FOCUS_AUDIO_DEVICE",
}


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


@dataclass
class ConfigLoadResult:
    path: Path
    file_loaded: bool
    env_overrides: set[str]
    error: str | None = None  # NEW: Capture error instead of crashing


class AppConfig(BaseModel):
    """Application-wide configuration modeled with Pydantic."""

    model_config = ConfigDict(extra="ignore")

    editor: str = Field(default="code -w", description="Editor command used for interactive edits.")
    notes_dir: Path = Field(default_factory=lambda: Path.home() / "Notes" / "quick-notes")
    workspace_root: Path = Field(default_factory=lambda: Path.home() / "Projects")
    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            ".venv",
            "__pycache__",
            "dist",
            "build",
            ".idea",
            ".vscode",
        ],
        description="Directory names to ignore when scanning for recent files.",
    )
    snapshots_dir: Path = Field(default_factory=lambda: Path.home() / "snapshots")
    log_level: str = Field(default="INFO", description="Log level for jp output.")
    worktree_root: Path | None = Field(default=None, description="Optional location for Git worktrees.")
    focus_audio_device: str | None = Field(default=None, description="Preferred audio device for focus helpers.")


def _resolve_config_path(config_path: Path | None, env_vars: Mapping[str, str]) -> Path:
    candidate = config_path or env_vars.get(CONFIG_ENV_VAR) or (Path.home() / ".jpconfig")
    return Path(candidate).expanduser()


def _read_env_overrides(env_vars: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for field, env_key in ENV_VAR_MAP.items():
        if env_key in env_vars:
            overrides[field] = env_vars[env_key]
    return overrides


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    parser = json.loads if suffix == ".json" else tomllib.loads

    try:
        data = parser(raw)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        # Re-raise as ConfigError to be caught by load_config
        raise ConfigError(f"Syntax error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config root in {path} must be a mapping.")

    return data


def load_config(
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[AppConfig, ConfigLoadResult]:
    """
    Load configuration with Safe Mode fallback.
    If the file is invalid, returns default config + error message.
    """
    env_vars: Mapping[str, str] = env or os.environ
    resolved_path = _resolve_config_path(config_path, env_vars)
    env_overrides = _read_env_overrides(env_vars)

    try:
        file_data = _read_config_file(resolved_path)
        merged = {**file_data, **env_overrides}
        config = AppConfig.model_validate(merged)
        error = None
        file_loaded = resolved_path.exists()
    except (ConfigError, ValidationError) as exc:
        # Safe Mode: Fallback to defaults + env vars only
        config = AppConfig.model_validate(env_overrides)
        error = str(exc)
        file_loaded = False

    load_result = ConfigLoadResult(
        path=resolved_path,
        file_loaded=file_loaded,
        env_overrides=set(env_overrides.keys()),
        error=error,
    )

    return config, load_result
