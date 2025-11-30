from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from jpscripts.core.security import WorkspaceValidationError, validate_workspace_root

CONFIG_ENV_VAR = "JPSCRIPTS_CONFIG"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


@dataclass
class ConfigLoadResult:
    path: Path
    file_loaded: bool
    env_overrides: set[str]
    error: str | None = None


class AppConfig(BaseSettings):
    """Application-wide configuration modeled with Pydantic Settings."""

    model_config = SettingsConfigDict(
        env_prefix="JP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    editor: str = Field(default="code -w", description="Editor command used for interactive edits.")
    notes_dir: Path = Field(default_factory=lambda: Path.home() / "Notes" / "quick-notes")
    workspace_root: Path = Field(default_factory=lambda: Path.home() / "Projects")
    default_model: str = Field(default="claude-opus-4-5", description="Default Codex/LLM model.")
    memory_store: Path = Field(
        default_factory=lambda: Path.home() / ".jp_memory.sqlite",
        description="Path to the memory store file.",
    )
    memory_model: str = Field(
        default="all-MiniLM-L6-v2", description="Embedding model for semantic memory search."
    )
    embedding_server_url: str | None = Field(
        default=None,
        description="Preferred local embedding HTTP endpoint, used before loading local weights.",
    )
    use_semantic_search: bool = Field(
        default=True, description="Enable semantic search with embeddings."
    )
    model_context_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "claude-opus-4-5": 200_000,
            "gpt-4-turbo": 32_000,
            "default": 50_000,
        },
        description="Per-model soft context limits used for prompt construction.",
    )
    max_file_context_chars: int = Field(
        default=50000, description="Maximum characters to read when attaching file context."
    )
    max_command_output_chars: int = Field(
        default=20000, description="Maximum characters from captured command output for prompts."
    )
    git_status_timeout: float = Field(default=5.0, description="Timeout for gathering git context.")
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
    worktree_root: Path | None = Field(
        default=None, description="Optional location for Git worktrees."
    )
    focus_audio_device: str | None = Field(
        default=None, description="Preferred audio device for focus helpers."
    )
    dry_run: bool = Field(
        default=False, description="If true, performs dry-run operations without side effects."
    )
    trace_dir: Path = Field(
        default_factory=lambda: Path.home() / ".jpscripts" / "traces",
        description="Directory for agent trace logs.",
    )
    use_docker_sandbox: bool = Field(
        default=False, description="Execute safe shell commands inside a Docker sandbox."
    )
    docker_image: str = Field(
        default="python:3.11-slim", description="Docker image used when sandboxing commands."
    )
    otel_endpoint: str | None = Field(
        default=None, description="OTLP endpoint for exporting traces."
    )
    otel_service_name: str = Field(
        default="jpscripts", description="Service name used for OpenTelemetry spans."
    )
    otel_export_enabled: bool = Field(
        default=False, description="Enable OTLP tracing export when true."
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
    ]:
        # Ensure environment variables override config file entries.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def _resolve_config_path(config_path: Path | None, env_vars: Mapping[str, str]) -> Path:
    candidate = config_path or env_vars.get(CONFIG_ENV_VAR) or (Path.home() / ".jpconfig")
    return Path(candidate).expanduser()


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    parser = json.loads if suffix == ".json" else tomllib.loads

    try:
        data = parser(raw)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Syntax error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config root in {path} must be a mapping.")

    return data


def _detect_env_overrides(env_vars: Mapping[str, str]) -> set[str]:
    prefix = AppConfig.model_config.get("env_prefix", "")
    overrides: set[str] = set()
    for field in AppConfig.model_fields:
        env_key = f"{prefix}{field}".upper()
        if env_key in env_vars:
            overrides.add(field)
    return overrides


def load_config(
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[AppConfig, ConfigLoadResult]:
    """
    Load configuration with Safe Mode fallback.
    If the file is invalid, returns default config + error message.
    """
    env_vars: Mapping[str, str] = os.environ if env is None else {**os.environ, **env}
    resolved_path = _resolve_config_path(config_path, env_vars)
    env_overrides = _detect_env_overrides(env_vars)

    error: str | None = None
    file_loaded = False
    file_data: dict[str, Any] = {}

    try:
        file_data = _read_config_file(resolved_path)
        file_loaded = resolved_path.exists()
    except ConfigError as exc:
        error = str(exc)

    context_manager = (
        patch.dict(os.environ, env_vars, clear=False) if env is not None else nullcontext()
    )

    try:
        with context_manager:
            config = AppConfig(**file_data)
    except ValidationError as exc:
        error = str(exc)
        config = AppConfig()

    try:
        resolved_root = validate_workspace_root(config.workspace_root)
        config = config.model_copy(update={"workspace_root": resolved_root})
    except WorkspaceValidationError as exc:
        error = f"{error}; {exc}" if error else str(exc)

    load_result = ConfigLoadResult(
        path=resolved_path,
        file_loaded=file_loaded,
        env_overrides=env_overrides,
        error=error,
    )

    return config, load_result
