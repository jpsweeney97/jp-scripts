"""System diagnostics and health checks.

Provides diagnostic checks for system dependencies:
    - External tool availability (git, fzf, rg, etc.)
    - Memory store health
    - Configuration validation
    - Workspace integrity
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from jpscripts.core import memory as memory_core
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Err
from jpscripts.core.security import WorkspaceValidationError, validate_workspace_root


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

        for label, path in (
            ("workspace_root", self.config.workspace_root),
            ("notes_dir", self.config.notes_dir),
        ):
            expanded = path.expanduser()
            if not expanded.exists():
                issues.append(f"{label} missing: {expanded}")
            elif not os.access(expanded, os.W_OK):
                issues.append(f"{label} not writable: {expanded}")
            else:
                try:
                    if label == "workspace_root":
                        validate_workspace_root(expanded)
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
            deps = memory_core._load_lancedb_dependencies()  # pyright: ignore[reportPrivateUsage]
            if deps is None:
                return "warn", "lancedb not installed; vector memory unavailable."

            lancedb_module, lance_model_base = deps
            store = memory_core.LanceDBStore(store_path, lancedb_module, lance_model_base)
            probe = store.search([0.0], limit=1)
            if isinstance(probe, Err):
                return "error", f"Vector DB check failed: {probe.error}"
            return "ok", f"LanceDB ready at {store_path}"
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
    ExternalTool(
        name="Git", binary="git", install_hint="Install via your package manager (brew, apt, etc.)"
    ),
    ExternalTool(
        name="ripgrep",
        binary="rg",
        install_hint="Install via your package manager (brew, apt, etc.)",
    ),
    ExternalTool(
        name="fzf", binary="fzf", install_hint="Install via your package manager (brew, apt, etc.)"
    ),
    ExternalTool(
        name="GitHub CLI",
        binary="gh",
        install_hint="Install via your package manager (brew, apt, etc.)",
    ),
    ExternalTool(name="Codex", binary="codex", install_hint="npm install -g @openai/codex"),
    ExternalTool(
        name="Python",
        binary="python3",
        install_hint="Install via your package manager (brew, apt, etc.)",
    ),
    ExternalTool(name="Homebrew", binary="brew", install_hint="macOS: https://brew.sh"),
    ExternalTool(
        name="System Clipboard",
        binary="pbcopy",
        install_hint="macOS: Built-in. Linux: Install xclip/xsel.",
        required=False,
    ),
    ExternalTool(name="SwitchAudioSource", binary="SwitchAudioSource", required=False),
    ExternalTool(
        name="zoxide",
        binary="zoxide",
        install_hint="Install via your package manager (brew, apt, etc.)",
        required=False,
    ),
]


def _select_tools(names: Iterable[str] | None) -> list[ExternalTool]:
    if not names:
        return DEFAULT_TOOLS

    requested = {name.lower() for name in names}
    selected = [
        tool
        for tool in DEFAULT_TOOLS
        if tool.name.lower() in requested or tool.binary.lower() in requested
    ]
    return selected or DEFAULT_TOOLS


async def _check_tool(tool: ExternalTool) -> ToolCheck:
    resolved = shutil.which(tool.binary)
    if not resolved:
        return ToolCheck(tool=tool, status="missing", version=None, message=tool.install_hint)

    try:
        process = await asyncio.create_subprocess_exec(
            resolved,
            *tool.version_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return ToolCheck(tool=tool, status="missing", version=None, message=tool.install_hint)

    stdout, stderr = await process.communicate()
    output = (stdout or b"").decode().strip() or (stderr or b"").decode().strip()
    version = output.splitlines()[0] if output else None

    if process.returncode != 0:
        return ToolCheck(
            tool=tool, status="error", version=version, message=output or "version command failed"
        )

    return ToolCheck(tool=tool, status="ok", version=version, message=None)


async def _run_doctor(tools: list[ExternalTool]) -> list[ToolCheck]:
    tasks = [asyncio.create_task(_check_tool(tool)) for tool in tools]
    return await asyncio.gather(*tasks)


async def _run_deep_checks(
    config: AppConfig, config_path: Path | None
) -> list[tuple[str, str, str]]:
    diag_checks: list[DiagnosticCheck] = [
        ConfigCheck(config, config_path),
        AuthCheck(config),
        VectorDBCheck(config),
        MCPCheck(),
    ]
    results = await asyncio.gather(*(check.run() for check in diag_checks))
    return [
        (check.name, status, message)
        for check, (status, message) in zip(diag_checks, results, strict=True)
    ]


async def run_diagnostics_suite(
    config: AppConfig,
    config_path: Path | None,
    tool_names: list[str] | None,
) -> tuple[list[tuple[str, str, str]], list[ToolCheck]]:
    """Run deep checks and tool checks in parallel."""
    tools = _select_tools(tool_names)
    deep_checks_task = asyncio.create_task(_run_deep_checks(config, config_path))
    tools_task = asyncio.create_task(_run_doctor(tools))
    return await asyncio.gather(deep_checks_task, tools_task)
