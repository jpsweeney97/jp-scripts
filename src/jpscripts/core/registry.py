from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import typer

logger = logging.getLogger(__name__)


class CommandModule(Protocol):
    app: typer.Typer


@dataclass
class CommandSpec:
    name: str
    handler: Callable[..., None]


_FUNCTION_COMMANDS: dict[str, dict[str, str]] = {
    "git_ops": {"status-all": "status_all", "whatpush": "whatpush", "sync": "sync"},
    "nav": {"recent": "recent", "proj": "proj"},
    "init": {"init": "init", "config-fix": "config_fix"},
    "web": {"web-snap": "web_snap"},
    "system": {
        "process-kill": "process_kill",
        "port-kill": "port_kill",
        "brew-explorer": "brew_explorer",
        "audioswap": "audioswap",
        "ssh-open": "ssh_open",
        "tmpserver": "tmpserver",
        "update": "update",
    },
    "notes": {
        "note": "note",
        "note-search": "note_search",
        "standup": "standup",
        "standup-note": "standup_note",
        "cliphist": "cliphist",
    },
    "map": {"map": "map_cmd", "repo-map": "map_cmd"},
    "search": {"ripper": "ripper", "todo-scan": "todo_scan", "loggrep": "loggrep"},
    "git_extra": {
        "gundo-last": "gundo_last",
        "gstage": "gstage",
        "gpr": "gpr",
        "gbrowse": "gbrowse",
        "git-branchcheck": "git_branchcheck",
        "stashview": "stashview",
    },
    "agent": {"fix": "codex_exec", "agent": "codex_exec"},
    "handbook": {"handbook": "handbook"},
}


def _module_name(path: Path) -> str:
    return path.stem


def _import_module(module_name: str) -> object | None:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to import command module %s: %s", module_name, exc)
        return None


def _build_function_commands(module_name: str, module: object) -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    mapping = _FUNCTION_COMMANDS.get(module_name, {})
    for cmd_name, attr in mapping.items():
        handler = getattr(module, attr, None)
        if callable(handler):
            specs.append(CommandSpec(name=cmd_name, handler=handler))
        else:  # pragma: no cover - defensive
            logger.error("Command %s.%s not found or not callable", module_name, attr)
    return specs


def discover_commands(package_path: Path, package: str = "jpscripts.commands") -> tuple[list[tuple[str, CommandModule]], list[CommandSpec]]:
    """
    Discover Typer command modules and standalone command callables.

    Returns:
        A tuple of (typer_modules, function_commands).
    """
    typer_modules: list[tuple[str, CommandModule]] = []
    function_commands: list[CommandSpec] = []

    for file in package_path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        module_name = _module_name(file)
        if module_name == "__init__":
            continue
        import_path = f"{package}.{module_name}"
        module = _import_module(import_path)
        if module is None:
            continue

        if module_name in _FUNCTION_COMMANDS:
            function_commands.extend(_build_function_commands(module_name, module))
            continue

        app = getattr(module, "app", None)
        if isinstance(app, typer.Typer):
            typer_modules.append((module_name.replace("_", "-"), module))  # type: ignore[arg-type]
            continue

        function_commands.extend(_build_function_commands(module_name, module))

    return typer_modules, function_commands
