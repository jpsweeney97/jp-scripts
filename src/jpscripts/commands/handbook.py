from __future__ import annotations

from pathlib import Path
from typing import Iterable

import typer
from rich.markdown import Markdown

from jpscripts.core.console import console


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_handbook() -> str | None:
    path = _project_root() / "HANDBOOK.md"
    if not path.exists():
        console.print(f"[red]HANDBOOK.md not found at {path}[/red]")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        console.print(f"[red]Failed to read handbook: {exc}[/red]")
        return None


def _extract_section(lines: list[str], topic: str) -> str | None:
    lowered = topic.lower()
    section_start = None
    section_level = None

    for idx, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        stripped = line.lstrip()
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped.lstrip("#").strip().lower()
        if lowered in title:
            section_start = idx
            section_level = level
            break

    if section_start is None or section_level is None:
        return None

    section_lines: list[str] = []
    for line in lines[section_start:]:
        if line.startswith("#"):
            stripped = line.lstrip()
            level = len(stripped) - len(stripped.lstrip("#"))
            if level <= section_level and section_lines:
                break
        section_lines.append(line)

    return "\n".join(section_lines).strip()


def handbook(topic: str | None = typer.Argument(None, help="Optional topic to filter by heading.")) -> None:
    """Render the project handbook, optionally filtering by a topic heading."""
    content = _read_handbook()
    if not content:
        return

    lines = content.splitlines()
    if topic:
        section = _extract_section(lines, topic)
        if not section:
            console.print(f"[yellow]No handbook section found matching '{topic}'.[/yellow]")
            return
        console.print(Markdown(section))
        return

    console.print(Markdown(content))
