from __future__ import annotations

from pathlib import Path

import pathspec
import pyperclip
import typer

from jpscripts.core.console import console
from jpscripts.core.git import is_repo


def load_gitignore(root: Path) -> pathspec.PathSpec:
    lines: list[str] = []
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines.extend(gitignore.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _estimate_tokens(text: str) -> int:
    # Rule of thumb: ~4 chars per token for code
    return len(text) // 4


def repo_map(
    ctx: typer.Context,
    max_lines: int = typer.Option(500, "--max-lines", "-n", help="Max lines per file before truncation"),
) -> None:
    """Pack the current repo into an LLM-friendly XML format (CDATA wrapped)."""
    root = Path.cwd()

    if not is_repo(root):
        console.print("[yellow]Not a git repo. Packing all files...[/yellow]")

    spec = load_gitignore(root)
    output: list[str] = []
    file_count = 0
    total_tokens = 0

    console.print(f"[cyan]Packing repository context from[/cyan] {root} ...")

    for path in root.rglob("*"):
        if path.is_dir():
            continue

        rel_path = path.relative_to(root)
        rel_str = str(rel_path)

        if ".git" in rel_str.split("/") or spec.match_file(rel_str):
            continue

        try:
            content_raw = path.read_text(encoding="utf-8")
            lines = content_raw.splitlines()

            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n... [truncated {len(lines) - max_lines} lines] ..."
            else:
                content = content_raw

            # CDATA wrapping prevents XML injection from code symbols like '<' or '&'
            xml_entry = f'<file path="{rel_path}">\n<![CDATA[\n{content}\n]]>\n</file>'
            output.append(xml_entry)
            file_count += 1
            total_tokens += _estimate_tokens(content)

        except UnicodeDecodeError:
            continue
        except Exception as e:
            console.print(f"[red]Error reading {rel_path}: {e}[/red]")

    final_payload = "\n".join(output)
    pyperclip.copy(final_payload)

    console.print(f"[green]Packed {file_count} files (~{total_tokens} tokens) to clipboard.[/green]")
    console.print("Ready to paste into ChatGPT/Gemini.")
