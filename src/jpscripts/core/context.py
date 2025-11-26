from __future__ import annotations

import ast
import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Callable

import yaml

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

HARD_FILE_CONTEXT_LIMIT = 100_000
STRUCTURED_EXTENSIONS = {".json", ".yml", ".yaml"}


async def run_and_capture(command: str, cwd: Path) -> str:
    """Run a shell command and return combined stdout/stderr."""
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (stdout + stderr).decode("utf-8", errors="replace")


def resolve_files_from_output(output: str, root: Path) -> set[Path]:
    """Parse command output for file paths that exist in the workspace."""
    found = set()

    for match in FILE_PATTERN.finditer(output):
        raw_path = match.group("path")
        clean_path = raw_path.strip(".'\"()")
        candidate = (root / clean_path).resolve()

        try:
            if candidate.is_file() and root in candidate.parents:
                found.add(candidate)
        except OSError:
            continue

    return found


async def gather_context(command: str, root: Path) -> tuple[str, set[Path]]:
    """Run a command, capture output, and find relevant files."""
    output = await run_and_capture(command, root)
    files = resolve_files_from_output(output, root)
    return output, files


def read_file_context(path: Path, max_chars: int) -> str | None:
    """
    Read file content safely and truncate to max_chars.
    Returns None on any read/encoding error.
    """
    limit = max(0, min(max_chars, HARD_FILE_CONTEXT_LIMIT))
    try:
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read(limit)
    except (OSError, UnicodeDecodeError):
        return None
    return text


def smart_read_context(path: Path, max_chars: int) -> str:
    """Read files with syntax-aware truncation to keep output parsable."""
    limit = max(0, min(max_chars, HARD_FILE_CONTEXT_LIMIT))
    if limit == 0:
        return ""

    text = _read_text_for_context(path)
    if text is None:
        return ""
    if len(text) <= limit:
        return text

    suffix = path.suffix.lower()
    if suffix == ".py":
        return _truncate_python_source(text, limit)
    if suffix in STRUCTURED_EXTENSIONS:
        loader = json.loads if suffix == ".json" else yaml.safe_load
        return _truncate_structured_text(text, limit, loader)
    return text[:limit]


def _read_text_for_context(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return fh.read(HARD_FILE_CONTEXT_LIMIT)
    except (OSError, UnicodeDecodeError):
        return None


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _nearest_line_boundary(offsets: list[int], limit: int) -> int:
    for idx in range(len(offsets) - 1, -1, -1):
        if offsets[idx] <= limit:
            return offsets[idx]
    return 0


def _is_parseable(snippet: str) -> bool:
    try:
        ast.parse(snippet)
    except SyntaxError:
        return False
    return True


def _find_definition_boundary(tree: ast.AST, offsets: list[int], limit: int) -> int:
    boundary = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end_line = getattr(node, "end_lineno", None)
            if end_line is None:
                continue
            offset_index = min(end_line, len(offsets) - 1)
            end_offset = offsets[offset_index]
            if boundary < end_offset <= limit:
                boundary = end_offset
    return boundary


def _truncate_to_parseable(text: str, limit: int, offsets: list[int]) -> str:
    for idx in range(len(offsets) - 1, -1, -1):
        pos = offsets[idx]
        if pos == 0 or pos > limit:
            continue
        snippet = text[:pos]
        if not snippet.endswith("\n"):
            snippet = f"{snippet}\n"
        if _is_parseable(snippet):
            return snippet

    return ""


def _truncate_python_source(text: str, limit: int) -> str:
    offsets = _line_offsets(text)
    limit_offset = min(limit, offsets[-1])

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _truncate_to_parseable(text, limit_offset, offsets)

    candidates: list[int] = []

    boundary = _find_definition_boundary(tree, offsets, limit_offset)
    if boundary:
        candidates.append(boundary)

    candidates.append(_nearest_line_boundary(offsets, limit_offset))

    for pos in sorted(set(candidates), reverse=True):
        snippet = text[:pos]
        if not snippet.endswith("\n"):
            snippet = f"{snippet}\n"
        if _is_parseable(snippet):
            return snippet

    return _truncate_to_parseable(text, limit_offset, offsets)


def _validate_structured_prefix(candidate: str, loader: Callable[[str], object]) -> str | None:
    lines = candidate.splitlines()
    while lines:
        text = "\n".join(lines)
        try:
            loader(text)
            return text
        except Exception:
            lines.pop()
    return None


def _truncate_structured_text(text: str, limit: int, loader: Callable[[str], object]) -> str:
    lines = text.splitlines()
    snippet_lines: list[str] = []
    total = 0
    for line in lines:
        line_with_newline = f"{line}\n"
        if total + len(line_with_newline) > limit:
            break
        snippet_lines.append(line)
        total += len(line_with_newline)

    candidate = "\n".join(snippet_lines)
    validated = _validate_structured_prefix(candidate, loader)
    return validated if validated is not None else ""
