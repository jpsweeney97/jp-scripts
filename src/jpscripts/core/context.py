from __future__ import annotations

import ast
import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped]
from rich.console import Console

from jpscripts.core.console import get_logger

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

# Hard safety cap to prevent excessive memory usage when reading files for context.
HARD_CONTEXT_CAP = 500_000
STRUCTURED_EXTENSIONS = {".json", ".yml", ".yaml"}
SYNTAX_WARNING = "# [WARN] Syntax error detected. AST features disabled.\n"

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator using ~4 characters per token."""
    return max(0, int(len(text) / 4))


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


def _warn_out_of_workspace_paths(command: str, root: Path) -> None:
    try:
        workspace_root = root.resolve()
    except OSError:
        workspace_root = root

    candidates = set(re.findall(r"/[^\s\"']+", command))
    outside: set[str] = set()
    for raw in candidates:
        normalized = raw.strip(",;")
        try:
            candidate_path = Path(normalized).resolve()
        except OSError:
            continue
        if not candidate_path.is_absolute():
            continue
        try:
            candidate_path.relative_to(workspace_root)
        except ValueError:
            outside.add(str(candidate_path))

    if outside:
        Console(stderr=True).print(
            "[yellow]Warning:[/yellow] command references paths outside workspace: "
            f"{', '.join(sorted(outside))}"
        )


async def gather_context(command: str, root: Path) -> tuple[str, set[Path]]:
    """Run a command, capture output, and find relevant files."""
    _warn_out_of_workspace_paths(command, root)
    output = await run_and_capture(command, root)
    files = resolve_files_from_output(output, root)
    return output, files


def read_file_context(path: Path, max_chars: int) -> str | None:
    """
    Read file content safely and truncate to max_chars.
    Returns None on any read/encoding error.
    """
    limit = max(0, min(max_chars, HARD_CONTEXT_CAP))
    try:
        estimated_tokens = int(path.stat().st_size / 4)
        if estimated_tokens > 10_000:
            logger.warning(
                "File %s estimated at %d tokens; context may be truncated.",
                path,
                estimated_tokens,
            )
    except OSError:
        pass
    try:
        with path.open("r", encoding="utf-8") as fh:
            text = fh.read(limit)
    except (OSError, UnicodeDecodeError):
        return None
    return text


def smart_read_context(path: Path, max_chars: int, max_tokens: int | None = None) -> str:
    """Read files with syntax-aware truncation to keep output parsable."""
    limits: list[int] = [max_chars, HARD_CONTEXT_CAP]
    if max_tokens is not None:
        limits.append(max(0, max_tokens * 4))
    limit = max(0, min(limits))
    if limit == 0:
        return ""

    text = _read_text_for_context(path)
    if text is None:
        return ""

    suffix = path.suffix.lower()
    if suffix == ".py":
        skeleton = get_file_skeleton(path)
        return skeleton[:limit]
    if len(text) <= limit:
        return text
    if suffix == ".json":
        return _truncate_json(text, limit)
    if suffix in {".yaml", ".yml"}:
        return _truncate_structured_text(text, limit, yaml.safe_load)
    return text[:limit]


def get_file_skeleton(path: Path) -> str:
    """Return a high-level AST skeleton of a Python file.

    The skeleton preserves imports, module-level assignments, class definitions,
    function signatures, and docstrings. Function and method bodies are replaced
    with ``pass`` (or ellipsis) when they span 5 or more lines; shorter bodies are
    preserved. Falls back to a line-based truncation on syntax errors.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""

    def _node_length(node: ast.AST) -> int:
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        return max(end - start + 1, 0)

    def _doc_expr(raw: str | None) -> ast.Expr | None:
        if raw is None:
            return None
        return ast.Expr(value=ast.Constant(value=raw))

    def _skeletonize_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        if _node_length(node) < 5:
            return node
        doc_expr = _doc_expr(ast.get_docstring(node, clean=False))
        body: list[ast.stmt] = []
        if doc_expr:
            body.append(doc_expr)
        body.append(ast.Pass())
        new_node = type(node)(
            name=node.name,
            args=node.args,
            body=body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=getattr(node, "type_comment", None),
        )
        return ast.copy_location(new_node, node)

    def _skeletonize_class(node: ast.ClassDef) -> ast.ClassDef:
        doc_expr = _doc_expr(ast.get_docstring(node, clean=False))
        new_body: list[ast.stmt] = []
        if doc_expr:
            new_body.append(doc_expr)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_body.append(_skeletonize_function(child))
            elif isinstance(child, ast.ClassDef):
                new_body.append(_skeletonize_class(child))
            elif isinstance(child, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
                new_body.append(child)
        if not new_body:
            new_body.append(ast.Pass())
        new_node = ast.ClassDef(
            name=node.name,
            bases=node.bases,
            keywords=node.keywords,
            body=new_body,
            decorator_list=node.decorator_list,
        )
        return ast.copy_location(new_node, node)

    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        offsets = _line_offsets(source)
        limit_offset = min(HARD_CONTEXT_CAP, offsets[-1] if offsets else HARD_CONTEXT_CAP)
        return _fallback_read(source, limit_offset, exc)

    new_body: list[ast.stmt] = []
    module_doc = ast.get_docstring(module, clean=False)
    if module_doc:
        new_body.append(ast.Expr(value=ast.Constant(value=module_doc)))

    for stmt in module.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            new_body.append(stmt)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_body.append(_skeletonize_function(stmt))
        elif isinstance(stmt, ast.ClassDef):
            new_body.append(_skeletonize_class(stmt))

    if not new_body:
        return source[:HARD_CONTEXT_CAP]

    lines: list[str] = []
    for stmt in new_body:
        try:
            lines.append(ast.unparse(ast.fix_missing_locations(stmt)))
        except Exception:
            continue

    if not lines:
        return source[:HARD_CONTEXT_CAP]

    return "\n\n".join(lines)[:HARD_CONTEXT_CAP]


def _read_text_for_context(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return fh.read(HARD_CONTEXT_CAP)
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

def _fallback_read(text: str, limit: int, error: SyntaxError | None) -> str:
    warning = SYNTAX_WARNING if error else ""
    if limit <= 0:
        return warning[:limit] if warning else ""

    if warning and limit <= len(warning):
        return warning[:limit]

    budget = limit - len(warning)
    lines = text.splitlines()
    if not lines:
        return warning.strip()

    head_budget = int(budget * 0.6)
    tail_budget = budget - head_budget

    head_lines: list[str] = []
    head_used = 0
    for line in lines:
        next_len = len(line) + 1
        if head_used + next_len > head_budget:
            break
        head_lines.append(line)
        head_used += next_len

    tail_lines: list[str] = []
    tail_used = 0
    for line in reversed(lines[len(head_lines) :]):
        next_len = len(line) + 1
        if tail_used + next_len > tail_budget:
            break
        tail_lines.append(line)
        tail_used += next_len
    tail_lines.reverse()

    middle: list[str] = []
    lineno = getattr(error, "lineno", None) if error else None
    if lineno is not None:
        idx = max(int(lineno) - 1, 0)
        if idx >= len(head_lines) and idx < len(lines) - len(tail_lines):
            start = max(idx - 3, len(head_lines))
            end = min(idx + 4, len(lines) - len(tail_lines))
            middle = ["# ... error context ...", *lines[start:end]]

    parts: list[str] = []
    parts.extend(head_lines)
    if middle:
        parts.extend(middle)
    if tail_lines:
        parts.append("# ... trailing context ...")
        parts.extend(tail_lines)

    body = "\n".join(parts) or text[:budget]
    snippet = f"{warning}{body}"
    if len(snippet) > limit:
        return snippet[:limit]
    return snippet


def _truncate_python_source(text: str, limit: int) -> str:
    offsets = _line_offsets(text)
    limit_offset = min(limit, offsets[-1])

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return _fallback_read(text, limit_offset, exc)

    if len(text) <= limit:
        return text

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

    truncated = _truncate_to_parseable(text, limit_offset, offsets)
    if truncated:
        return truncated

    return _fallback_read(text, limit_offset, None)


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


def _truncate_json(text: str, limit: int) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _truncate_structured_text(text, limit, json.loads)

    def _serialized_length(obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False))

    def _shrink(value: Any) -> Any:
        if _serialized_length(value) <= limit:
            return value
        if isinstance(value, dict):
            pruned: dict[str, Any] = {}
            for key, val in value.items():
                candidate = _shrink(val)
                pruned[key] = candidate
                if _serialized_length(pruned) > limit:
                    pruned.pop(key, None)
                    break
            return pruned
        if isinstance(value, list):
            pruned_list: list[Any] = []
            for item in value:
                candidate = _shrink(item)
                pruned_list.append(candidate)
                if _serialized_length(pruned_list) > limit:
                    pruned_list.pop()
                    break
            return pruned_list
        if isinstance(value, str):
            return value[: max(limit // 4, 0)]
        return value

    pruned_value = _shrink(data)
    serialized = json.dumps(pruned_value, ensure_ascii=False)
    if len(serialized) <= limit:
        return serialized

    if isinstance(pruned_value, list):
        while pruned_value and len(json.dumps(pruned_value, ensure_ascii=False)) > limit:
            pruned_value.pop()
    elif isinstance(pruned_value, dict):
        for key in list(pruned_value.keys())[::-1]:
            pruned_value.pop(key, None)
            if len(json.dumps(pruned_value, ensure_ascii=False)) <= limit:
                break

    serialized = json.dumps(pruned_value, ensure_ascii=False)
    if len(serialized) > limit:
        minimal = "{}" if isinstance(pruned_value, dict) else "[]"
        return minimal[:limit]
    return serialized


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
