"""Context gathering for AI agent prompts.

Collects and processes context from various sources:
    - Command output capture
    - File content reading with token limits
    - Code skeleton extraction
    - Sensitive data redaction
"""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import json
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from rich.console import Console

from jpscripts.analysis import structure
from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.console import get_logger
from jpscripts.ai.tokens import DEFAULT_MODEL_CONTEXT_LIMIT as _TOKEN_DEFAULT

logger = get_logger(__name__)

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

# Default model context limit (can be overridden per-model at runtime)
DEFAULT_MODEL_CONTEXT_LIMIT = _TOKEN_DEFAULT
STRUCTURED_EXTENSIONS = {".json", ".yml", ".yaml"}
SYNTAX_WARNING = "# [WARN] Syntax error detected. AST features disabled.\n"
SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "id_rsa"]


class GatherContextResult(BaseModel):
    """Structured result for context gathering."""

    output: str
    files: set[Path]
    ordered_files: tuple[Path, ...] = ()

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def __iter__(self) -> Any:
        return iter((self.output, self.files))


async def run_and_capture(command: str, cwd: Path) -> str:
    """Run a command without shell interpolation and return combined stdout/stderr."""
    # Security: Validate command before execution
    verdict, reason = validate_command(command, cwd)
    if verdict != CommandVerdict.ALLOWED:
        logger.warning("Command blocked by security policy: %s (reason: %s)", command, reason)
        return f"[SECURITY BLOCK] Command rejected: {reason}"

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        logger.warning("Failed to parse context command: %s", exc)
        return f"Unable to parse command; simplify quoting. ({exc})"

    if not tokens:
        return "Invalid command."

    logger.debug("Executing context command: %s", tokens)
    process = await asyncio.create_subprocess_exec(
        *tokens,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (stdout + stderr).decode("utf-8", errors="replace")


def _batch_check_files(paths: list[Path], workspace_root: Path) -> dict[Path, bool]:
    """Check which paths are files within the workspace. Returns path -> is_valid mapping.

    Runs all stat calls in a single thread to minimize overhead.
    Uses a local cache to avoid repeated stat calls on the same path.
    """
    result: dict[Path, bool] = {}
    for path in paths:
        if path in result:
            continue
        try:
            if path.is_file() and workspace_root in path.parents:
                result[path] = True
            else:
                result[path] = False
        except OSError:
            result[path] = False
    return result


async def resolve_files_from_output(output: str, root: Path) -> tuple[list[Path], set[Path]]:
    """Parse command output for file paths that exist in the workspace and their dependencies.

    Optimized to batch stat calls and avoid redundant file checks.
    """
    try:
        workspace_root = root.resolve()
    except OSError:
        workspace_root = root

    # Phase 1: Collect all candidate paths from output (no I/O yet)
    raw_candidates: list[Path] = []
    for match in FILE_PATTERN.finditer(output):
        raw_path = match.group("path")
        clean_path = raw_path.strip(".'\"()")
        try:
            candidate = (workspace_root / clean_path).resolve()
            raw_candidates.append(candidate)
        except OSError:
            continue

    # Phase 2: Batch check which files exist (single threaded batch)
    file_status = await asyncio.to_thread(_batch_check_files, raw_candidates, workspace_root)

    # Build found set and ordered list from batch results
    found: set[Path] = set()
    ordered: list[Path] = []
    python_candidates: set[Path] = set()

    for candidate in raw_candidates:
        if file_status.get(candidate, False):
            if candidate not in found:
                ordered.append(candidate)
            found.add(candidate)
            if candidate.suffix.lower() == ".py":
                python_candidates.add(candidate)

    # Phase 3: Get dependencies for Python files (already cached in structure module)
    all_deps: set[Path] = set()
    for path in python_candidates:
        try:
            deps = await asyncio.to_thread(structure.get_import_dependencies, path, workspace_root)
            for dep in deps:
                try:
                    all_deps.add(dep.resolve())
                except OSError:
                    continue
        except Exception as exc:
            logger.debug("Dependency discovery failed for %s: %s", path, exc)
            continue

    # Phase 4: Batch check dependency files (reusing cache for any already checked)
    dep_paths = [dep for dep in all_deps if dep not in file_status]
    if dep_paths:
        dep_status = await asyncio.to_thread(_batch_check_files, dep_paths, workspace_root)
        file_status.update(dep_status)

    # Add valid dependencies to result
    dependencies: set[Path] = set()
    for dep in all_deps:
        if file_status.get(dep, False):
            dependencies.add(dep)
            if dep not in found:
                ordered.append(dep)

    return ordered, found | dependencies


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


async def gather_context(command: str, root: Path) -> GatherContextResult:
    """Run a command, capture output, and find relevant files."""
    _warn_out_of_workspace_paths(command, root)
    output = await run_and_capture(command, root)
    ordered, files = await resolve_files_from_output(output, root)
    return GatherContextResult(output=output, files=files, ordered_files=tuple(ordered))


def read_file_context(
    path: Path, max_chars: int, *, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
) -> str | None:
    """
    Read file content safely and truncate to max_chars.
    Returns None on any read/encoding error.

    Args:
        path: File path to read.
        max_chars: Maximum characters to return.
        limit: Hard cap for safety (default: DEFAULT_MODEL_CONTEXT_LIMIT).
    """
    effective_limit = max(0, min(max_chars, limit))
    for pattern in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(path.name, pattern):
            logger.warning("Blocked sensitive file read: %s", path)
            return None
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
            text = fh.read(effective_limit)
    except (OSError, UnicodeDecodeError):
        return None
    return text


def smart_read_context(
    path: Path,
    max_chars: int,
    max_tokens: int | None = None,
    *,
    limit: int = DEFAULT_MODEL_CONTEXT_LIMIT,
) -> str:
    """Read files with syntax-aware truncation to keep output parsable.

    Args:
        path: File path to read.
        max_chars: Maximum characters to return.
        max_tokens: Optional token limit (converted to chars at 4 chars/token).
        limit: Hard cap for safety (default: DEFAULT_MODEL_CONTEXT_LIMIT).
    """
    limits: list[int] = [max_chars, limit]
    if max_tokens is not None:
        limits.append(max(0, max_tokens * 4))
    effective_limit = max(0, min(limits))
    if effective_limit == 0:
        return ""

    text = _read_text_for_context(path)
    if text is None:
        return ""

    suffix = path.suffix.lower()
    if suffix == ".py":
        skeleton = get_file_skeleton(path, limit=effective_limit)
        return skeleton[:effective_limit]
    if len(text) <= effective_limit:
        return text
    if suffix == ".json":
        return _truncate_json(text, effective_limit)
    if suffix in {".yaml", ".yml"}:
        return _truncate_structured_text(text, effective_limit, yaml.safe_load)
    return text[:effective_limit]


def get_file_skeleton(path: Path, *, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT) -> str:
    """Return a high-level AST skeleton of a Python file.

    The skeleton preserves imports, module-level assignments, class definitions,
    function signatures, and docstrings. Function and method bodies are replaced
    with ``pass`` (or ellipsis) when they span 5 or more lines; shorter bodies are
    preserved. Falls back to a line-based truncation on syntax errors.

    Args:
        path: File path to read.
        limit: Hard cap for safety (default: DEFAULT_MODEL_CONTEXT_LIMIT).
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

    def _skeletonize_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if _node_length(node) < 5:
            return node
        doc_expr = _doc_expr(ast.get_docstring(node, clean=False))
        body: list[ast.stmt] = []
        if doc_expr:
            body.append(doc_expr)
        body.append(ast.Pass())
        # Python 3.12+ requires type_params for FunctionDef/AsyncFunctionDef
        new_node = type(node)(  # type: ignore[call-arg]  # typeshed missing type_params
            name=node.name,
            args=node.args,
            body=body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=getattr(node, "type_comment", None),
            type_params=getattr(node, "type_params", []),
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
        # Python 3.12+ requires type_params for ClassDef
        new_node = ast.ClassDef(  # type: ignore[call-arg]  # typeshed missing type_params
            name=node.name,
            bases=node.bases,
            keywords=node.keywords,
            body=new_body,
            decorator_list=node.decorator_list,
            type_params=getattr(node, "type_params", []),
        )
        return ast.copy_location(new_node, node)

    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        offsets = _line_offsets(source)
        limit_offset = min(limit, offsets[-1] if offsets else limit)
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
        return source[:limit]

    lines: list[str] = []
    for stmt in new_body:
        try:
            lines.append(ast.unparse(ast.fix_missing_locations(stmt)))
        except Exception:
            continue

    if not lines:
        return source[:limit]

    return "\n\n".join(lines)[:limit]


def _read_text_for_context(path: Path, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return fh.read(limit)
    except (OSError, UnicodeDecodeError):
        return None


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _is_parseable(snippet: str) -> bool:
    try:
        ast.parse(snippet)
    except SyntaxError:
        return False
    return True


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


__all__ = [
    "DEFAULT_MODEL_CONTEXT_LIMIT",
    "FILE_PATTERN",
    "SENSITIVE_PATTERNS",
    "STRUCTURED_EXTENSIONS",
    "SYNTAX_WARNING",
    "GatherContextResult",
    "gather_context",
    "get_file_skeleton",
    "read_file_context",
    "resolve_files_from_output",
    "run_and_capture",
    "smart_read_context",
]
