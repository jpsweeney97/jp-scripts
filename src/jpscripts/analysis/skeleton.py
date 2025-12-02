"""AST skeleton extraction for Python source code.

Provides high-level code structure extraction that preserves:
- Imports and module-level assignments
- Class definitions with method signatures
- Function signatures with docstrings
- Bodies replaced with 'pass' for functions spanning 5+ lines

This module works on source strings, not file paths, to keep I/O
concerns separate from AST processing.
"""

from __future__ import annotations

import ast

from jpscripts.core.console import get_logger

logger = get_logger(__name__)

# Warning message for files with syntax errors
SYNTAX_WARNING = "# [WARN] Syntax error detected. AST features disabled.\n"


def _node_length(node: ast.AST) -> int:
    """Calculate line span of an AST node."""
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    return max(end - start + 1, 0)


def _doc_expr(raw: str | None) -> ast.Expr | None:
    """Convert docstring to AST expression node."""
    if raw is None:
        return None
    return ast.Expr(value=ast.Constant(value=raw))


def _skeletonize_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Reduce function body to signature + docstring + pass for functions >= 5 lines."""
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
    """Reduce class body to method signatures and class-level definitions."""
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


def _line_offsets(text: str) -> list[int]:
    """Build list of character offsets for each line in text."""
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _is_parseable(snippet: str) -> bool:
    """Check if a Python snippet is syntactically valid."""
    try:
        ast.parse(snippet)
    except SyntaxError:
        return False
    return True


def _fallback_read(text: str, limit: int, error: SyntaxError | None) -> str:
    """Return truncated source with head/tail context when AST parsing fails."""
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


def get_file_skeleton(source: str, *, limit: int = 1_000_000) -> str:
    """Return a high-level AST skeleton of Python source code.

    The skeleton preserves imports, module-level assignments, class definitions,
    function signatures, and docstrings. Function and method bodies are replaced
    with ``pass`` (or ellipsis) when they span 5 or more lines; shorter bodies are
    preserved. Falls back to a line-based truncation on syntax errors.

    Args:
        source: Python source code as a string.
        limit: Maximum output length in characters (default 1M).

    Returns:
        Skeleton representation of the source code.
    """
    if not source:
        return ""

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


__all__ = [
    "SYNTAX_WARNING",
    "_doc_expr",
    # Internal helpers (exported for testing)
    "_fallback_read",
    "_is_parseable",
    "_line_offsets",
    "_node_length",
    "_skeletonize_class",
    "_skeletonize_function",
    "get_file_skeleton",
]
