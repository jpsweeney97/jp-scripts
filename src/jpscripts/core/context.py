from __future__ import annotations

import ast
import asyncio
import importlib
import fnmatch
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence, cast

import yaml
from rich.console import Console

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.console import get_logger

# Regex to catch file paths, often with line numbers (e.g., "src/main.py:42")
# Matches: (start of line or space) (relative path) (:line_number optional)
FILE_PATTERN = re.compile(r"(?:^|\s)(?P<path>[\w./-]+)(?::\d+)?", re.MULTILINE | re.IGNORECASE)

# Default model context limit (can be overridden per-model at runtime)
DEFAULT_MODEL_CONTEXT_LIMIT = 200_000
STRUCTURED_EXTENSIONS = {".json", ".yml", ".yaml"}
SYNTAX_WARNING = "# [WARN] Syntax error detected. AST features disabled.\n"
SENSITIVE_PATTERNS = [".env", ".env.*", "*.pem", "*.key", "id_rsa"]

logger = get_logger(__name__)

class _EncoderProtocol(Protocol):
    def encode(self, text: str, *, disallowed_special: Sequence[str] | set[str] | tuple[str, ...] = ()) -> list[int]:
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...


class TokenCounter:
    """Token counter backed by tiktoken with heuristic fallback."""

    def __init__(self, default_model: str = "gpt-4o") -> None:
        self.default_model = default_model
        self._encoders: dict[str, _EncoderProtocol | None] = {}
        self._warned_missing = False

    def count_tokens(self, text: str, model: str | None = None) -> int:
        target_model = model or self.default_model
        encoder = self._get_encoder(target_model)
        if encoder is None:
            return self._heuristic_tokens(text)
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception as exc:
            logger.warning("Token counting failed for model %s: %s", target_model, exc)
            return self._heuristic_tokens(text)

    def trim_to_fit(self, text: str, max_tokens: int, model: str | None = None) -> str:
        """Trim text to fit within max_tokens."""
        if max_tokens <= 0:
            return ""

        target_model = model or self.default_model
        encoder = self._get_encoder(target_model)
        if encoder is None:
            return text[: self.tokens_to_characters(max_tokens)]

        try:
            encoded = encoder.encode(text, disallowed_special=())
            if len(encoded) <= max_tokens:
                return text
            return encoder.decode(encoded[:max_tokens])
        except Exception as exc:
            logger.warning("Token trim failed for model %s: %s", target_model, exc)
            return text[: self.tokens_to_characters(max_tokens)]

    def tokens_to_characters(self, tokens: int) -> int:
        """Coarse conversion from tokens to characters (upper bound)."""
        if tokens <= 0:
            return 0
        return tokens * 4

    def _heuristic_tokens(self, text: str) -> int:
        return max(0, len(text) // 4)

    def _get_encoder(self, model: str) -> _EncoderProtocol | None:
        if model in self._encoders:
            return self._encoders[model]

        try:
            tiktoken_module = importlib.import_module("tiktoken")
        except ImportError:
            if not self._warned_missing:
                logger.warning("tiktoken is not installed; falling back to heuristic token estimates.")
                self._warned_missing = True
            self._encoders[model] = None
            return None
        except Exception as exc:  # pragma: no cover - defensive import guard
            logger.warning("Failed to import tiktoken: %s", exc)
            self._encoders[model] = None
            return None

        try:
            encoding_for_model = getattr(tiktoken_module, "encoding_for_model", None)
            if not callable(encoding_for_model):
                raise AttributeError("encoding_for_model is unavailable on tiktoken module")
            encoder = encoding_for_model(model)
        except Exception as exc:
            logger.warning("Failed to load encoding for model %s: %s", model, exc)
            self._encoders[model] = None
            return None

        cached = cast(_EncoderProtocol, encoder)
        self._encoders[model] = cached
        return cached


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


def get_file_skeleton(
    path: Path, *, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
) -> str:
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
        new_node = type(node)(
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
        new_node = ast.ClassDef(
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


def _read_text_for_context(
    path: Path, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
) -> str | None:
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


# ---------------------------------------------------------------------------
# Token Budget Manager
# ---------------------------------------------------------------------------

Priority = Literal[1, 2, 3]
TRUNCATION_MARKER = "[...truncated]"


@dataclass
class TokenBudgetManager:
    """Priority-based token budget allocation using precise token counts."""

    total_budget: int
    reserved_budget: int = 0
    model_context_limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
    model: str = "gpt-4o"
    token_counter: TokenCounter = field(default_factory=TokenCounter)
    _used_tokens: int = field(default=0, repr=False)
    _allocations: dict[Priority, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.total_budget < 0:
            raise ValueError("total_budget must be non-negative")
        if self.reserved_budget < 0:
            raise ValueError("reserved_budget must be non-negative")
        if self.reserved_budget > self.total_budget:
            raise ValueError("reserved_budget cannot exceed total_budget")
        if self.model_context_limit <= 0:
            raise ValueError("model_context_limit must be positive")
        self._allocations = {1: 0, 2: 0, 3: 0}

    def remaining(self) -> int:
        """Return remaining token budget available for allocation."""
        return max(0, self.total_budget - self.reserved_budget - self._used_tokens)

    def tokens_to_characters(self, tokens: int) -> int:
        """Convert token budget to a conservative character budget."""
        char_budget = self.token_counter.tokens_to_characters(tokens)
        return min(char_budget, self.token_counter.tokens_to_characters(self.model_context_limit))

    def allocate(
        self,
        priority: Priority,
        content: str,
        source_path: Path | None = None,
    ) -> str:
        """Allocate content within token budget, with optional syntax-aware truncation."""
        if not content:
            return ""

        token_budget = self.remaining()
        if token_budget <= 0:
            return ""

        token_count = self.token_counter.count_tokens(content, model=self.model)
        if token_count <= token_budget:
            self._track_allocation(priority, token_count)
            return content

        truncated = self._truncate_content(content, token_budget, source_path)
        if not truncated:
            return ""

        final_tokens = self.token_counter.count_tokens(truncated, model=self.model)
        if final_tokens > token_budget:
            truncated = self.token_counter.trim_to_fit(truncated, token_budget, model=self.model)
            final_tokens = self.token_counter.count_tokens(truncated, model=self.model)

        self._track_allocation(priority, final_tokens)
        return truncated

    def _track_allocation(self, priority: Priority, tokens: int) -> None:
        self._used_tokens += tokens
        self._allocations[priority] += tokens

    def _truncate_content(self, content: str, token_budget: int, source_path: Path | None) -> str:
        """Truncate content using syntax-aware readers when paths are provided."""
        char_budget = self.tokens_to_characters(token_budget)
        if char_budget <= 0:
            return ""

        truncated = (
            smart_read_context(
                source_path,
                char_budget,
                max_tokens=token_budget,
                limit=self.tokens_to_characters(self.model_context_limit),
            )
            if source_path
            else self._truncate_plain(content, char_budget)
        )
        if not truncated:
            return ""
        return truncated

    def _truncate_plain(self, content: str, limit: int) -> str:
        """Truncate plain content with marker, preferring line boundaries."""
        marker_len = len(TRUNCATION_MARKER) + 1  # +1 for newline
        if limit <= marker_len:
            return ""

        available = limit - marker_len
        truncated = content[:available]

        last_newline = truncated.rfind("\n")
        if last_newline > available // 2:
            truncated = truncated[:last_newline]

        return f"{truncated}\n{TRUNCATION_MARKER}"

    def summary(self) -> dict[str, int]:
        """Return allocation summary by priority (tokens)."""
        return {f"priority_{p}": tokens for p, tokens in self._allocations.items()}
