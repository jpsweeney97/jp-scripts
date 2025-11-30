"""
Constitutional compliance checker for jp-scripts.

This module enforces AGENTS.md rules programmatically by parsing diffs
and detecting violations via AST analysis. It implements a "warn + prompt"
strategy where violations are fed back to the agent for correction.

Key invariants enforced:
- No blocking I/O in async context (subprocess.run without asyncio.to_thread)
- No bare except clauses
- No shell=True in subprocess calls
- No untyped Any without type: ignore comment
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (?P<name>[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)
    \s*=\s*
    (?P<quote>['"]?)(?P<value>[A-Za-z0-9+/=_\-]{16,})(?P=quote)
    """
)


class ViolationType(Enum):
    """Types of constitutional violations."""

    SYNC_SUBPROCESS = auto()  # subprocess.run without async wrapping
    BARE_EXCEPT = auto()  # except: without specific exception
    SHELL_TRUE = auto()  # shell=True in subprocess calls
    UNTYPED_ANY = auto()  # Any type without type: ignore comment
    SYNC_OPEN = auto()  # open() in async context without to_thread
    OS_SYSTEM = auto()  # os.system() usage (always forbidden)
    DESTRUCTIVE_FS = auto()  # Destructive filesystem call without safety override
    DYNAMIC_EXECUTION = auto()  # eval/exec/dynamic imports without safety override
    SECRET_LEAK = auto()  # Secret or token detected in diff
    PROCESS_EXIT = auto()  # sys.exit(), quit(), exit()
    DEBUG_LEFTOVER = auto()  # breakpoint(), pdb.set_trace(), ipdb.set_trace()


@dataclass(frozen=True)
class Violation:
    """A single constitutional violation."""

    type: ViolationType
    file: Path
    line: int
    column: int
    message: str
    suggestion: str
    severity: str  # "error" | "warning"
    fatal: bool = False  # Fatal violations block patch application


class ConstitutionChecker(ast.NodeVisitor):
    """AST visitor that detects constitutional violations.

    Tracks async context to properly identify blocking I/O calls.
    """

    def __init__(self, file_path: Path, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.lines = source.splitlines()
        self.violations: list[Violation] = []
        self._async_depth: int = 0

    @property
    def _in_async_context(self) -> bool:
        return self._async_depth > 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track entry into async context."""
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        """Check for prohibited function calls."""
        self._check_subprocess_run(node)
        self._check_shell_true(node)
        self._check_os_system(node)
        self._check_sync_open(node)
        self._check_destructive_fs(node)
        self._check_dynamic_execution(node)
        self._check_process_exit(node)
        self._check_debug_leftover(node)
        self.generic_visit(node)

    def _check_subprocess_run(self, node: ast.Call) -> None:
        """Detect blocking subprocess calls in async context without asyncio.to_thread."""
        blocking_func = self._get_blocking_subprocess_func(node)
        if blocking_func is None:
            return

        line_content = self._get_line(node.lineno)
        if "# safety: checked" in line_content:
            return

        if self._in_async_context:
            self.violations.append(
                Violation(
                    type=ViolationType.SYNC_SUBPROCESS,
                    file=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    message=f"subprocess.{blocking_func} called in async context without asyncio.to_thread",
                    suggestion=(
                        f"Wrap with asyncio.to_thread(subprocess.{blocking_func}, ...) or use "
                        "asyncio.create_subprocess_exec for true async"
                    ),
                    severity="error",
                    fatal=True,
                )
            )

    def _check_shell_true(self, node: ast.Call) -> None:
        """Detect shell=True in subprocess calls."""
        if not self._is_subprocess_call(node):
            return

        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant):
                if keyword.value.value is True:
                    self.violations.append(
                        Violation(
                            type=ViolationType.SHELL_TRUE,
                            file=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            message="shell=True is forbidden by AGENTS.md constitution",
                            suggestion=(
                                "Use shlex.split() to tokenize the command and pass "
                                "as a list to subprocess without shell=True"
                            ),
                            severity="error",
                            fatal=True,
                        )
                    )

    def _check_os_system(self, node: ast.Call) -> None:
        """Detect os.system() usage (always forbidden)."""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "system":
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    self.violations.append(
                        Violation(
                            type=ViolationType.OS_SYSTEM,
                            file=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            message="os.system() is forbidden by AGENTS.md constitution",
                            suggestion=(
                                "Use asyncio.create_subprocess_exec or core.system.run_safe_shell "
                                "with proper command validation"
                            ),
                            severity="error",
                            fatal=True,
                        )
                    )

    def _check_destructive_fs(self, node: ast.Call) -> None:
        """Detect destructive filesystem operations without explicit safety override."""
        if not self._is_destructive_fs_call(node):
            return

        line_content = self._get_line(node.lineno)
        if "# safety: checked" in line_content:
            return

        self.violations.append(
            Violation(
                type=ViolationType.DESTRUCTIVE_FS,
                file=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                message="Destructive filesystem call without '# safety: checked' override",
                suggestion="Avoid destructive filesystem calls or add '# safety: checked' when explicitly audited.",
                severity="error",
                fatal=True,
            )
        )

    def _check_dynamic_execution(self, node: ast.Call) -> None:
        """Detect dynamic execution patterns (eval/exec/compile/dynamic import)."""
        func = node.func
        line_content = self._get_line(node.lineno)

        def _has_safety_override() -> bool:
            return "# safety: checked" in line_content

        if isinstance(func, ast.Name) and func.id in {"eval", "exec", "compile", "__import__"}:
            self.violations.append(
                Violation(
                    type=ViolationType.DYNAMIC_EXECUTION,
                    file=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    message=f"Dynamic execution via {func.id}() is forbidden.",
                    suggestion="Remove dynamic execution or replace with explicit imports and functions.",
                    severity="error",
                    fatal=True,
                )
            )
            return

        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            # importlib.import_module or alias; allow safety override
            if _has_safety_override():
                return
            is_importlib = False
            if (isinstance(func.value, ast.Name) and func.value.id == "importlib") or (isinstance(func.value, ast.Attribute) and func.value.attr == "importlib"):
                is_importlib = True

            if is_importlib:
                self.violations.append(
                    Violation(
                        type=ViolationType.DYNAMIC_EXECUTION,
                        file=self.file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        message="Dynamic import via importlib.import_module is forbidden without explicit safety override.",
                        suggestion="Use static imports or add '# safety: checked' only after manual review.",
                        severity="error",
                        fatal=True,
                    )
                )

    def _check_process_exit(self, node: ast.Call) -> None:
        """Detect process exit calls (sys.exit, quit, exit)."""
        # Check sys.exit()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "exit" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "sys":
                    self.violations.append(
                        Violation(
                            type=ViolationType.PROCESS_EXIT,
                            file=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            message="Direct process exit forbidden. Let the function return normally or raise an exception.",
                            suggestion="Remove the exit call and use return or raise an appropriate exception.",
                            severity="error",
                            fatal=True,
                        )
                    )
                    return

        # Check quit() and exit()
        if isinstance(node.func, ast.Name):
            if node.func.id in ("quit", "exit"):
                self.violations.append(
                    Violation(
                        type=ViolationType.PROCESS_EXIT,
                        file=self.file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        message="Direct process exit forbidden. Let the function return normally or raise an exception.",
                        suggestion="Remove the exit call and use return or raise an appropriate exception.",
                        severity="error",
                        fatal=True,
                    )
                )

    def _check_debug_leftover(self, node: ast.Call) -> None:
        """Detect debug breakpoints (breakpoint, pdb.set_trace, ipdb.set_trace)."""
        # Check breakpoint()
        if isinstance(node.func, ast.Name):
            if node.func.id == "breakpoint":
                self.violations.append(
                    Violation(
                        type=ViolationType.DEBUG_LEFTOVER,
                        file=self.file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        message="Debug breakpoints are forbidden in production code.",
                        suggestion="Remove the debugging statement before committing.",
                        severity="error",
                        fatal=True,
                    )
                )
                return

        # Check pdb.set_trace() and ipdb.set_trace()
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "set_trace" and isinstance(node.func.value, ast.Name):
                if node.func.value.id in ("pdb", "ipdb"):
                    self.violations.append(
                        Violation(
                            type=ViolationType.DEBUG_LEFTOVER,
                            file=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            message="Debug breakpoints are forbidden in production code.",
                            suggestion="Remove the debugging statement before committing.",
                            severity="error",
                            fatal=True,
                        )
                    )

    def _check_sync_open(self, node: ast.Call) -> None:
        """Detect open() in async context without wrapping."""
        if not self._in_async_context:
            return

        is_open_call = (
            (isinstance(node.func, ast.Name)
            and node.func.id == "open")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "open"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("builtins", "io")
            )
        )

        if is_open_call:
            # Check if line has aiofiles or asyncio.to_thread context
            line_content = self._get_line(node.lineno)
            if "aiofiles" not in line_content and "to_thread" not in line_content:
                self.violations.append(
                    Violation(
                        type=ViolationType.SYNC_OPEN,
                        file=self.file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        message="Synchronous open() in async context",
                        suggestion=("Use aiofiles.open() or wrap with asyncio.to_thread()"),
                        severity="warning",
                    )
                )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Check for bare except clauses."""
        if node.type is None:
            self.violations.append(
                Violation(
                    type=ViolationType.BARE_EXCEPT,
                    file=self.file_path,
                    line=node.lineno,
                    column=node.col_offset,
                    message="Bare except clause is forbidden by AGENTS.md constitution",
                    suggestion=(
                        "Catch specific exceptions and wrap in Result[T, JPScriptsError] "
                        "or at minimum use 'except Exception:'"
                    ),
                    severity="error",
                    fatal=True,
                )
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check for Any type usage in type annotations."""
        # Handle typing.Any subscript case
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for Any type usage."""
        if node.id == "Any":
            line_content = self._get_line(node.lineno)
            # Check for type: ignore comment on this line
            if "type: ignore" not in line_content and "# type:" not in line_content:
                # Only flag if it appears to be a type annotation context
                # (heuristic: check if 'from typing' or similar is in file)
                if self._appears_to_be_type_annotation(node):
                    self.violations.append(
                        Violation(
                            type=ViolationType.UNTYPED_ANY,
                            file=self.file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            message="Any type used without type: ignore comment",
                            suggestion=(
                                "Add '# type: ignore[<code>]' with justification or "
                                "use a more specific type"
                            ),
                            severity="warning",
                        )
                    )
        self.generic_visit(node)

    def _appears_to_be_type_annotation(self, node: ast.Name) -> bool:
        """Heuristic to check if Any is used as a type annotation."""
        # Check if typing module is imported
        for line in self.lines[:50]:  # Check first 50 lines for imports
            if "from typing import" in line or "import typing" in line:
                if "Any" in line:
                    return True
        return False

    # Blocking subprocess functions that should be wrapped with asyncio.to_thread
    _BLOCKING_SUBPROCESS_FUNCS: frozenset[str] = frozenset(
        {
            "run",
            "call",
            "check_call",
            "check_output",
            "Popen",
            "getoutput",
            "getstatusoutput",
        }
    )

    def _get_blocking_subprocess_func(self, node: ast.Call) -> str | None:
        """Check if call is a blocking subprocess function.

        Returns the function name if it's a blocking subprocess call, None otherwise.
        """
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in self._BLOCKING_SUBPROCESS_FUNCS:
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "subprocess":
                        return node.func.attr
        return None

    def _is_subprocess_run(self, node: ast.Call) -> bool:
        """Check if call is subprocess.run (legacy compatibility)."""
        return self._get_blocking_subprocess_func(node) == "run"

    def _is_subprocess_call(self, node: ast.Call) -> bool:
        """Check if call is any subprocess module function."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id == "subprocess"
        return False

    def _is_destructive_fs_call(self, node: ast.Call) -> bool:
        """Check if call targets destructive filesystem operations."""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False

        target = func.value
        if func.attr == "rmtree" and isinstance(target, ast.Name):
            return target.id == "shutil"

        if func.attr in {"remove", "unlink"} and isinstance(target, ast.Name):
            return target.id == "os"

        if func.attr != "unlink":
            return False

        if isinstance(target, ast.Name):
            return target.id == "Path"

        if isinstance(target, ast.Attribute):
            return target.attr == "Path"

        if isinstance(target, ast.Call):
            call_func = target.func
            if isinstance(call_func, ast.Name):
                return call_func.id == "Path"
            if isinstance(call_func, ast.Attribute):
                return call_func.attr == "Path"

        return False

    def _get_line(self, lineno: int) -> str:
        """Get source line content (1-indexed)."""
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1]
        return ""


def check_compliance(diff: str, root: Path) -> list[Violation]:
    """
    Parse a diff and check changed code for constitutional violations.

    Args:
        diff: Unified diff text
        root: Workspace root for resolving paths

    Returns:
        List of violations found in the changed code
    """
    violations: list[Violation] = []

    # Parse diff to extract changed files and new/modified lines
    changed_files = _parse_diff_files(diff, root)

    for file_path, changed_lines in changed_files.items():
        if not file_path.suffix == ".py":
            continue

        if not file_path.exists():
            continue

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue

        checker = ConstitutionChecker(file_path, source)
        checker.visit(tree)
        checker.violations.extend(check_for_secrets(source, file_path))

        # Filter to only violations on changed lines (or include all if no line info)
        for v in checker.violations:
            if not changed_lines or v.line in changed_lines:
                violations.append(v)

    return violations


def check_for_secrets(content: str, file_path: Path) -> list[Violation]:
    """Detect obvious secret-like assignments in content."""
    matches = list(_SECRET_PATTERN.finditer(content))
    lines = content.splitlines()
    violations: list[Violation] = []
    for match in matches:
        name = match.group("name")
        value = match.group("value")
        line = content.count("\n", 0, match.start()) + 1
        column = match.start() - content.rfind("\n", 0, match.start()) - 1
        # Check for safety override
        if 0 < line <= len(lines) and "# safety: checked" in lines[line - 1]:
            continue
        entropy = _estimate_entropy(value)
        if entropy < 3.5:
            continue
        violations.append(
            Violation(
                type=ViolationType.SECRET_LEAK,
                file=file_path,
                line=line,
                column=column,
                message=f"Potential secret detected in {name} with high-entropy value.",
                suggestion="Remove the secret, rotate credentials, and load from environment or secret manager.",
                severity="error",
                fatal=True,
            )
        )
    return violations


def _estimate_entropy(value: str) -> float:
    """Rough entropy estimator for secret-like strings."""
    if not value:
        return 0.0
    freq = {ch: value.count(ch) for ch in set(value)}
    length = len(value)
    import math

    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log(p, 2)
    return entropy


def check_source_compliance(source: str, file_path: Path) -> list[Violation]:
    """
    Check a source string directly for constitutional violations.

    Useful for checking patches before applying them.

    Args:
        source: Python source code
        file_path: Path for error reporting

    Returns:
        List of violations found
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    checker = ConstitutionChecker(file_path, source)
    checker.visit(tree)
    secret_violations = check_for_secrets(source, file_path)
    return checker.violations + secret_violations


def _parse_diff_files(diff: str, root: Path) -> dict[Path, set[int]]:
    """Parse diff to extract file paths and changed line numbers.

    Returns a mapping from file path to set of modified line numbers.
    An empty set means all lines should be checked.
    """
    files: dict[Path, set[int]] = {}
    current_file: Path | None = None
    current_line = 0

    for line in diff.splitlines():
        # Match new file header: +++ b/path/to/file.py
        if line.startswith("+++ b/"):
            path_str = line[6:]
            current_file = root / path_str
            files[current_file] = set()
        elif line.startswith("+++ "):
            # Handle other diff formats: +++ path/to/file.py
            path_str = line[4:].strip()
            if path_str.startswith("b/"):
                path_str = path_str[2:]
            current_file = root / path_str
            files[current_file] = set()
        elif line.startswith("@@ "):
            # Parse hunk header: @@ -start,count +start,count @@
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            # Added line
            if current_file is not None:
                files[current_file].add(current_line)
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Deleted line - don't increment current_line
            pass
        else:
            # Context line or other
            current_line += 1

    return files


def format_violations_for_agent(violations: Sequence[Violation]) -> str:
    """Format violations as structured feedback for the agent to fix.

    Returns markdown-formatted text suitable for injection into agent history.
    """
    if not violations:
        return ""

    lines = [
        "## Constitutional Violations Detected",
        "",
        "Your proposed changes violate the following AGENTS.md constitutional rules.",
        "Please revise your patch to address these issues:",
        "",
    ]

    # Group by severity
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    if errors:
        lines.append("### Errors (must fix)")
        lines.append("")
        for v in errors:
            lines.append(f"**{v.type.name}** at `{v.file.name}:{v.line}`")
            lines.append(f"- **Issue:** {v.message}")
            lines.append(f"- **Fix:** {v.suggestion}")
            lines.append("")

    if warnings:
        lines.append("### Warnings (should fix)")
        lines.append("")
        for v in warnings:
            lines.append(f"**{v.type.name}** at `{v.file.name}:{v.line}`")
            lines.append(f"- **Issue:** {v.message}")
            lines.append(f"- **Fix:** {v.suggestion}")
            lines.append("")

    lines.append("Please update your patch to address these violations before proceeding.")
    return "\n".join(lines)


def count_violations_by_severity(violations: Sequence[Violation]) -> tuple[int, int]:
    """Count violations by severity.

    Returns:
        Tuple of (error_count, warning_count)
    """
    errors = sum(1 for v in violations if v.severity == "error")
    warnings = sum(1 for v in violations if v.severity == "warning")
    return errors, warnings


def has_fatal_violations(violations: Sequence[Violation]) -> bool:
    """Check if any violation in the sequence is fatal.

    Fatal violations must block patch application entirely.
    """
    return any(v.fatal for v in violations)


__all__ = [
    "ConstitutionChecker",
    "Violation",
    "ViolationType",
    "check_compliance",
    "check_source_compliance",
    "count_violations_by_severity",
    "format_violations_for_agent",
    "has_fatal_violations",
]
