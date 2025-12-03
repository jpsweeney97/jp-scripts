"""AST-based constitutional compliance checker.

Rules are loaded from safety_rules.yaml via the config module.
Falls back to embedded defaults if the YAML is missing.
"""

from __future__ import annotations

import ast
from pathlib import Path

from jpscripts.governance.config import SafetyConfig, load_safety_config
from jpscripts.governance.secret_scanner import check_for_secrets
from jpscripts.governance.types import Violation, ViolationType


class ConstitutionChecker(ast.NodeVisitor):
    """AST visitor that detects constitutional violations.

    Tracks async context to properly identify blocking I/O calls.
    Rules are loaded from safety_rules.yaml configuration.
    """

    def __init__(self, file_path: Path, source: str, config: SafetyConfig | None = None) -> None:
        self.file_path = file_path
        self.source = source
        self.lines = source.splitlines()
        self.violations: list[Violation] = []
        self._async_depth: int = 0
        self._imports: dict[str, str] = {}  # Maps alias -> "module" or "module.function"
        self._config = config or load_safety_config()

    @property
    def _in_async_context(self) -> bool:
        return self._async_depth > 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track entry into async context."""
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        """Track module imports and aliases."""
        for alias in node.names:
            # alias.name is "subprocess", alias.asname is "sp" (or None)
            name = alias.asname or alias.name
            self._imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-imports and aliases."""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            if alias.name == "*":
                continue  # Can't track wildcard imports
            # Store as "module.function" for function imports
            self._imports[name] = f"{module}.{alias.name}" if module else alias.name
        self.generic_visit(node)

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
        if self._config.safety_override_pattern in line_content:
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
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
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
        """Detect os.system() usage (always forbidden).

        Handles import aliasing (e.g., import os as o; o.system()).
        """
        module, func = self._resolve_call_target(node)
        if module == "os" and func == "system":
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
        if self._config.safety_override_pattern in line_content:
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
            return self._config.safety_override_pattern in line_content

        if isinstance(func, ast.Name) and func.id in self._config.forbidden_dynamic_builtins:
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
            if (isinstance(func.value, ast.Name) and func.value.id == "importlib") or (
                isinstance(func.value, ast.Attribute) and func.value.attr == "importlib"
            ):
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
        """Detect process exit calls (sys.exit, quit, exit).

        Handles import aliasing (e.g., import sys as s; s.exit()).
        """
        module, func = self._resolve_call_target(node)

        # Check sys.exit() or alias
        if module == "sys" and func == "exit":
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

        # Check quit() and exit() (direct name calls, not aliased)
        if isinstance(node.func, ast.Name) and node.func.id in self._config.exit_builtins:
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
        """Detect debug breakpoints (breakpoint, pdb.set_trace, ipdb.set_trace).

        Handles import aliasing (e.g., import pdb as p; p.set_trace()).
        """
        # Check breakpoint() - direct name, no alias possible
        if isinstance(node.func, ast.Name) and node.func.id in self._config.debug_builtins:
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

        # Check pdb.set_trace() and ipdb.set_trace() with alias support
        module, func = self._resolve_call_target(node)
        if module in self._config.debug_modules and func == "set_trace":
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

        is_open_call = (isinstance(node.func, ast.Name) and node.func.id == "open") or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("builtins", "io")
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
            if (
                "type: ignore" not in line_content
                and "# type:" not in line_content
                and self._appears_to_be_type_annotation(node)
            ):
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
            if ("from typing import" in line or "import typing" in line) and "Any" in line:
                return True
        return False

    def _resolve_call_target(self, node: ast.Call) -> tuple[str | None, str | None]:
        """Resolve a call to (module, function) tuple, handling import aliases.

        Returns (module, function) where:
        - sp.run() → ('subprocess', 'run') if sp is aliased to subprocess
        - r() → ('subprocess', 'run') if r is aliased to subprocess.run
        - subprocess.run() → ('subprocess', 'run')
        """
        func = node.func

        # Case 1: name.attr (e.g., sp.run, subprocess.run)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            alias = func.value.id
            resolved = self._imports.get(alias)
            if resolved:
                return (resolved, func.attr)
            return (alias, func.attr)

        # Case 2: direct name (e.g., run() after from subprocess import run)
        if isinstance(func, ast.Name):
            resolved = self._imports.get(func.id)
            if resolved and "." in resolved:
                parts = resolved.rsplit(".", 1)
                return (parts[0], parts[1])

        return (None, None)

    def _get_blocking_subprocess_func(self, node: ast.Call) -> str | None:
        """Check if call is a blocking subprocess function.

        Returns the function name if it's a blocking subprocess call, None otherwise.
        Handles import aliasing (e.g., import subprocess as sp).
        Uses blocking functions list from safety_rules.yaml config.
        """
        module, func = self._resolve_call_target(node)
        if module == "subprocess" and func in self._config.blocking_subprocess_funcs:
            return func
        return None

    def _is_subprocess_run(self, node: ast.Call) -> bool:
        """Check if call is subprocess.run (legacy compatibility)."""
        return self._get_blocking_subprocess_func(node) == "run"

    def _is_subprocess_call(self, node: ast.Call) -> bool:
        """Check if call is any subprocess module function.

        Handles import aliasing (e.g., import subprocess as sp).
        """
        module, _ = self._resolve_call_target(node)
        return module == "subprocess"

    def _is_destructive_fs_call(self, node: ast.Call) -> bool:
        """Check if call targets destructive filesystem operations.

        Handles import aliasing (e.g., import shutil as sh; sh.rmtree()).
        Uses destructive functions list from safety_rules.yaml config.
        """
        # Use resolution helper for shutil.rmtree and os.remove/unlink
        module, func_name = self._resolve_call_target(node)

        if module == "shutil" and func_name in self._config.destructive_shutil_funcs:
            return True

        if module == "os" and func_name in self._config.destructive_os_funcs:
            return True

        # Path.unlink() is a method on Path instances, needs special handling
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "unlink":
            return False

        target = func.value
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

    def check_and_collect_secrets(self) -> None:
        """Run secret detection and add violations to the collection."""
        self.violations.extend(check_for_secrets(self.source, self.file_path))


__all__ = [
    "ConstitutionChecker",
]
