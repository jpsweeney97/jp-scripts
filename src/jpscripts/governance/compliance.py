"""High-level compliance checking functions."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from jpscripts.core.console import get_logger
from jpscripts.governance.ast_checker import ConstitutionChecker
from jpscripts.governance.diff_parser import apply_patch_in_memory, parse_diff_files
from jpscripts.governance.secret_scanner import check_for_secrets
from jpscripts.governance.types import Violation, ViolationType

logger = get_logger(__name__)


def check_compliance(diff: str, root: Path) -> list[Violation]:
    """
    Parse a diff and check PATCHED code for constitutional violations.

    CRITICAL: This function checks the code that WILL exist after the patch
    is applied, NOT the current code on disk. This prevents bypassing
    governance by introducing violations in new files or modifications.

    Args:
        diff: Unified diff text
        root: Workspace root for resolving paths

    Returns:
        List of violations found in the patched code
    """
    violations: list[Violation] = []

    # Apply patch in memory to get post-patch content
    patched_files = apply_patch_in_memory(diff, root)

    # Also get changed line numbers for filtering
    changed_lines_map = parse_diff_files(diff, root)

    for file_path, source in patched_files.items():
        if file_path.suffix != ".py":
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            # Flag syntax errors as violations - could be hiding other issues
            violations.append(
                Violation(
                    type=ViolationType.SYNTAX_ERROR,
                    file=file_path,
                    line=exc.lineno or 1,
                    column=exc.offset or 0,
                    message=f"Python syntax error prevents AST analysis: {exc.msg}",
                    suggestion="Fix the syntax error to enable full constitutional checking.",
                    severity="warning",
                    fatal=False,  # Warning, not fatal - allow agent to fix
                )
            )
            continue

        checker = ConstitutionChecker(file_path, source)
        checker.visit(tree)
        checker.violations.extend(check_for_secrets(source, file_path))

        # Get changed lines for this file (if any)
        changed_lines = changed_lines_map.get(file_path, set())

        # Filter to only violations on changed lines (or include all if no line info)
        for v in checker.violations:
            if not changed_lines or v.line in changed_lines:
                violations.append(v)

    return violations


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
    except SyntaxError as exc:
        # Return syntax error as a warning violation rather than silently skipping
        return [
            Violation(
                type=ViolationType.SYNTAX_ERROR,
                file=file_path,
                line=exc.lineno or 1,
                column=exc.offset or 0,
                message=f"Python syntax error prevents AST analysis: {exc.msg}",
                suggestion="Fix the syntax error to enable full constitutional checking.",
                severity="warning",
                fatal=False,
            )
        ]

    checker = ConstitutionChecker(file_path, source)
    checker.visit(tree)
    secret_violations = check_for_secrets(source, file_path)
    return checker.violations + secret_violations


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


def scan_codebase_compliance(root: Path) -> tuple[list[Violation], int]:
    """Scan all Python files in a directory tree for constitutional violations.

    Args:
        root: Root directory to scan (e.g., src/)

    Returns:
        Tuple of (violations_list, files_scanned_count)
    """
    violations: list[Violation] = []
    file_count = 0
    for py_file in root.rglob("*.py"):
        file_count += 1
        try:
            source = py_file.read_text(encoding="utf-8")
            violations.extend(check_source_compliance(source, py_file))
        except Exception as exc:
            # Skip files that can't be read (permissions, encoding issues)
            logger.debug("Skipping unreadable file %s: %s", py_file, exc)
    return violations, file_count


__all__ = [
    "check_compliance",
    "check_source_compliance",
    "count_violations_by_severity",
    "format_violations_for_agent",
    "has_fatal_violations",
    "scan_codebase_compliance",
]
