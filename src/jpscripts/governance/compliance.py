"""High-level compliance checking functions.

Provides both sync and async versions of compliance checking:
- check_compliance / check_compliance_async
- check_source_compliance / check_source_compliance_async
- scan_codebase_compliance / scan_codebase_compliance_async
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from pathlib import Path

from jpscripts.core.console import get_logger
from jpscripts.core.sys.execution import run_cpu_bound
from jpscripts.governance.ast_checker import ConstitutionChecker
from jpscripts.governance.diff_parser import apply_patch_in_memory, parse_diff_files
from jpscripts.governance.secret_scanner import check_for_secrets
from jpscripts.governance.types import Violation, ViolationType

logger = get_logger(__name__)


def _detect_safety_bypass_additions(diff: str, root: Path) -> list[Violation]:
    """Detect attempts to add '# safety: checked' overrides in a patch.

    Agents are not allowed to self-approve violations by adding safety
    override comments. This function scans added lines (those starting with '+')
    for the safety override pattern and flags them as SECURITY_BYPASS violations.

    Args:
        diff: Unified diff text
        root: Workspace root for resolving paths

    Returns:
        List of SECURITY_BYPASS violations for any added safety overrides
    """
    violations: list[Violation] = []
    current_file: Path | None = None
    current_line = 0

    for line in diff.splitlines():
        # Track current file
        if line.startswith("+++ b/"):
            path_str = line[6:].strip()
            current_file = root / path_str
        elif line.startswith("+++ "):
            path_str = line[4:].strip()
            if path_str.startswith("b/"):
                path_str = path_str[2:]
            current_file = root / path_str

        # Track line numbers from hunk headers
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_line = int(match.group(1))

        # Check added lines for safety override
        elif line.startswith("+") and not line.startswith("+++"):
            if current_file is not None and "# safety: checked" in line:
                violations.append(
                    Violation(
                        type=ViolationType.SECURITY_BYPASS,
                        file=current_file,
                        line=current_line,
                        column=line.find("# safety: checked"),
                        message="Agent attempted to add '# safety: checked' override",
                        suggestion=(
                            "Safety overrides must be pre-existing in the codebase. "
                            "Agents cannot self-approve violations by adding this comment."
                        ),
                        severity="error",
                        fatal=True,
                    )
                )
            current_line += 1

        # Context and deleted lines
        elif line.startswith("-") and not line.startswith("---"):
            pass  # Deleted lines don't increment
        else:
            current_line += 1

    return violations


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

    # SECURITY: Check for attempts to add safety overrides in the patch
    # This must happen BEFORE AST checking, as it catches bypass attempts
    violations.extend(_detect_safety_bypass_additions(diff, root))

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


async def check_compliance_async(diff: str, root: Path) -> list[Violation]:
    """Async version of check_compliance.

    Parse a diff and check PATCHED code for constitutional violations,
    without blocking the asyncio event loop.

    Args:
        diff: Unified diff text
        root: Workspace root for resolving paths

    Returns:
        List of violations found in the patched code
    """
    return await run_cpu_bound(check_compliance, diff, root)


async def check_source_compliance_async(source: str, file_path: Path) -> list[Violation]:
    """Async version of check_source_compliance.

    Check a source string directly for constitutional violations,
    without blocking the asyncio event loop.

    Args:
        source: Python source code
        file_path: Path for error reporting

    Returns:
        List of violations found
    """
    return await run_cpu_bound(check_source_compliance, source, file_path)


async def scan_codebase_compliance_async(root: Path) -> tuple[list[Violation], int]:
    """Async version of scan_codebase_compliance.

    Scan all Python files in a directory tree for constitutional violations,
    without blocking the asyncio event loop.

    Args:
        root: Root directory to scan (e.g., src/)

    Returns:
        Tuple of (violations_list, files_scanned_count)
    """
    return await run_cpu_bound(scan_codebase_compliance, root)


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
    "check_compliance_async",
    "check_source_compliance",
    "check_source_compliance_async",
    "count_violations_by_severity",
    "format_violations_for_agent",
    "has_fatal_violations",
    "scan_codebase_compliance",
    "scan_codebase_compliance_async",
]
