"""Low-level agent operations and utilities.

This module contains helper functions for command execution, syntax verification,
file operations, and output summarization used by the repair loop.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from jpscripts.core import security
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok
from jpscripts.core.sys import run_safe_shell

logger = get_logger(__name__)


def summarize_output(stdout: str, stderr: str, limit: int) -> str:
    """Combine stdout/stderr and truncate if needed.

    Args:
        stdout: Standard output from command.
        stderr: Standard error from command.
        limit: Maximum character limit for output.

    Returns:
        Combined and potentially truncated output string.
    """
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    if not combined:
        return "Command failed without output."
    if len(combined) <= limit:
        return combined
    return summarize_stack_trace(combined, limit)


def summarize_stack_trace(text: str, limit: int) -> str:
    """Truncate stack traces while preserving head and tail context.

    Uses smart truncation to keep the most useful parts of stack traces:
    - Beginning (often shows the entry point)
    - End (shows the actual error)
    - Sample of the middle (for context)

    Args:
        text: The full text to truncate.
        limit: Maximum character limit.

    Returns:
        Truncated text with context preserved.
    """
    if limit <= 0:
        return ""
    lines = text.splitlines()
    if len(text) <= limit:
        return text
    if len(lines) < 4:
        return text[:limit] + "... [truncated]"

    head_keep = max(3, min(12, len(lines) // 3))
    tail_keep = max(6, min(20, len(lines) // 2))
    head_lines = lines[:head_keep]
    tail_lines = lines[-tail_keep:]
    middle_lines = lines[head_keep:-tail_keep] if tail_keep < len(lines) - head_keep else []

    middle_summary = ""
    if middle_lines:
        mid_idx = len(middle_lines) // 2
        window = middle_lines[max(0, mid_idx - 3) : min(len(middle_lines), mid_idx + 4)]
        middle_summary = (
            "\n[... middle truncated ...]\n" + "\n".join(window) + "\n[... resumes ...]\n"
        )

    assembled = "\n".join(head_lines) + middle_summary + "\n".join(tail_lines)
    if len(assembled) > limit:
        head_budget = max(limit // 3, 1)
        tail_budget = max(limit - head_budget - 40, 1)
        trimmed_head = "\n".join(lines)[:head_budget]
        trimmed_tail = "\n".join(lines)[-tail_budget:]
        return f"{trimmed_head}\n[... truncated for length ...]\n{trimmed_tail}"

    return assembled


async def run_agent_command(command: str, root: Path) -> tuple[int, str, str]:
    """Execute a shell command via centralized security validation.

    This is a thin adapter around run_safe_shell that converts the Result
    type to the (exit_code, stdout, stderr) tuple expected by run_repair_loop.

    Args:
        command: The shell command to execute.
        root: The working directory.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    result = await run_safe_shell(command, root, "agent.repair_loop")
    if isinstance(result, Ok):
        return (result.value.returncode, result.value.stdout, result.value.stderr)
    # Synthetic failure for blocked/invalid commands
    return (1, "", str(result.error))


async def verify_syntax(files: list[Path]) -> str | None:
    """Verify Python syntax for changed files using py_compile.

    Args:
        files: List of file paths to verify.

    Returns:
        Error message if syntax check fails, None if all files pass.
    """
    py_files = [path for path in files if path.suffix == ".py"]
    if not py_files:
        return None

    for path in py_files:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "py_compile",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return "Python interpreter not found for syntax check."
        except Exception as exc:  # pragma: no cover - defensive
            return f"Syntax check failed for {path}: {exc}"

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = (
                stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            )
            return f"Syntax error in {path}: {message or 'py_compile failed'}"

    return None


async def revert_files(paths: Sequence[Path], root: Path) -> None:
    """Revert modified files to git HEAD state.

    Validates each path for security before reverting, and disables git hooks
    to prevent malicious hook execution during the checkout.

    Args:
        paths: Sequence of file paths to revert.
        root: The workspace root directory.
    """
    if not paths:
        return

    safe_paths: list[Path] = []
    for path in paths:
        result = await security.validate_path_safe_async(path, root)
        if isinstance(result, Err):
            logger.debug("Skipping revert for unsafe path %s: %s", path, result.error.message)
            continue
        safe_paths.append(result.value)

    if not safe_paths:
        return

    try:
        # Disable git hooks to prevent malicious hook execution during revert.
        # The -c flag must come before the subcommand.
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "checkout",
            "--",
            *[str(path) for path in safe_paths],
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return

    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug(
            "Failed to revert files after unsuccessful loop: %s", stderr.decode(errors="replace")
        )


__all__ = [
    "revert_files",
    "run_agent_command",
    "summarize_output",
    "summarize_stack_trace",
    "verify_syntax",
]
