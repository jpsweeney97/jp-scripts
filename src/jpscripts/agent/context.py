"""Context collection helpers for agent prompts.

This module provides functions for gathering git context, file context,
and dependency information to build rich agent prompts.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from jpscripts.ai.tokens import Priority, TokenBudgetManager
from jpscripts.analysis.structure import get_import_dependencies
from jpscripts.core import security
from jpscripts.core.console import get_logger
from jpscripts.core.context_gatherer import (
    get_file_skeleton,
    read_file_context,
    resolve_files_from_output,
    smart_read_context,
)
from jpscripts.core.result import Err, Ok
from jpscripts.features.navigation import scan_recent
from jpscripts.git import client as git_core
from jpscripts.git import ops as git_ops

logger = get_logger(__name__)


async def load_constitution(root: Path) -> dict[str, object]:
    """Load and validate the constitutional JSON from AGENTS.md."""
    try:
        candidate = security.validate_path(root / "AGENTS.md", root)
    except Exception as exc:
        logger.debug("Unable to resolve AGENTS.md under %s: %s", root, exc)
        return {"status": "unavailable", "message": "AGENTS.md not accessible", "error": str(exc)}

    exists = await asyncio.to_thread(candidate.exists)
    if not exists:
        return {"status": "missing", "message": "AGENTS.md not found"}

    try:
        content = await asyncio.to_thread(candidate.read_text, encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to read AGENTS.md: %s", exc)
        return {"status": "unreadable", "message": "AGENTS.md unreadable", "error": str(exc)}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse AGENTS.md as JSON: %s", exc)
        return {"status": "parse_error", "message": str(exc)}

    if not isinstance(parsed, dict):
        return {"status": "invalid", "message": "AGENTS.md root must be an object"}

    constitution = parsed.get("constitution")
    if not isinstance(constitution, dict):
        return {"status": "invalid", "message": "Missing or invalid 'constitution' object"}

    return cast(dict[str, object], constitution)


async def collect_git_context(root: Path) -> tuple[str, str, bool]:
    """Collect git branch, commit hash, and dirty state."""
    if not root.exists() or not (root / ".git").exists():
        return "(no repo)", "(no repo)", False

    match await git_core.AsyncRepo.open(root):
        case Err(err):
            logger.error("Failed to open git repo at %s: %s", root, err)
            return "(error)", "(error)", False
        case Ok(repo):
            pass

    branch = "(unknown)"
    commit = "(unknown)"
    is_dirty = False

    match await repo.status():
        case Err(err):
            logger.error("Failed to describe git status for %s: %s", root, err)
            return "(error)", "(error)", False
        case Ok(status):
            branch = status.branch
            is_dirty = status.dirty
            _ = git_ops.format_status(status)

    match await repo.head(short=True):
        case Err(err):
            logger.error("Failed to resolve git head for %s: %s", root, err)
            commit = "(error)"
        case Ok(head_ref):
            commit = head_ref

    return branch, commit, is_dirty


async def collect_git_diff(root: Path, max_chars: int) -> str | None:
    """Collect git diff output, truncated to max_chars."""
    if not root.exists() or not (root / ".git").exists():
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "HEAD",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return None

    if proc.returncode != 0:
        return None

    diff = stdout.decode(errors="replace")
    if not diff.strip():
        return None

    if len(diff) > max_chars:
        return f"{diff[:max_chars]}... [truncated]"

    return diff


async def build_file_context_section(
    paths: Sequence[Path],
    budget: TokenBudgetManager,
) -> tuple[str, list[Path]]:
    """Build file context section using sequential greedy allocation.

    Each file is read only if budget remains, and allocated individually
    to preserve syntax boundaries per file.
    """
    sections: list[str] = []
    attached: list[Path] = []

    def _count_lines(target: Path) -> int:
        try:
            with target.open("r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except (OSError, UnicodeDecodeError):
            return 0

    for idx, path in enumerate(paths):
        # Check budget before reading each file
        remaining_tokens = budget.remaining()
        if remaining_tokens <= 0:
            break
        char_budget = budget.tokens_to_characters(remaining_tokens)

        priority: Priority = 3
        label = ""
        snippet = ""

        if idx == 0:
            priority = 2
            primary_text = await asyncio.to_thread(
                read_file_context,
                path,
                char_budget,
                limit=budget.tokens_to_characters(budget.model_context_limit),
            )
            snippet = primary_text or ""
        else:
            line_count = await asyncio.to_thread(_count_lines, path)
            use_skeleton = path.suffix.lower() == ".py" and line_count > 200

            if use_skeleton:
                snippet = await asyncio.to_thread(get_file_skeleton, path)
            else:
                snippet = await asyncio.to_thread(
                    smart_read_context,
                    path,
                    char_budget,
                    remaining_tokens,
                    limit=budget.tokens_to_characters(budget.model_context_limit),
                )
            if use_skeleton:
                label = " (Skeleton - Request full content if needed)"

        if not snippet:
            snippet = "(content unavailable)"

        file_entry = f"Path: {path}{label}\n---\n{snippet}\n"

        # Allocate this file's content - may be truncated if over budget
        allocated = budget.allocate(priority, file_entry, source_path=path)
        if allocated:
            sections.append(allocated)
            attached.append(path)

    if not sections:
        return "", attached
    return "\n".join(sections), attached


async def build_dependency_section(
    paths: Sequence[Path],
    root: Path,
    budget: TokenBudgetManager,
) -> str:
    """Build dependency section using sequential greedy allocation.

    Dependencies are read only if budget remains after file context.
    """
    dependencies: set[Path] = set()
    for path in paths:
        deps = await asyncio.to_thread(get_import_dependencies, path, root)
        dependencies.update(deps)

    if not dependencies:
        return ""

    sections: list[str] = []
    for dep in sorted(dependencies):
        # Check budget before reading each dependency
        remaining_tokens = budget.remaining()
        if remaining_tokens <= 0:
            break
        char_budget = budget.tokens_to_characters(remaining_tokens)

        snippet = await asyncio.to_thread(
            get_file_skeleton,
            dep,
            limit=char_budget,
        )
        if snippet and len(snippet) > char_budget:
            snippet = snippet[:char_budget]
        if not snippet:
            continue

        dep_entry = f"Dependency: {dep}\n---\n{snippet}\n"

        # Allocate this dependency's content
        allocated = budget.allocate(3, dep_entry, source_path=dep)
        if allocated:
            sections.append(allocated)

    return "\n".join(sections)


async def expand_context_paths(
    error_output: str,
    root: Path,
    changed_files: set[Path],
    ignore_dirs: Sequence[str],
) -> set[Path]:
    """Derive additional context paths from the latest failure."""
    discovered: set[Path] = set()
    discovered.update(changed_files)
    _, resolved = await resolve_files_from_output(error_output, root)
    discovered.update(resolved)

    dependencies: set[Path] = set()
    for path in discovered:
        try:
            deps = await asyncio.to_thread(get_import_dependencies, path, root)
            dependencies.update(deps)
        except Exception as exc:
            logger.debug("Dependency discovery failed for %s: %s", path, exc)

    if not discovered and not dependencies:
        match await scan_recent(root, 3, False, set(ignore_dirs)):
            case Err(err):
                logger.debug("Recent scan fallback failed: %s", err)
            case Ok(recents):
                discovered.update(entry.path for entry in recents[:3])

    return {
        security.validate_path(path, root) for path in (discovered | dependencies) if path.exists()
    }


# Re-export scan_recent for backwards compatibility with test patches
__all__ = [
    "build_dependency_section",
    "build_file_context_section",
    "collect_git_context",
    "collect_git_diff",
    "expand_context_paths",
    "load_constitution",
    "scan_recent",
]
