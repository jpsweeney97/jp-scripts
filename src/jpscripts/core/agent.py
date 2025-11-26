from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jpscripts.core import git as git_core
from jpscripts.core import git_ops
from jpscripts.core.console import get_logger
from jpscripts.core.context import gather_context, smart_read_context
from jpscripts.core.nav import scan_recent
from jpscripts.core.structure import generate_map, get_import_dependencies

logger = get_logger(__name__)

AGENT_PROMPT_TEMPLATE = (
    "<system_context>\n"
    "  <workspace_root>{workspace_root}</workspace_root>\n"
    "  <mode>God-Mode CLI</mode>\n"
    "  <git_context>\n"
    "    <branch>{branch}</branch>\n"
    "    <head>{head}</head>\n"
    "    <dirty>{dirty}</dirty>\n"
    "  </git_context>\n"
    "  <repository_map>\n"
    "  <![CDATA[\n"
    "{repository_map}\n"
    "  ]]>\n"
    "  </repository_map>\n"
    "</system_context>\n"
    "{diagnostic_section}"
    "{file_context_section}"
    "{dependency_section}"
    "{git_diff_section}"
    "<instruction>\n"
    "{instruction}\n"
    "</instruction>"
)


@dataclass
class PreparedPrompt:
    prompt: str
    attached_files: list[Path]


async def prepare_agent_prompt(
    base_prompt: str,
    *,
    root: Path,
    run_command: str | None,
    attach_recent: bool,
    include_diff: bool = False,
    ignore_dirs: Sequence[str],
    max_file_context_chars: int,
    max_command_output_chars: int,
) -> PreparedPrompt:
    """
    Builds a structured, XML-delimited prompt for Codex.
    """
    branch, commit, is_dirty = await _collect_git_context(root)

    repository_map = await asyncio.to_thread(generate_map, root, 3)
    safe_repo_map = _safe_cdata(repository_map)

    attached: list[Path] = []

    diagnostic_section = ""
    file_context_section = ""
    dependency_section = ""

    # 2. Command Output (Diagnostic)
    if run_command:
        output, detected_files = await gather_context(run_command, root)
        trimmed = output[-max_command_output_chars:]
        diagnostic_section = (
            "<diagnostic_command>\n"
            f"  <cmd>{run_command}</cmd>\n"
            "  <output>\n"
            f"{trimmed}\n"
            "  </output>\n"
            "</diagnostic_command>\n\n"
        )

        # Prioritize files detected in the stack trace
        detected_paths = list(sorted(detected_files))[:5]
        file_context_section, attached = await _build_file_context_section(detected_paths, max_file_context_chars)
        dependency_section = await _build_dependency_section(detected_paths, root, max_file_context_chars)

    # 3. Recent Context (Ambient)
    elif attach_recent:
        recents = await scan_recent(root, 3, False, set(ignore_dirs))
        recent_paths = [entry.path for entry in recents[:5]]
        file_context_section, attached = await _build_file_context_section(recent_paths, max_file_context_chars)
        dependency_section = await _build_dependency_section(recent_paths[:1], root, max_file_context_chars)

    # 4. Git Diff (Work in Progress)
    git_diff_section = ""
    if include_diff:
        diff_text = await _collect_git_diff(root, 10_000)
        if diff_text:
            safe_diff = _safe_cdata(diff_text)
            git_diff_section = (
                "<git_diff>\n"
                "<![CDATA[\n"
                f"{safe_diff}\n"
                "]]>\n"
                "</git_diff>\n\n"
            )
        else:
            git_diff_section = "<git_diff>NO CHANGES</git_diff>\n\n"

    prompt = AGENT_PROMPT_TEMPLATE.format(
        workspace_root=root,
        branch=branch,
        head=commit,
        dirty=is_dirty,
        repository_map=safe_repo_map,
        diagnostic_section=diagnostic_section,
        file_context_section=file_context_section,
        dependency_section=dependency_section,
        git_diff_section=git_diff_section,
        instruction=base_prompt.strip(),
    )

    return PreparedPrompt(prompt=prompt, attached_files=attached)


def _safe_cdata(content: str) -> str:
    """Escape CDATA terminators inside arbitrary content."""
    return content.replace("]]>", "]]]]><![CDATA[>")


async def _collect_git_context(root: Path) -> tuple[str, str, bool]:
    if not root.exists() or not (root / ".git").exists():
        return "(no repo)", "(no repo)", False

    try:
        repo = await git_core.AsyncRepo.open(root)
    except git_core.GitOperationError as exc:
        logger.error("Failed to open git repo at %s: %s", root, exc)
        return "(error)", "(error)", False
    except Exception as exc:
        logger.error("Failed to open git repo at %s: %s", root, exc)
        return "(error)", "(error)", False

    branch = "(unknown)"
    commit = "(unknown)"
    is_dirty = False

    try:
        status = await repo.status()
        branch = status.branch
        is_dirty = status.dirty
        _ = git_ops.format_status(status)
    except Exception as exc:
        logger.error("Failed to describe git status for %s: %s", root, exc)
        return "(error)", "(error)", False

    try:
        commit = await repo.head(short=True)
    except Exception as exc:
        logger.error("Failed to resolve git head for %s: %s", root, exc)
        commit = "(error)"

    return branch, commit, is_dirty


async def _build_file_context_section(paths: Sequence[Path], max_file_context_chars: int) -> tuple[str, list[Path]]:
    sections: list[str] = []
    attached: list[Path] = []
    for path in paths:
        snippet = await asyncio.to_thread(smart_read_context, path, max_file_context_chars)
        if not snippet:
            continue
        safe_snippet = _safe_cdata(snippet)
        sections.append(f"<file_context path='{path.name}'>\n<![CDATA[\n{safe_snippet}\n]]>\n</file_context>\n")
        attached.append(path)
    if not sections:
        return "", attached
    return "".join(sections) + "\n", attached


async def _build_dependency_section(paths: Sequence[Path], root: Path, max_file_context_chars: int) -> str:
    dependencies: set[Path] = set()
    for path in paths:
        deps = await asyncio.to_thread(get_import_dependencies, path, root)
        dependencies.update(deps)

    if not dependencies:
        return ""

    sections: list[str] = []
    for dep in sorted(dependencies):
        snippet = await asyncio.to_thread(smart_read_context, dep, max_file_context_chars)
        if not snippet:
            continue
        safe_snippet = _safe_cdata(snippet)
        sections.append(
            f"<dependency_context path='{dep.name}'>\n<![CDATA[\n{safe_snippet}\n]]>\n</dependency_context>\n"
        )

    return "".join(sections) + "\n"


async def _collect_git_diff(root: Path, max_chars: int) -> str | None:
    if not root.exists() or not (root / ".git").exists():
        return None

    try:
        proc = await asyncio.create_subprocess_shell(
            "git diff HEAD",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except asyncio.TimeoutError:
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
