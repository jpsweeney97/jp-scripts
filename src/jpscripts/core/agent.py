from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jpscripts.core import git as git_core
from jpscripts.core import git_ops
from jpscripts.core.context import gather_context, read_file_context
from jpscripts.core.nav import scan_recent


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

    # 1. System Pulse
    prompt = (
        f"<system_context>\n"
        f"  <workspace_root>{root}</workspace_root>\n"
        f"  <mode>God-Mode CLI</mode>\n"
        f"  <git_context>\n"
        f"    <branch>{branch}</branch>\n"
        f"    <head>{commit}</head>\n"
        f"    <dirty>{is_dirty}</dirty>\n"
        f"  </git_context>\n"
        f"</system_context>\n\n"
    )

    attached: list[Path] = []

    # 2. Command Output (Diagnostic)
    if run_command:
        output, detected_files = await gather_context(run_command, root)
        trimmed = output[-max_command_output_chars:]
        prompt += (
            f"<diagnostic_command>\n"
            f"  <cmd>{run_command}</cmd>\n"
            f"  <output>\n{trimmed}\n  </output>\n"
            f"</diagnostic_command>\n\n"
        )

        # Prioritize files detected in the stack trace
        for path in sorted(detected_files)[:5]:
            snippet = read_file_context(path, max_file_context_chars)
            if snippet:
                prompt += f"<file_context path='{path.name}'>\n<![CDATA[\n{snippet}\n]]>\n</file_context>\n"
                attached.append(path)

    # 3. Recent Context (Ambient)
    elif attach_recent:
        recents = await scan_recent(root, 3, False, set(ignore_dirs))
        for entry in recents[:5]:
            snippet = read_file_context(entry.path, max_file_context_chars)
            if snippet:
                prompt += f"<file_context path='{entry.path.name}'>\n<![CDATA[\n{snippet}\n]]>\n</file_context>\n"
                attached.append(entry.path)

    # 4. Git Diff (Work in Progress)
    if include_diff:
        diff_text = await _collect_git_diff(root, 10_000)
        if diff_text:
            prompt += (
                "<git_diff>\n"
                "<![CDATA[\n"
                f"{diff_text}\n"
                "]]>\n"
                "</git_diff>\n\n"
            )
        else:
            prompt += "<git_diff>NO CHANGES</git_diff>\n\n"

    # 4. The User Instruction
    prompt += f"\n<instruction>\n{base_prompt.strip()}\n</instruction>"

    return PreparedPrompt(prompt=prompt, attached_files=attached)


async def _collect_git_context(root: Path) -> tuple[str, str, bool]:
    if not root.exists() or not (root / ".git").exists():
        return "(no repo)", "(no repo)", False

    try:
        repo = await asyncio.to_thread(git_core.open_repo, root)
    except Exception:
        return "(unknown)", "(unknown)", False

    branch = "(unknown)"
    commit = "(unknown)"
    is_dirty = False

    try:
        status = await asyncio.to_thread(git_core.describe_status, repo)
        branch = status.branch
        is_dirty = status.dirty
        _ = git_ops.format_status(status)
    except Exception:
        pass

    try:
        commit = await asyncio.to_thread(repo.git.rev_parse, "--short", "HEAD")
        commit = str(commit).strip()
    except Exception:
        try:
            commit = repo.head.commit.hexsha[:7]
        except Exception:
            commit = "(unknown)"

    return branch, commit, is_dirty


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
