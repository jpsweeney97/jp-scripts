from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    ignore_dirs: Sequence[str],
    max_file_context_chars: int,
    max_command_output_chars: int,
) -> PreparedPrompt:
    """
    Build the Codex prompt with optional diagnostic command output or recent file snippets.
    """
    prompt = base_prompt.strip()
    attached: list[Path] = []
    workspace_root = root.expanduser()

    if run_command:
        output, detected_files = await gather_context(run_command, workspace_root)
        trimmed_output = output[-max_command_output_chars:] if max_command_output_chars > 0 else ""
        if trimmed_output:
            prompt += f"\n\nCommand `{run_command}` Output:\n```\n{trimmed_output}\n```"

        for path in sorted(detected_files)[:5]:
            snippet = read_file_context(path, max_file_context_chars)
            if snippet:
                prompt += f"\n\nFile: {path}\n```\n{snippet}\n```"
                attached.append(path)
    elif attach_recent:
        recents = await scan_recent(
            workspace_root,
            max_depth=3,
            include_dirs=False,
            ignore_dirs=set(ignore_dirs),
        )
        for entry in recents[:5]:
            snippet = read_file_context(entry.path, max_file_context_chars)
            if snippet:
                prompt += f"\n\nFile: {entry.path}\n```\n{snippet}\n```"
                attached.append(entry.path)

    return PreparedPrompt(prompt=prompt, attached_files=attached)
