from __future__ import annotations

import shutil
import subprocess

from jpscripts.core.console import console


def fzf_select(
    lines: list[str],
    prompt: str = "> ",
    multi: bool = False,
    extra_args: list[str] | None = None,
) -> str | list[str] | None:
    """
    Run fzf interactively. Returns the selected line(s) or None if cancelled.
    """
    if not shutil.which("fzf"):
        console.print("[yellow]fzf not found. Please install it for interactive selection.[/yellow]")
        return None

    cmd = ["fzf", "--prompt", prompt]
    if multi:
        cmd.append("--multi")
    if extra_args:
        cmd.extend(extra_args)

    # fzf expects input via stdin
    proc = subprocess.run(
        cmd,
        input="\n".join(lines),
        text=True,
        capture_output=True,
    )

    if proc.returncode != 0:
        return None

    output = proc.stdout.strip()
    if not output:
        return None

    if multi:
        return output.splitlines()
    return output
