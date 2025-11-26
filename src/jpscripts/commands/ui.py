from __future__ import annotations

import shutil
import subprocess
from typing import IO

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


def fzf_stream(
    input_stream: IO[bytes] | int,
    prompt: str = "> ",
    multi: bool = False,
    ansi: bool = False,
    extra_args: list[str] | None = None,
) -> str | list[str] | None:
    """Run fzf with a streaming stdin source (e.g., subprocess stdout)."""
    if not shutil.which("fzf"):
        console.print("[yellow]fzf not found. Please install it for interactive selection.[/yellow]")
        return None

    cmd = ["fzf", "--prompt", prompt]
    if ansi:
        cmd.append("--ansi")
    if multi:
        cmd.append("--multi")
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        stdin=input_stream,
        text=False,
        capture_output=True,
    )

    if proc.returncode != 0:
        return None

    output = proc.stdout.decode("utf-8", errors="replace").strip()
    if not output:
        return None

    if multi:
        return output.splitlines()
    return output
