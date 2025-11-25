from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

def _ensure_rg() -> str:
    binary = shutil.which("rg")
    if not binary:
        raise RuntimeError("ripgrep (rg) not found. Please install it.")
    return binary

def run_ripgrep(
    pattern: str,
    path: Path,
    context: int = 0,
    line_number: bool = False,
    follow: bool = False,
    pcre2: bool = False,
    extra_args: list[str] | None = None,
) -> str:
    """
    Execute ripgrep and return the standard output as a string.

    Args:
        pattern: Regex pattern to search for.
        path: Root path to search in.
        context: Number of context lines (-C).
        line_number: Show line numbers (-n).
        follow: Follow symlinks.
        pcre2: Use PCRE2 regex engine.
        extra_args: Additional CLI arguments for rg.
    """
    binary = _ensure_rg()
    path = path.expanduser()

    cmd = [binary, "--color=always", str(path)]

    # Arguments mapping
    if context > 0:
        cmd.append(f"-C{context}")
    if line_number:
        cmd.append("--line-number")
    if follow:
        cmd.append("--follow")
    if pcre2:
        cmd.append("--pcre2")
    if extra_args:
        cmd.extend(extra_args)

    # Pattern must often be last or near end depending on rg syntax,
    # but standard is `rg [options] PATTERN [PATH]`
    # We constructed `rg [options] PATH` above, which is wrong for `rg`.
    # `rg` syntax is `rg [OPTIONS] PATTERN [PATH ...]`.

    # Correction:
    cmd = [binary, "--color=always"]
    if context > 0:
        cmd.append(f"-C{context}")
    if line_number:
        cmd.append("--line-number")
    if follow:
        cmd.append("--follow")
    if pcre2:
        cmd.append("--pcre2")
    if extra_args:
        cmd.extend(extra_args)

    cmd.append(pattern)
    cmd.append(str(path))

    try:
        # Capture stdout and stderr
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ripgrep execution failed.")

    # rg returns 0 on match, 1 on no match, 2 on error
    if proc.returncode == 2:
        raise RuntimeError(f"ripgrep error: {proc.stderr.strip()}")

    return proc.stdout.strip()

def get_ripgrep_cmd(
    pattern: str,
    path: Path,
    context: int = 0,
    line_number: bool = False,
    follow: bool = False,
    pcre2: bool = False,
) -> list[str]:
    """
    Return the command list for usage in interactive pipes (e.g. fzf).
    """
    binary = _ensure_rg()
    cmd = [binary, "--color=always"]
    if context > 0:
        cmd.append(f"-C{context}")
    if line_number:
        cmd.append("--line-number")
    if follow:
        cmd.append("--follow")
    if pcre2:
        cmd.append("--pcre2")

    cmd.append(pattern)
    cmd.append(str(path.expanduser()))
    return cmd
