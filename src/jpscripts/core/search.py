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
    max_chars: int | None = None,
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
        max_chars: Optional cap on bytes read from stdout.
    """
    binary = _ensure_rg()
    path = path.expanduser()

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
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=4096,
        ) as proc:
            chunks: list[str] = []
            bytes_read = 0
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(4096), ""):
                if chunk == "":
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
                if max_chars is not None and bytes_read >= max_chars:
                    proc.terminate()
                    return "".join(chunks)[:max_chars] + "\n... [truncated]"

            stdout = "".join(chunks)
            stderr = proc.stderr.read() if proc.stderr else ""
            proc.wait()

            if proc.returncode == 2:
                raise RuntimeError(f"ripgrep error: {stderr.strip()}")

            return stdout.strip()
    except FileNotFoundError:
        raise RuntimeError("ripgrep execution failed.")

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
