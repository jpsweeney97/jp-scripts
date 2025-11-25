from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
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
    """
    binary = _ensure_rg()
    path = path.expanduser()

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

@dataclass
class TodoEntry:
    file: str
    line: int
    type: str
    text: str

def scan_todos(path: Path, types: str = "TODO|FIXME|HACK|BUG") -> list[TodoEntry]:
    """
    Scan for TODO markers and return structured data using ripgrep --json.
    """
    binary = _ensure_rg()
    path = path.expanduser()

    # We use --json to safely parse output
    cmd = [binary, "--json", types, str(path)]

    entries: list[TodoEntry] = []

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise RuntimeError("ripgrep execution failed.")

    for line in proc.stdout.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "match":
            # Extract structured data from rg JSON event
            data_data = data.get("data", {})
            file_path = data_data.get("path", {}).get("text", "")
            line_num = data_data.get("line_number", 0)

            # Identify which tag matched
            submatches = data_data.get("submatches", [])
            tag_type = submatches[0].get("match", {}).get("text", "TODO") if submatches else "TODO"

            # Get full line text
            line_text = data_data.get("lines", {}).get("text", "").strip()

            entries.append(TodoEntry(
                file=file_path,
                line=line_num,
                type=tag_type,
                text=line_text
            ))

    return entries
