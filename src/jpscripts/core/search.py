from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
            assert proc.stdout is not None and proc.stderr is not None
            stdout_pipe = proc.stdout
            stderr_pipe = proc.stderr

            with ThreadPoolExecutor(max_workers=1) as executor:
                stderr_future = executor.submit(stderr_pipe.read)

                truncated = False
                for chunk in iter(lambda: stdout_pipe.read(4096), ""):
                    if chunk == "":
                        break
                    chunks.append(chunk)
                    bytes_read += len(chunk)
                    if max_chars is not None and bytes_read >= max_chars:
                        truncated = True
                        proc.terminate()
                        break

                stdout = "".join(chunks)
                stderr = stderr_future.result()
                proc.wait()

            if proc.returncode == 2:
                raise RuntimeError(f"ripgrep error: {stderr.strip()}")

            if truncated and max_chars is not None:
                return stdout[:max_chars] + "\n... [truncated]"

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

async def scan_todos(path: Path, types: str = "TODO|FIXME|HACK|BUG") -> list[TodoEntry]:
    """
    Scan for TODO markers and return structured data using ripgrep --json.
    OPTIMIZED: Uses async streaming to avoid loading massive outputs into memory.
    """
    binary = _ensure_rg()
    path = path.expanduser()

    # Use create_subprocess_exec for async streaming
    # We use --json to safely parse output
    proc = await asyncio.create_subprocess_exec(
        binary, "--json", types, str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    entries: list[TodoEntry] = []

    # Stream output line by line
    # This prevents memory spikes on large repos (e.g. monorepos)
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break

        try:
            # Decode line-by-line
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            data = json.loads(line_str)
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

    await proc.wait()

    # We ignore return codes because grep returns 1 if no matches found, which is valid.
    return entries
