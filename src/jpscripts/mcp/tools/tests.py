from __future__ import annotations

import asyncio

from pathlib import Path

from jpscripts.core.security import validate_path
from jpscripts.mcp import get_config, tool, tool_error_handler


@tool()
@tool_error_handler
async def run_tests(target: str = ".", verbose: bool = False) -> str:
    """
    Run pytest on a specific target (directory or file) and return the results.
    Use this to verify fixes.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    root = cfg.workspace_root.expanduser()
    candidate = Path(target)
    resolved_target = candidate if candidate.is_absolute() else root / candidate
    try:
        safe_target = validate_path(resolved_target, root)
    except PermissionError as exc:
        return f"Error: {exc}"
    if not safe_target.exists():
        return f"Error: Target {safe_target} does not exist."

    cmd = ["pytest"]
    if verbose:
        cmd.append("-vv")
    cmd.append(str(safe_target))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        output = (stdout + stderr).decode(errors="replace")

        if proc.returncode == 0:
            return f"Tests Passed:\n{output}"
        return f"Tests Failed (Exit Code {proc.returncode}):\n{output[-5000:]}"
    except Exception as e:
        return f"Error executing tests: {e}"
