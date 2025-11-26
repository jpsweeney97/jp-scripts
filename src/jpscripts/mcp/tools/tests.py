from __future__ import annotations

import asyncio

from jpscripts.mcp import get_config, tool


@tool()
async def run_tests(target: str = ".", verbose: bool = False) -> str:
    """
    Run pytest on a specific target (directory or file) and return the results.
    Use this to verify fixes.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    cmd = ["pytest"]
    if verbose:
        cmd.append("-vv")
    cmd.append(target)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cfg.workspace_root.expanduser(),
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
