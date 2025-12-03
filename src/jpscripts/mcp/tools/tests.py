"""MCP test tools for running pytest.

Provides tools for test execution:
    - run_tests: Execute pytest on targets
    - Path validation for security
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jpscripts.core.result import Err
from jpscripts.core.runtime import get_runtime
from jpscripts.core.security import validate_path_async
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def run_tests(target: str = ".", verbose: bool = False) -> str:
    """
    Run pytest on a specific target (directory or file) and return the results.
    Use this to verify fixes.
    """
    ctx = get_runtime()
    root = ctx.workspace_root
    candidate = Path(target)
    resolved_target = candidate if candidate.is_absolute() else root / candidate
    path_result = await validate_path_async(resolved_target, root)
    if isinstance(path_result, Err):
        return f"Error: {path_result.error.message}"
    safe_target = path_result.value
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
    except Exception as exc:
        return f"Error executing tests: {exc}"
