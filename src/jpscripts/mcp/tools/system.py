"""MCP system tools for process and shell operations.

Provides tools for system interactions:
    - list_processes: List running processes
    - run_command: Execute validated shell commands
    - Command validation for safety
"""

from __future__ import annotations

from jpscripts.core import system as system_core
from jpscripts.core.engine import AUDIT_PREFIX, run_safe_shell
from jpscripts.core.result import Err, Ok
from jpscripts.core.runtime import get_runtime
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def list_processes(name_filter: str | None = None, port_filter: int | None = None) -> str:
    """List running processes."""
    match await system_core.find_processes(name_filter, port_filter):
        case Err(err):
            return f"Error listing processes: {err.message}"
        case Ok(procs):
            if not procs:
                return "No matching processes found."
            lines = [f"{p.pid} - {p.name} ({p.username}) [{p.cmdline}]" for p in procs[:50]]
            if len(procs) > 50:
                lines.append(f"... and {len(procs) - 50} more.")
            return "\n".join(lines)


@tool()
@tool_error_handler
async def kill_process(pid: int, force: bool = False) -> str:
    """Kill a process by PID."""
    _ = get_runtime()
    match await system_core.kill_process_async(pid, force):
        case Err(err):
            return f"Error killing process {pid}: {err.message}"
        case Ok(result):
            return f"Process {pid}: {result}"


@tool()
@tool_error_handler
async def run_shell(command: str) -> str:
    """
    Execute a safe, sandboxed command without shell interpolation.
    Only allows read-only inspection commands.
    """
    ctx = get_runtime()

    if not isinstance(command, str) or not command.strip():
        return "Invalid command argument."

    return await run_safe_shell(
        command=command,
        root=ctx.workspace_root,
        audit_prefix=AUDIT_PREFIX,
        config=ctx.config,
    )
