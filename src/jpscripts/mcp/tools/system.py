from __future__ import annotations

from jpscripts.core import system as system_core
from jpscripts.core.engine import AUDIT_PREFIX, run_safe_shell
from jpscripts.core.runtime import get_runtime
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def list_processes(name_filter: str | None = None, port_filter: int | None = None) -> str:
    """List running processes."""
    try:
        procs = await system_core.find_processes(name_filter, port_filter)
        if not procs:
            return "No matching processes found."
        lines = [f"{p.pid} - {p.name} ({p.username}) [{p.cmdline}]" for p in procs[:50]]
        if len(procs) > 50:
            lines.append(f"... and {len(procs) - 50} more.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing processes: {str(e)}"


@tool()
@tool_error_handler
async def kill_process(pid: int, force: bool = False) -> str:
    """Kill a process by PID."""
    try:
        ctx = get_runtime()
        result = await system_core.kill_process_async(pid, force, ctx.config)
        return f"Process {pid}: {result}"
    except Exception as e:
        return f"Error killing process {pid}: {str(e)}"


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
    )
