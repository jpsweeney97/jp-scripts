from __future__ import annotations

import asyncio

from jpscripts.core import system as system_core
from jpscripts.mcp import get_config
from jpscripts.mcp import tool


@tool()
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
async def kill_process(pid: int, force: bool = False) -> str:
    """Kill a process by PID."""
    try:
        cfg = get_config()
        result = await system_core.kill_process_async(pid, force, cfg)
        return f"Process {pid}: {result}"
    except Exception as e:
        return f"Error killing process {pid}: {str(e)}"
