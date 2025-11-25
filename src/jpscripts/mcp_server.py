import asyncio
from pathlib import Path
import json

from mcp.server.fastmcp import FastMCP
from jpscripts.core.config import load_config
from jpscripts.core.notes_impl import append_to_daily_note
from jpscripts.core.console import get_logger

# Import new core modules
from jpscripts.core import system as system_core
from jpscripts.core import nav as nav_core
from jpscripts.core import search as search_core

# Initialize logger for observability
logger = get_logger("mcp")

# Initialize the server
mcp = FastMCP("jpscripts")

# Load config independently of Typer
try:
    config, _ = load_config()
    logger.info(f"MCP Server loaded config from {config.notes_dir}")
except Exception as e:
    logger.error("Failed to load config during MCP startup", exc_info=e)
    # We continue; config might be needed for tools, handled individually if so.

# --- NOTES ---

@mcp.tool()
def append_daily_note(message: str) -> str:
    """
    Append a log entry to the user's daily note system.
    This uses the configured 'notes_dir' from ~/.jpconfig.
    """
    try:
        target_dir = config.notes_dir.expanduser()
        logger.debug(f"Appending note to {target_dir}: {message[:20]}...")
        path = append_to_daily_note(target_dir, message)
        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        logger.error("MCP Tool Execution Failed: append_daily_note", exc_info=e)
        return f"Error appending note: {str(e)}"

# --- SYSTEM ---

@mcp.tool()
def list_processes(name_filter: str | None = None, port_filter: int | None = None) -> str:
    """
    List running processes, optionally filtering by name or port.
    Returns a formatted string list of 'PID - Name (User) [Command]'.
    """
    try:
        procs = system_core.find_processes(name_filter, port_filter)
        if not procs:
            return "No matching processes found."

        # Format for LLM consumption
        lines = [f"{p.pid} - {p.name} ({p.username}) [{p.cmdline}]" for p in procs[:50]]
        if len(procs) > 50:
            lines.append(f"... and {len(procs) - 50} more.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing processes: {str(e)}"

@mcp.tool()
def kill_process(pid: int, force: bool = False) -> str:
    """
    Kill a process by PID. Set force=True to use SIGKILL.
    """
    try:
        result = system_core.kill_process(pid, force)
        return f"Process {pid}: {result}"
    except Exception as e:
        return f"Error killing process {pid}: {str(e)}"

# --- NAVIGATION ---

@mcp.tool()
def list_recent_files(limit: int = 20) -> str:
    """
    List files modified recently in the current workspace root.
    Useful for understanding active context.
    """
    try:
        root = config.workspace_root.expanduser()
        # We assume sensible defaults for the agent
        entries = nav_core.scan_recent(
            root,
            max_depth=3,
            include_dirs=False,
            ignore_dirs=set(config.ignore_dirs)
        )

        lines = [f"{e.path.relative_to(root) if e.path.is_relative_to(root) else e.path}" for e in entries[:limit]]
        return "\n".join(lines) if lines else "No recent files found."
    except Exception as e:
        return f"Error scanning recent files: {str(e)}"

@mcp.tool()
def list_projects() -> str:
    """List known projects (via zoxide) to find valid paths."""
    try:
        paths = nav_core.get_zoxide_projects()
        return "\n".join(paths) if paths else "No projects found."
    except Exception as e:
        return f"Error listing projects: {str(e)}"

# --- SEARCH ---

@mcp.tool()
def search_codebase(pattern: str, path: str = ".") -> str:
    """
    Search the codebase using ripgrep (grep).
    Returns the raw text matches with line numbers.
    """
    try:
        # Resolve path relative to workspace if it's "."
        search_root = Path(path).expanduser()
        if str(search_root) == ".":
             search_root = Path.cwd()

        result = search_core.run_ripgrep(
            pattern,
            search_root,
            line_number=True,
            context=1 # Provide 1 line of context for the LLM
        )
        return result if result else "No matches found."
    except Exception as e:
        return f"Error searching codebase: {str(e)}"

if __name__ == "__main__":
    mcp.run()
