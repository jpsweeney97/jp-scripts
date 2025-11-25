from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.core import nav as nav_core
from jpscripts.core import search as search_core
from jpscripts.core import system as system_core
from jpscripts.core.config import load_config
from jpscripts.core.console import get_logger
from jpscripts.core import memory as memory_core
from jpscripts.core.notes_impl import append_to_daily_note
from jpscripts.core.context import read_file_context
from jpscripts.core.security import validate_path

# Initialize logger
logger = get_logger("mcp")

# Initialize server
mcp = FastMCP("jpscripts")

# Load config
config = None
try:
    config, _ = load_config()
    logger.info("MCP Server loaded config from %s", config.notes_dir)
except Exception as e:
    logger.error("Failed to load config during MCP startup", exc_info=e)

# --- OS PRIMITIVES (God Mode Enablers) ---

@mcp.tool()
async def read_file(path: str) -> str:
    """
    Read the content of a file (truncated to JP_MAX_FILE_CONTEXT_CHARS).
    Use this to inspect code, config files, or logs.
    """
    try:
        if config is None:
            return "Config not loaded."

        root = config.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        target = validate_path(candidate, root)
        if not target.exists():
            return f"Error: File {target} does not exist."
        if not target.is_file():
            return f"Error: {target} is not a file."

        max_chars = getattr(config, "max_file_context_chars", 50000)
        content = await asyncio.to_thread(read_file_context, target, max_chars)
        if content is None:
            return f"Error: Could not read file {target} (unsupported encoding or IO error)."
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
async def list_directory(path: str) -> str:
    """
    List contents of a directory (like ls).
    Returns a list of 'd: dir_name' and 'f: file_name'.
    """
    try:
        if config is None:
            return "Config not loaded."

        root = config.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        target = validate_path(candidate, root)
        if not target.exists():
            return f"Error: Path {target} does not exist."
        if not target.is_dir():
            return f"Error: {target} is not a directory."

        def _ls():
            entries = []
            with os.scandir(target) as it:
                for entry in it:
                    prefix = "d" if entry.is_dir() else "f"
                    entries.append(f"{prefix}: {entry.name}")
            return "\n".join(sorted(entries))

        return await asyncio.to_thread(_ls)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

# --- SEARCH ---

@mcp.tool()
async def search_codebase(pattern: str, path: str = ".") -> str:
    """
    Search the codebase using ripgrep (grep).
    Returns the raw text matches with line numbers.
    """
    try:
        if config is None:
            return "Config not loaded."

        max_chars = getattr(config, "max_file_context_chars", 50000)
        root = config.workspace_root.expanduser()
        base = Path(path)
        candidate = base if base.is_absolute() else root / base
        search_root = validate_path(candidate, root)
        safe_pattern = re.escape(pattern)

        # Offload the subprocess blocking call to a thread
        result = await asyncio.to_thread(
            search_core.run_ripgrep,
            safe_pattern,
            search_root,
            line_number=True,
            context=1,
            max_chars=max_chars,
        )
        if not result:
            return "No matches found."

        return result
    except Exception as e:
        return f"Error searching codebase: {str(e)}"


# --- NAVIGATION ---

@mcp.tool()
async def list_recent_files(limit: int = 20) -> str:
    """List files modified recently in the current workspace root and surface related memories."""
    try:
        if config is None:
            return "Config not loaded."
        root = config.workspace_root.expanduser()

        # 1. Parallel execution: Scan Files AND Query Memory
        # We offload the blocking memory query to a thread
        scan_task = nav_core.scan_recent(
            root,
            max_depth=3,
            include_dirs=False,
            ignore_dirs=set(config.ignore_dirs)
        )

        # Generate a heuristic query based on recent files isn't possible until we have them.
        # So we await scan first.
        entries = await scan_task
        lines = [f"{e.path.relative_to(root) if e.path.is_relative_to(root) else e.path}" for e in entries[:limit]]

        # 2. Query Memory (Non-Blocking)
        query_hint = " ".join(Path(line).stem for line in lines[:5]) or Path.cwd().name

        # FIX: Wrap synchronous IO in to_thread
        memories = await asyncio.to_thread(
            memory_core.query_memory,
            query_hint,
            limit=3,
            config=config
        )

        mem_block = "\n".join(memories) if memories else "No related memories."
        recent_block = "\n".join(lines) if lines else "No recent files found."

        return f"Recent files:\n{recent_block}\n\nRelevant memories:\n{mem_block}"
    except Exception as e:
        return f"Error scanning recent files: {str(e)}"

@mcp.tool()
async def list_projects() -> str:
    """List known projects (via zoxide)."""
    try:
        paths = await nav_core.get_zoxide_projects()
        return "\n".join(paths) if paths else "No projects found."
    except Exception as e:
        return f"Error listing projects: {str(e)}"

# --- NOTES ---

@mcp.tool()
def append_daily_note(message: str) -> str:
    """Append a log entry to the user's daily note system."""
    try:
        if config is None:
            return "Config not loaded."
        target_dir = config.notes_dir.expanduser()
        path = append_to_daily_note(target_dir, message)
        return f"Successfully logged to daily note: {path}"
    except Exception as e:
        return f"Error appending note: {str(e)}"

# --- SYSTEM ---

@mcp.tool()
def list_processes(name_filter: str | None = None, port_filter: int | None = None) -> str:
    """List running processes."""
    try:
        procs = system_core.find_processes(name_filter, port_filter)
        if not procs:
            return "No matching processes found."
        lines = [f"{p.pid} - {p.name} ({p.username}) [{p.cmdline}]" for p in procs[:50]]
        if len(procs) > 50:
            lines.append(f"... and {len(procs) - 50} more.")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing processes: {str(e)}"

@mcp.tool()
def kill_process(pid: int, force: bool = False) -> str:
    """Kill a process by PID."""
    try:
        result = system_core.kill_process(pid, force)
        return f"Process {pid}: {result}"
    except Exception as e:
        return f"Error killing process {pid}: {str(e)}"

# --- GIT ---

@mcp.tool()
def get_git_status() -> str:
    """Return a summarized git status."""
    try:
        repo = git_core.open_repo(Path.cwd())
        status = git_core.describe_status(repo)
        return git_ops_core.format_status(status)
    except Exception as e:
        return f"Error retrieving git status: {str(e)}"

@mcp.tool()
def git_commit(message: str) -> str:
    """Stage all changes and create a commit."""
    try:
        repo = git_core.open_repo(Path.cwd())
        sha = git_ops_core.commit_all(repo, message)
        status = git_core.describe_status(repo)
        return f"Committed {sha} on {status.branch}\n{git_ops_core.format_status(status)}"
    except git_ops_core.GitOperationError as exc:
        return f"Git commit failed: {exc}"
    except Exception as e:
        return f"Error committing changes: {str(e)}"

# --- WEB ---

@mcp.tool()
def fetch_url_content(url: str) -> str:
    """Fetch and parse a webpage into clean Markdown."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Failed to download {url}"
        text = trafilatura.extract(downloaded, include_comments=False, output_format="markdown", url=url)
        return text if text else "Error: Could not extract content."
    except ImportError:
        return "Error: trafilatura not installed. Run `pip install jpscripts[full]`"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


# --- MEMORY ---

@mcp.tool()
def remember(fact: str, tags: str | None = None) -> str:
    """
    Save a fact or lesson to the persistent memory store.
    Tags can be provided as a comma-separated list.
    """
    if config is None:
        return "Config not loaded."

    tag_list = [t.strip() for t in (tags.split(",") if tags else []) if t.strip()]
    entry = memory_core.save_memory(fact, tag_list, config=config)
    return f"Saved memory at {entry.ts}"


@mcp.tool()
def recall(query: str, limit: int = 5) -> str:
    """
    Retrieve the most relevant memories for a query.
    """
    if config is None:
        return "Config not loaded."

    results = memory_core.query_memory(query, limit=limit, config=config)
    return "\n".join(results) if results else "No matching memories."

if __name__ == "__main__":
    mcp.run()
