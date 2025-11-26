from __future__ import annotations

from pathlib import Path

from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def get_git_status() -> str:
    """Return a summarized git status."""
    repo = await git_core.AsyncRepo.open(Path.cwd())
    status = await repo.status()
    return git_ops_core.format_status(status)


@tool()
@tool_error_handler
async def git_commit(message: str) -> str:
    """Stage all changes and create a commit."""
    repo = await git_core.AsyncRepo.open(Path.cwd())
    sha = await git_ops_core.commit_all(repo, message)
    status = await repo.status()
    formatted = git_ops_core.format_status(status)
    return f"Committed {sha} on {status.branch}\n{formatted}"
