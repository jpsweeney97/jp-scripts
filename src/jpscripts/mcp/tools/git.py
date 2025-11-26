from __future__ import annotations

import asyncio
from pathlib import Path

from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.mcp import tool


@tool()
async def get_git_status() -> str:
    """Return a summarized git status."""
    try:
        return await asyncio.to_thread(_describe_status)
    except Exception as e:
        return f"Error retrieving git status: {str(e)}"


@tool()
async def git_commit(message: str) -> str:
    """Stage all changes and create a commit."""
    try:
        sha, branch, formatted = await asyncio.to_thread(_commit_all, message)
        return f"Committed {sha} on {branch}\n{formatted}"
    except git_ops_core.GitOperationError as exc:
        return f"Git commit failed: {exc}"
    except Exception as e:
        return f"Error committing changes: {str(e)}"


def _describe_status() -> str:
    repo = git_core.open_repo(Path.cwd())
    status = git_core.describe_status(repo)
    return git_ops_core.format_status(status)


def _commit_all(message: str) -> tuple[str, str, str]:
    repo = git_core.open_repo(Path.cwd())
    sha = git_ops_core.commit_all(repo, message)
    status = git_core.describe_status(repo)
    return sha, status.branch, git_ops_core.format_status(status)
