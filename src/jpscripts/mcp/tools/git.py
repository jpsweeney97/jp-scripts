from __future__ import annotations

import asyncio
from pathlib import Path

from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.core.runtime import get_runtime
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


@tool()
@tool_error_handler
async def get_workspace_status(max_depth: int = 2) -> str:
    """Summarize branch status for repositories in the workspace.

    Args:
        max_depth: Depth to search for git repositories under workspace_root.

    Returns:
        Formatted summary lines containing repo name, branch, and ahead/behind counts.
    """
    ctx = get_runtime()
    root = ctx.workspace_root
    repos = list(git_core.iter_git_repos(root, max_depth=max_depth))
    if not repos:
        return f"No git repositories found under {root}."

    async def _describe_repo(path: Path) -> list[str]:
        try:
            repo = await git_core.AsyncRepo.open(path)
            branches = await git_ops_core.branch_statuses(repo)
        except Exception as exc:
            return [f"{path.name} | error | {exc}"]

        lines: list[str] = []
        for branch in branches:
            if branch.error:
                lines.append(f"{path.name} | {branch.name} | error: {branch.error}")
            else:
                lines.append(f"{path.name} | {branch.name} | +{branch.ahead}/-{branch.behind}")
        return lines

    results = await asyncio.gather(*(_describe_repo(repo_path) for repo_path in repos))
    flattened = [line for repo_lines in results for line in repo_lines]
    if not flattened:
        return "No branch data found."
    return "\n".join(flattened)
