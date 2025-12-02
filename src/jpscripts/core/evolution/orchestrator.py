"""Evolution orchestration logic.

This module provides:
- cleanup_branch: Clean up failed evolution branch
- abort_evolution: Abort evolution with cleanup and memory logging
- run_evolution: Execute full evolution workflow
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path

from jpscripts.agent import PreparedPrompt, run_repair_loop
from jpscripts.analysis.complexity import TechnicalDebtScore, calculate_debt_scores
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result
from jpscripts.git import client as git_core
from jpscripts.memory import save_memory
from jpscripts.providers import CompletionOptions, Message, infer_provider_type
from jpscripts.providers.factory import get_provider

from .pr import create_evolution_pr
from .prompting import build_optimizer_prompt
from .types import EvolutionError, EvolutionResult
from .verification import run_verification

logger = get_logger(__name__)


async def cleanup_branch(repo: git_core.AsyncRepo, branch_name: str) -> None:
    """Clean up a failed evolution branch by returning to main.

    Args:
        repo: Git repository handle
        branch_name: Name of the branch to clean up
    """
    await repo.run_git("checkout", "main")
    await repo.run_git("branch", "-D", branch_name)


async def abort_evolution(
    repo: git_core.AsyncRepo,
    branch_name: str,
    message: str,
    memory_tags: list[str],
    config: AppConfig,
    reset_hard: bool = False,
) -> None:
    """Abort evolution with cleanup and optional memory logging.

    Args:
        repo: Git repository handle
        branch_name: Name of the evolution branch
        message: Error message to log
        memory_tags: Tags for memory persistence
        config: Application configuration
        reset_hard: Whether to reset hard before cleanup
    """
    if reset_hard:
        await repo.run_git("reset", "--hard", "main")
    await cleanup_branch(repo, branch_name)
    if memory_tags:
        await asyncio.to_thread(save_memory, message, memory_tags, config=config)


async def run_evolution(
    config: AppConfig,
    dry_run: bool,
    model: str | None,
    threshold: float,
) -> Result[EvolutionResult | None, EvolutionError]:
    """Execute the full evolution workflow.

    Args:
        config: Application configuration
        dry_run: If True, analyze without making changes
        model: Optional model override for optimization
        threshold: Minimum debt score to trigger optimization

    Returns:
        Ok(None) if no optimization needed (below threshold or no files)
        Ok(EvolutionResult) on successful optimization
        Err(EvolutionError) on failure
    """
    root = Path(config.user.workspace_root).expanduser().resolve()

    # Step 1: Check clean git state
    match await git_core.AsyncRepo.open(root):
        case Err(git_err):
            return Err(EvolutionError(f"Git error: {git_err}"))
        case Ok(repo):
            pass

    match await repo.status():
        case Err(status_err):
            return Err(EvolutionError(f"Status error: {status_err}"))
        case Ok(status):
            if status.dirty:
                return Err(EvolutionError("Workspace is dirty. Commit or stash changes first."))

    # Step 2: Calculate debt scores
    match await calculate_debt_scores(root, config):
        case Err(debt_err):
            return Err(EvolutionError(f"Analysis failed: {debt_err}"))
        case Ok(scores):
            if not scores:
                return Ok(None)  # No files require optimization

    # Step 3: Select target
    target = scores[0]
    if target.debt_score < threshold:
        return Ok(None)  # Below threshold

    if dry_run:
        return Ok(None)  # Dry run - no changes

    # Step 4: Create branch
    branch_name = f"evolve/{target.path.stem}-optimization"

    try:
        await repo.run_git("checkout", "-b", branch_name)
    except Exception as exc:
        return Err(EvolutionError(f"Failed to create branch: {exc}"))

    # Step 5: Launch optimizer agent
    optimizer_prompt = build_optimizer_prompt(target)
    target_model = model or config.ai.default_model

    # Determine provider from model
    ptype = infer_provider_type(target_model)
    provider = get_provider(config, model_id=target_model, provider_type=ptype)

    async def fetch_response(prepared: PreparedPrompt) -> str:
        messages = [Message(role="user", content=prepared.prompt)]
        options = CompletionOptions(
            temperature=prepared.temperature,
            reasoning_effort=prepared.reasoning_effort,
            max_tokens=8192,
        )
        response = await provider.complete(messages, model=target_model, options=options)
        return response.content

    def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
        return fetch_response(prepared)

    # Use py_compile as the validation command
    validation_cmd = f"python -m py_compile {target.path}"

    success = await run_repair_loop(
        base_prompt=optimizer_prompt,
        command=validation_cmd,
        model=target_model,
        attach_recent=False,
        include_diff=True,
        fetch_response=fetcher,
        app_config=config,
        workspace_root=root,
        auto_archive=True,
        max_retries=3,
        keep_failed=False,
    )

    if not success:
        await cleanup_branch(repo, branch_name)
        return Err(EvolutionError("Optimization failed. Returning to main branch."))

    # Step 6: Determine changed files for targeted testing
    match await repo.run_git("diff", "--name-only", "main..HEAD"):
        case Err(err):
            await cleanup_branch(repo, branch_name)
            return Err(EvolutionError(f"Failed to detect changed files: {err}"))
        case Ok(diff_output):
            changed_paths = [line.strip() for line in diff_output.splitlines() if line.strip()]

    verification_result = await run_verification(changed_paths, root, config)
    match verification_result:
        case Err(verif_err):
            await abort_evolution(
                repo,
                branch_name,
                str(verif_err),
                ["evolve", "failure", "tests"],
                config,
                reset_hard=True,
            )
            return Err(verif_err)
        case Ok(verification):
            if not verification.success:
                await abort_evolution(
                    repo,
                    branch_name,
                    f"Verification failed (exit {verification.exit_code}).",
                    ["evolve", "failure", "tests"],
                    config,
                    reset_hard=True,
                )
                return Err(
                    EvolutionError(f"Verification failed (exit {verification.exit_code}).")
                )

    # Step 7: Create PR
    pr_result = await create_evolution_pr(
        repo,
        target,
        branch_name,
        root,
        config,
        verification,
    )

    return Ok(
        EvolutionResult(
            target=target,
            branch_name=branch_name,
            pr_url=pr_result.pr_url,
        )
    )


def get_debt_scores(
    scores: list[TechnicalDebtScore],
) -> list[tuple[str, float, int, float]]:
    """Get debt score summary for display.

    Args:
        scores: List of technical debt scores

    Returns:
        List of (filename, complexity, fix_freq, debt_score) tuples
    """
    return [
        (score.path.name, score.complexity_score, score.fix_frequency, score.debt_score)
        for score in scores
    ]


__all__ = [
    "abort_evolution",
    "cleanup_branch",
    "get_debt_scores",
    "run_evolution",
]
