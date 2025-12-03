"""Evolution orchestration logic.

This module provides:
- EvolutionSession: Encapsulates state and methods for evolution workflow
- run_evolution: Entry point for executing evolution workflow
- cleanup_branch: Clean up failed evolution branch
- abort_evolution: Abort evolution with cleanup and memory logging
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
from .types import EvolutionError, EvolutionResult, VerificationResult
from .verification import run_verification

logger = get_logger(__name__)


class EvolutionSession:
    """Encapsulates state and workflow for a single evolution session.

    This class manages the lifecycle of an evolution operation, including:
    - Git state validation
    - Technical debt analysis
    - Branch management
    - Optimization execution
    - Verification and PR creation

    Attributes:
        config: Application configuration.
        root: Workspace root path.
        model: Model to use for optimization (defaults to config.ai.default_model).
        threshold: Minimum debt score to trigger optimization.
        repo: Git repository handle (set after initialization).
        target: Technical debt target for optimization (set after analysis).
        branch_name: Name of the evolution branch (set after branch creation).
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        model: str | None = None,
        threshold: float = 10.0,
    ) -> None:
        """Initialize an evolution session.

        Args:
            config: Application configuration.
            model: Optional model override for optimization.
            threshold: Minimum debt score to trigger optimization.
        """
        self.config = config
        self.root = Path(config.user.workspace_root).expanduser().resolve()
        self.model = model or config.ai.default_model
        self.threshold = threshold

        # Set during session lifecycle
        self.repo: git_core.AsyncRepo | None = None
        self.target: TechnicalDebtScore | None = None
        self.branch_name: str | None = None

    async def _check_git_state(self) -> Result[git_core.AsyncRepo, EvolutionError]:
        """Validate git state and return repository handle.

        Returns:
            Ok(AsyncRepo) if repository is clean and valid.
            Err(EvolutionError) if repository is dirty or invalid.
        """
        match await git_core.AsyncRepo.open(self.root):
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

        return Ok(repo)

    async def _analyze_debt(self) -> Result[TechnicalDebtScore | None, EvolutionError]:
        """Calculate debt scores and select optimization target.

        Returns:
            Ok(TechnicalDebtScore) if a valid target exists above threshold.
            Ok(None) if no files require optimization.
            Err(EvolutionError) on analysis failure.
        """
        match await calculate_debt_scores(self.root, self.config):
            case Err(debt_err):
                return Err(EvolutionError(f"Analysis failed: {debt_err}"))
            case Ok(scores):
                if not scores:
                    return Ok(None)

        target = scores[0]
        if target.debt_score < self.threshold:
            return Ok(None)

        return Ok(target)

    async def _create_branch(self) -> Result[str, EvolutionError]:
        """Create evolution branch for the target.

        Requires: self.target and self.repo must be set.

        Returns:
            Ok(branch_name) on success.
            Err(EvolutionError) on failure.
        """
        if self.target is None or self.repo is None:
            return Err(EvolutionError("Session not properly initialized"))

        branch_name = f"evolve/{self.target.path.stem}-optimization"

        try:
            await self.repo.run_git("checkout", "-b", branch_name)
        except Exception as exc:
            return Err(EvolutionError(f"Failed to create branch: {exc}"))

        return Ok(branch_name)

    async def _run_optimizer(self) -> Result[bool, EvolutionError]:
        """Execute the optimization agent.

        Requires: self.target, self.repo, and self.branch_name must be set.

        Returns:
            Ok(True) if optimization succeeded.
            Ok(False) if optimization failed (branch cleaned up).
            Err(EvolutionError) on unexpected error.
        """
        if self.target is None or self.repo is None or self.branch_name is None:
            return Err(EvolutionError("Session not properly initialized"))

        optimizer_prompt = build_optimizer_prompt(self.target)

        # Determine provider from model
        ptype = infer_provider_type(self.model)
        provider = get_provider(self.config, model_id=self.model, provider_type=ptype)

        async def fetch_response(prepared: PreparedPrompt) -> str:
            messages = [Message(role="user", content=prepared.prompt)]
            options = CompletionOptions(
                temperature=prepared.temperature,
                reasoning_effort=prepared.reasoning_effort,
                max_tokens=8192,
            )
            response = await provider.complete(messages, model=self.model, options=options)
            return response.content

        def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
            return fetch_response(prepared)

        # Use py_compile as the validation command
        validation_cmd = f"python -m py_compile {self.target.path}"

        success = await run_repair_loop(
            base_prompt=optimizer_prompt,
            command=validation_cmd,
            model=self.model,
            attach_recent=False,
            include_diff=True,
            fetch_response=fetcher,
            app_config=self.config,
            workspace_root=self.root,
            auto_archive=True,
            max_retries=3,
            keep_failed=False,
        )

        if not success:
            await cleanup_branch(self.repo, self.branch_name)
            return Ok(False)

        return Ok(True)

    async def _verify_and_pr(
        self,
    ) -> Result[tuple[VerificationResult, str | None], EvolutionError]:
        """Run verification and create PR if successful.

        Requires: self.target, self.repo, and self.branch_name must be set.

        Returns:
            Ok((verification, pr_url)) on success.
            Err(EvolutionError) on verification failure.
        """
        if self.target is None or self.repo is None or self.branch_name is None:
            return Err(EvolutionError("Session not properly initialized"))

        # Determine changed files for targeted testing
        match await self.repo.run_git("diff", "--name-only", "main..HEAD"):
            case Err(err):
                await cleanup_branch(self.repo, self.branch_name)
                return Err(EvolutionError(f"Failed to detect changed files: {err}"))
            case Ok(diff_output):
                changed_paths = [line.strip() for line in diff_output.splitlines() if line.strip()]

        verification_result = await run_verification(changed_paths, self.root, self.config)
        match verification_result:
            case Err(verif_err):
                await abort_evolution(
                    self.repo,
                    self.branch_name,
                    str(verif_err),
                    ["evolve", "failure", "tests"],
                    self.config,
                    reset_hard=True,
                )
                return Err(verif_err)
            case Ok(verification):
                if not verification.success:
                    await abort_evolution(
                        self.repo,
                        self.branch_name,
                        f"Verification failed (exit {verification.exit_code}).",
                        ["evolve", "failure", "tests"],
                        self.config,
                        reset_hard=True,
                    )
                    return Err(
                        EvolutionError(f"Verification failed (exit {verification.exit_code}).")
                    )

        # Create PR
        pr_result = await create_evolution_pr(
            self.repo,
            self.target,
            self.branch_name,
            self.root,
            self.config,
            verification,
        )

        return Ok((verification, pr_result.pr_url))

    async def run(self, dry_run: bool = False) -> Result[EvolutionResult | None, EvolutionError]:
        """Execute the full evolution workflow.

        Args:
            dry_run: If True, analyze without making changes.

        Returns:
            Ok(None) if no optimization needed (below threshold or no files).
            Ok(EvolutionResult) on successful optimization.
            Err(EvolutionError) on failure.
        """
        # Step 1: Check git state
        match await self._check_git_state():
            case Err(err):
                return Err(err)
            case Ok(repo):
                self.repo = repo

        # Step 2: Analyze debt
        match await self._analyze_debt():
            case Err(err):
                return Err(err)
            case Ok(None):
                return Ok(None)
            case Ok(target):
                self.target = target

        if dry_run:
            return Ok(None)

        # Step 3: Create branch
        match await self._create_branch():
            case Err(err):
                return Err(err)
            case Ok(branch_name):
                self.branch_name = branch_name

        # Step 4: Run optimizer
        match await self._run_optimizer():
            case Err(err):
                return Err(err)
            case Ok(False):
                return Err(EvolutionError("Optimization failed. Returning to main branch."))
            case Ok(True):
                pass

        # Step 5: Verify and create PR
        match await self._verify_and_pr():
            case Err(err):
                return Err(err)
            case Ok((_, pr_url)):
                pass

        # At this point target and branch_name are guaranteed to be set
        assert self.target is not None
        assert self.branch_name is not None

        return Ok(
            EvolutionResult(
                target=self.target,
                branch_name=self.branch_name,
                pr_url=pr_url,
            )
        )


async def cleanup_branch(repo: git_core.AsyncRepo, branch_name: str) -> None:
    """Clean up a failed evolution branch by returning to main.

    Args:
        repo: Git repository handle.
        branch_name: Name of the branch to clean up.
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
        repo: Git repository handle.
        branch_name: Name of the evolution branch.
        message: Error message to log.
        memory_tags: Tags for memory persistence.
        config: Application configuration.
        reset_hard: Whether to reset hard before cleanup.
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

    This is a thin wrapper around EvolutionSession for backwards compatibility.

    Args:
        config: Application configuration.
        dry_run: If True, analyze without making changes.
        model: Optional model override for optimization.
        threshold: Minimum debt score to trigger optimization.

    Returns:
        Ok(None) if no optimization needed.
        Ok(EvolutionResult) on successful optimization.
        Err(EvolutionError) on failure.
    """
    session = EvolutionSession(config, model=model, threshold=threshold)
    return await session.run(dry_run=dry_run)


def get_debt_scores(
    scores: list[TechnicalDebtScore],
) -> list[tuple[str, float, int, float]]:
    """Get debt score summary for display.

    Args:
        scores: List of technical debt scores.

    Returns:
        List of (filename, complexity, fix_freq, debt_score) tuples.
    """
    return [
        (score.path.name, score.complexity_score, score.fix_frequency, score.debt_score)
        for score in scores
    ]


__all__ = [
    "EvolutionSession",
    "abort_evolution",
    "cleanup_branch",
    "get_debt_scores",
    "run_evolution",
]
