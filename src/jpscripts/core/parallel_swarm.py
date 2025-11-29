"""Parallel swarm controller with git worktree isolation.

This module provides the ParallelSwarmController for executing DAG-based
tasks in parallel using isolated git worktrees. Each parallel task runs
in its own worktree to prevent filesystem conflicts and git index.lock
contention.

Key classes:
- WorktreeManager: Manages lifecycle of git worktrees for task isolation
- TaskResult: Result of executing a single task
- ParallelSwarmController: Orchestrates parallel task execution

[invariant:typing] All types are explicit; mypy --strict compliant.
[invariant:async-io] All I/O operations use async patterns.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from jpscripts.core.config import AppConfig
from jpscripts.core.dag import DAGGraph, DAGTask, TaskStatus, WorktreeContext
from jpscripts.core.git import AsyncRepo
from jpscripts.core.result import (
    Err,
    GitError,
    Ok,
    Result,
    ValidationError,
    WorkspaceError,
)


@dataclass
class TaskResult:
    """Result of executing a single DAG task.

    Attributes:
        task_id: ID of the completed task
        status: Final task status
        branch_name: Git branch created for this task
        commit_sha: Final commit SHA if successful
        error_message: Error details if failed
        artifacts: Any artifacts produced by the task
    """

    task_id: str
    status: TaskStatus
    branch_name: str
    commit_sha: str | None = None
    error_message: str | None = None
    artifacts: list[str] = field(default_factory=list)


class WorktreeManager:
    """Manages git worktrees for parallel task isolation.

    Each parallel task runs in its own worktree to prevent:
    - Git index.lock contention
    - Filesystem race conditions
    - Merge conflicts during parallel execution

    Attributes:
        repo: The main repository
        worktree_root: Directory where worktrees are created
        preserve_on_failure: Keep failed worktrees for debugging

    [invariant:async-io] All operations use async subprocess
    """

    def __init__(
        self,
        repo: AsyncRepo,
        worktree_root: Path | None = None,
        preserve_on_failure: bool = False,
    ) -> None:
        """Initialize the worktree manager.

        Args:
            repo: The main git repository
            worktree_root: Directory for worktrees (default: temp dir)
            preserve_on_failure: Keep worktrees on failure for debugging
        """
        self._repo = repo
        self._worktree_root = worktree_root or Path(tempfile.gettempdir()) / "jp-worktrees"
        self._preserve_on_failure = preserve_on_failure
        self._active_worktrees: dict[str, WorktreeContext] = {}
        self._initialized = False

    @property
    def worktree_root(self) -> Path:
        """Get the worktree root directory."""
        return self._worktree_root

    async def initialize(self) -> Result[None, GitError]:
        """Initialize the worktree manager.

        Creates the worktree root directory if it doesn't exist.

        [invariant:async-io] Uses asyncio.to_thread for mkdir
        """
        if self._initialized:
            return Ok(None)

        def _create_root() -> None:
            self._worktree_root.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(_create_root)
            self._initialized = True
            return Ok(None)
        except OSError as exc:
            return Err(GitError(f"Failed to create worktree root: {exc}"))

    async def _create_worktree_context(self, task_id: str) -> WorktreeContext:
        """Create a new worktree for a task.

        Args:
            task_id: Unique task identifier

        Returns:
            WorktreeContext with paths and branch info
        """
        # Generate unique branch name
        unique_suffix = uuid.uuid4().hex[:8]
        branch_name = f"swarm/{task_id}-{unique_suffix}"
        worktree_path = self._worktree_root / f"worktree-{task_id}-{unique_suffix}"

        # Create the worktree
        result = await self._repo.worktree_add(
            worktree_path,
            branch_name,
            new_branch=True,
        )

        if isinstance(result, Err):
            raise RuntimeError(f"Failed to create worktree: {result.error}")

        ctx = WorktreeContext(
            task_id=task_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
            status=TaskStatus.RUNNING,
        )

        self._active_worktrees[task_id] = ctx
        return ctx

    async def cleanup_worktree(
        self,
        ctx: WorktreeContext,
        *,
        failed: bool = False,
    ) -> Result[None, GitError]:
        """Clean up a worktree after task completion.

        Args:
            ctx: The worktree context to clean up
            failed: Whether the task failed

        Returns:
            Ok(None) on success, Err on failure
        """
        # Preserve on failure if configured
        if failed and self._preserve_on_failure:
            return Ok(None)

        # Remove the worktree
        result = await self._repo.worktree_remove(ctx.worktree_path, force=True)

        # Remove from active tracking
        if ctx.task_id in self._active_worktrees:
            del self._active_worktrees[ctx.task_id]

        if isinstance(result, Err):
            # Try pruning as fallback
            await self._repo.worktree_prune()

        return result

    @asynccontextmanager
    async def create_worktree(self, task_id: str) -> AsyncIterator[WorktreeContext]:
        """Create a worktree with automatic cleanup.

        This is the primary interface for creating worktrees.
        Uses context manager pattern to ensure cleanup.

        Args:
            task_id: Unique task identifier

        Yields:
            WorktreeContext for the created worktree

        Example:
            async with manager.create_worktree("task-001") as ctx:
                # Execute task in ctx.worktree_path
                pass
            # Worktree is automatically cleaned up
        """
        ctx = await self._create_worktree_context(task_id)
        failed = False

        try:
            yield ctx
        except Exception:
            failed = True
            raise
        finally:
            await self.cleanup_worktree(ctx, failed=failed)

    async def cleanup_all(self, force: bool = False) -> None:
        """Clean up all active worktrees.

        Args:
            force: Force cleanup even if dirty

        [invariant:async-io] Uses async worktree removal
        """
        for task_id, ctx in list(self._active_worktrees.items()):
            await self._repo.worktree_remove(ctx.worktree_path, force=force)
            del self._active_worktrees[task_id]

        # Final prune to clean up any orphaned references
        await self._repo.worktree_prune()


class MergeResult(BaseModel):
    """Result of merging parallel branches.

    Attributes:
        success: Whether all merges succeeded
        merged_branches: List of successfully merged branches
        conflict_branches: Branches with conflicts requiring resolution
        final_commit: Final merge commit SHA if successful
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    merged_branches: list[str] = Field(default_factory=list)
    conflict_branches: list[str] = Field(default_factory=list)
    final_commit: str | None = None


class ParallelSwarmController:
    """Orchestrates parallel swarm execution with worktree isolation.

    This controller manages the execution of DAG-based tasks in parallel,
    using git worktrees to isolate each task's file changes.

    Workflow:
    1. Receive DAG from Architect
    2. Validate DAG (no cycles, valid dependencies)
    3. Execute tasks in topological order with max parallelism
    4. Merge branches back to main
    5. Handle conflicts via MergeConflictResolver

    Attributes:
        objective: High-level goal for the swarm
        config: Application configuration
        repo_root: Path to the git repository
        max_parallel: Maximum concurrent tasks

    [invariant:typing] All types explicit; mypy --strict compliant
    [invariant:async-io] All I/O uses async patterns
    """

    def __init__(
        self,
        objective: str,
        config: AppConfig,
        repo_root: Path,
        model: str | None = None,
        *,
        max_parallel: int = 4,
        preserve_on_failure: bool = False,
    ) -> None:
        """Initialize the parallel swarm controller.

        Args:
            objective: High-level goal for the swarm
            config: Application configuration
            repo_root: Path to the git repository
            model: LLM model to use
            max_parallel: Maximum concurrent tasks
            preserve_on_failure: Keep failed worktrees for debugging
        """
        self.objective = objective.strip()
        self.config = config
        self.repo_root = repo_root.expanduser()
        self.model = model or config.default_model
        self.max_parallel = max_parallel
        self.preserve_on_failure = preserve_on_failure

        self._repo: AsyncRepo | None = None
        self._worktree_manager: WorktreeManager | None = None
        self._dag: DAGGraph | None = None
        self._completed_tasks: set[str] = set()
        self._task_results: dict[str, TaskResult] = {}

    async def _initialize(self) -> Result[None, GitError]:
        """Initialize the controller and worktree manager."""
        match await AsyncRepo.open(self.repo_root):
            case Ok(repo):
                self._repo = repo
            case Err(err):
                return Err(err)

        worktree_root = self.config.worktree_root
        if worktree_root:
            worktree_path = worktree_root.expanduser()
        else:
            worktree_path = None

        self._worktree_manager = WorktreeManager(
            repo=self._repo,
            worktree_root=worktree_path,
            preserve_on_failure=self.preserve_on_failure,
        )

        return await self._worktree_manager.initialize()

    def set_dag(self, dag: DAGGraph) -> Result[None, ValidationError]:
        """Set the DAG for execution.

        Validates the DAG is acyclic before accepting.

        Args:
            dag: The task dependency graph

        Returns:
            Ok(None) if valid, Err(ValidationError) if invalid
        """
        if not dag.validate_acyclic():
            return Err(ValidationError("DAG contains cycles"))

        self._dag = dag
        return Ok(None)

    async def _execute_task(
        self,
        task: DAGTask,
        ctx: WorktreeContext,
    ) -> TaskResult:
        """Execute a single task in a worktree.

        This is a placeholder implementation. The actual implementation
        would spawn an Engineer agent in the worktree.

        Args:
            task: The task to execute
            ctx: The worktree context

        Returns:
            TaskResult with execution outcome
        """
        # Placeholder: actual implementation would run an agent here
        # For now, just mark as completed
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            branch_name=ctx.branch_name,
            commit_sha=None,  # Would be set after agent commits
        )

    async def _run_parallel_batch(
        self,
        tasks: list[DAGTask],
    ) -> list[TaskResult]:
        """Execute a batch of tasks in parallel.

        Uses asyncio.TaskGroup for true parallelism (Python 3.11+).

        Args:
            tasks: Tasks to execute (should be disjoint)

        Returns:
            List of task results
        """
        if self._worktree_manager is None:
            return [
                TaskResult(
                    task_id=t.id,
                    status=TaskStatus.FAILED,
                    branch_name="",
                    error_message="Worktree manager not initialized",
                )
                for t in tasks
            ]

        results: list[TaskResult] = []

        async def _run_one(task: DAGTask) -> TaskResult:
            assert self._worktree_manager is not None
            try:
                async with self._worktree_manager.create_worktree(task.id) as ctx:
                    return await self._execute_task(task, ctx)
            except Exception as exc:
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    branch_name=f"swarm/{task.id}",
                    error_message=str(exc),
                )

        # Use TaskGroup for Python 3.11+ parallelism
        async with asyncio.TaskGroup() as tg:
            task_handles = [tg.create_task(_run_one(task)) for task in tasks]

        results = [handle.result() for handle in task_handles]

        # Update completed set
        for result in results:
            if result.status == TaskStatus.COMPLETED:
                self._completed_tasks.add(result.task_id)
            self._task_results[result.task_id] = result

        return results

    async def _merge_branches(
        self,
        results: list[TaskResult],
    ) -> MergeResult:
        """Merge completed task branches back to main.

        Args:
            results: Completed task results with branch names

        Returns:
            MergeResult indicating success/conflicts
        """
        if self._repo is None:
            return MergeResult(success=False, conflict_branches=[])

        merged: list[str] = []
        conflicts: list[str] = []

        for result in results:
            if result.status != TaskStatus.COMPLETED:
                continue

            if not result.branch_name:
                continue

            match await self._repo.merge(result.branch_name):
                case Ok(_):
                    merged.append(result.branch_name)
                case Err(_):
                    conflicts.append(result.branch_name)

        # Get final commit if all successful
        final_commit = None
        if not conflicts:
            match await self._repo.head(short=False):
                case Ok(sha):
                    final_commit = sha
                case Err(_):
                    pass

        return MergeResult(
            success=len(conflicts) == 0,
            merged_branches=merged,
            conflict_branches=conflicts,
            final_commit=final_commit,
        )

    async def run(self) -> Result[MergeResult, WorkspaceError]:
        """Execute the parallel swarm workflow.

        1. Initialize worktree manager
        2. Execute DAG in topological order
        3. Merge branches back

        Returns:
            Ok(MergeResult) on completion, Err(WorkspaceError) on failure
        """
        if self._dag is None:
            return Err(WorkspaceError("No DAG set for execution"))

        # Initialize
        init_result = await self._initialize()
        if isinstance(init_result, Err):
            return Err(
                WorkspaceError(
                    f"Initialization failed: {init_result.error}",
                    context={"error": str(init_result.error)},
                )
            )

        all_results: list[TaskResult] = []

        # Execute in waves based on dependencies
        while True:
            ready_tasks = self._dag.get_ready_tasks(self._completed_tasks)
            if not ready_tasks:
                break

            # Limit parallelism
            batch = ready_tasks[: self.max_parallel]
            batch_results = await self._run_parallel_batch(batch)
            all_results.extend(batch_results)

            # Check for failures
            failures = [r for r in batch_results if r.status == TaskStatus.FAILED]
            if failures:
                # Return partial result on failure
                failed_ids = [f.task_id for f in failures]
                return Err(
                    WorkspaceError(
                        f"Tasks failed: {failed_ids}",
                        context={"failed_tasks": failed_ids},
                    )
                )

        # Merge all branches
        merge_result = await self._merge_branches(all_results)

        # Cleanup
        if self._worktree_manager:
            await self._worktree_manager.cleanup_all(force=True)

        return Ok(merge_result)


__all__ = [
    "TaskResult",
    "WorktreeManager",
    "MergeResult",
    "ParallelSwarmController",
]
