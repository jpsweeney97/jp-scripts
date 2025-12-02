"""Parallel swarm controller for DAG-based task execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from jpscripts.core.config import AppConfig
from jpscripts.structures.dag import DAGGraph, DAGTask, TaskStatus, WorktreeContext
from jpscripts.core.result import (
    Err,
    GitError,
    Ok,
    Result,
    ValidationError,
    WorkspaceError,
)
from jpscripts.agent import PreparedPrompt
from jpscripts.git import AsyncRepo
from jpscripts.swarm.agent_adapter import SwarmAgentExecutor, TaskExecutor
from jpscripts.swarm.types import MergeResult, TaskResult
from jpscripts.swarm.worktree import WorktreeManager


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
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]] | None = None,
        max_turns_per_task: int = 5,
        task_executor: TaskExecutor | None = None,
    ) -> None:
        """Initialize the parallel swarm controller.

        Args:
            objective: High-level goal for the swarm
            config: Application configuration
            repo_root: Path to the git repository
            model: LLM model to use
            max_parallel: Maximum concurrent tasks
            preserve_on_failure: Keep failed worktrees for debugging
            fetch_response: Callback to fetch LLM responses
            max_turns_per_task: Maximum agent turns per task
            task_executor: Optional custom task executor (overrides fetch_response)
        """
        self.objective = objective.strip()
        self.config = config
        self.repo_root = repo_root.expanduser()
        self.model = model or config.ai.default_model
        self.max_parallel = max_parallel
        self.preserve_on_failure = preserve_on_failure

        # Create task executor: prefer explicit executor, then build from fetch_response
        if task_executor is not None:
            self._task_executor: TaskExecutor | None = task_executor
        elif fetch_response is not None:
            self._task_executor = SwarmAgentExecutor(
                config=config,
                model=self.model,
                fetch_response=fetch_response,
                max_turns=max_turns_per_task,
            )
        else:
            self._task_executor = None

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

        worktree_root = self.config.infra.worktree_root
        worktree_path = worktree_root.expanduser() if worktree_root else None

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
        """Execute a single task in a worktree using the task executor.

        Delegates actual execution to the configured TaskExecutor.

        Args:
            task: The task to execute
            ctx: The worktree context (provides worktree_path, branch_name)

        Returns:
            TaskResult with execution outcome

        [invariant:async-io] All I/O operations use async patterns.
        """
        # Validate we have a task executor
        if self._task_executor is None:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                branch_name=ctx.branch_name,
                error_message="No task executor configured",
            )

        # Delegate to the task executor
        return await self._task_executor.execute(task, ctx)

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
    "ParallelSwarmController",
]
