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
import logging
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from jpscripts.core.agent.execution import (
    apply_patch_text,
    verify_syntax,
)
from jpscripts.core.agent.prompting import prepare_agent_prompt
from jpscripts.core.config import AppConfig
from jpscripts.core.dag import DAGGraph, DAGTask, TaskStatus, WorktreeContext
from jpscripts.core.engine import (
    Message,
    PreparedPrompt,
    ToolCall,
    parse_agent_response,
)
from jpscripts.core.mcp_registry import get_tool_registry
from jpscripts.core.result import (
    Err,
    GitError,
    Ok,
    Result,
    ValidationError,
    WorkspaceError,
)
from jpscripts.core.system import run_safe_shell
from jpscripts.git import AsyncRepo


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

        Creates the worktree root directory if it doesn't exist and prunes
        any orphaned worktrees from previous crashed sessions.

        [invariant:async-io] Uses asyncio.to_thread for mkdir
        """
        if self._initialized:
            return Ok(None)

        def _create_root() -> None:
            self._worktree_root.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(_create_root)
        except OSError as exc:
            return Err(GitError(f"Failed to create worktree root: {exc}"))

        # Auto-detect and clean orphans from previous sessions
        removed = await self.prune_orphaned_worktrees()
        if removed > 0:
            logger = logging.getLogger(__name__)
            logger.info("Pruned %d orphaned worktrees", removed)

        self._initialized = True
        return Ok(None)

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

    async def detect_orphaned_worktrees(self) -> list[Path]:
        """Detect worktree directories from previous sessions not in memory.

        Scans worktree_root for directories matching the `worktree-*-*` pattern
        that are not currently tracked in _active_worktrees.

        Returns:
            List of orphaned worktree paths

        [invariant:async-io] Uses asyncio.to_thread for directory scan
        """
        if not self._worktree_root.exists():
            return []

        # Pattern: worktree-{task_id}-{8-char-hex}
        pattern = re.compile(r"^worktree-[\w-]+-[a-f0-9]{8}$")

        def _scan() -> list[Path]:
            return [
                d for d in self._worktree_root.iterdir() if d.is_dir() and pattern.match(d.name)
            ]

        candidates = await asyncio.to_thread(_scan)
        active_paths = {ctx.worktree_path for ctx in self._active_worktrees.values()}

        orphans: list[Path] = []
        for path in candidates:
            if path not in active_paths:
                orphans.append(path)

        return orphans

    async def prune_orphaned_worktrees(self, force: bool = True) -> int:
        """Remove orphaned worktrees from previous crashed sessions.

        Args:
            force: If True, use --force to remove even if dirty

        Returns:
            Number of worktrees successfully removed

        [invariant:async-io] Uses async worktree removal with fallback
        """
        orphans = await self.detect_orphaned_worktrees()
        if not orphans:
            return 0

        logger = logging.getLogger(__name__)
        logger.warning(
            "Found %d orphaned worktrees from previous session: %s",
            len(orphans),
            [p.name for p in orphans],
        )

        removed = 0
        for path in orphans:
            result = await self._repo.worktree_remove(path, force=force)
            if isinstance(result, Ok):
                removed += 1
            else:
                # Try manual cleanup if git command fails (orphan may not be registered)
                try:
                    await asyncio.to_thread(shutil.rmtree, path)
                    removed += 1
                except OSError as exc:
                    logger.error("Failed to remove orphan %s: %s", path, exc)

        # Final prune to clean git refs
        await self._repo.worktree_prune()

        return removed


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
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]] | None = None,
        max_turns_per_task: int = 5,
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
        """
        self.objective = objective.strip()
        self.config = config
        self.repo_root = repo_root.expanduser()
        self.model = model or config.default_model
        self.max_parallel = max_parallel
        self.preserve_on_failure = preserve_on_failure
        self._fetch_response = fetch_response
        self._max_turns_per_task = max_turns_per_task

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
        """Execute a single task in a worktree using an AI agent.

        This runs a bounded multi-turn agent loop that:
        1. Prepares context from the worktree
        2. Sends task objective to the agent
        3. Processes tool calls or patches
        4. Commits successful changes
        5. Returns result with commit SHA

        Args:
            task: The task to execute
            ctx: The worktree context (provides worktree_path, branch_name)

        Returns:
            TaskResult with execution outcome

        [invariant:async-io] All I/O operations use async patterns.
        """
        logger = logging.getLogger(__name__)

        # Validate we have a fetch_response callback
        if self._fetch_response is None:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                branch_name=ctx.branch_name,
                error_message="No fetch_response callback configured",
            )

        # Open repo in the worktree for git operations
        worktree_repo_result = await AsyncRepo.open(ctx.worktree_path)
        if isinstance(worktree_repo_result, Err):
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                branch_name=ctx.branch_name,
                error_message=f"Failed to open worktree repo: {worktree_repo_result.error}",
            )
        worktree_repo = worktree_repo_result.value

        # Build the task instruction
        instruction = self._build_task_instruction(task)

        # Agent loop state
        history: list[Message] = []
        changed_files: set[Path] = set()
        last_error: str | None = None

        try:
            for _turn in range(self._max_turns_per_task):
                # Build prompt with worktree context
                prepared = await self._prepare_prompt_for_task(
                    task=task,
                    instruction=instruction,
                    history=history,
                    worktree_path=ctx.worktree_path,
                    last_error=last_error,
                )

                # Fetch agent response
                raw_response = await self._fetch_response(prepared)

                # Parse response
                try:
                    agent_response = parse_agent_response(raw_response)
                except PydanticValidationError as exc:
                    last_error = f"Failed to parse agent response: {exc}"
                    history.append(
                        Message(
                            role="system",
                            content=f"<Error>Parse failed: {exc}</Error>",
                        )
                    )
                    continue

                # Handle tool call
                if agent_response.tool_call:
                    tool_output = await self._execute_tool_in_worktree(
                        agent_response.tool_call,
                        ctx.worktree_path,
                    )
                    history.append(
                        Message(
                            role="system",
                            content=(
                                f"<Turn>\n"
                                f"Tool: {agent_response.tool_call.tool}\n"
                                f"Args: {agent_response.tool_call.arguments}\n"
                                f"Output: {tool_output}\n"
                                f"</Turn>"
                            ),
                        )
                    )
                    continue

                # Handle file patch
                if agent_response.file_patch:
                    patch_text = agent_response.file_patch.strip()
                    applied_paths = await apply_patch_text(patch_text, ctx.worktree_path)

                    if not applied_paths:
                        last_error = "Patch application failed - check diff format"
                        history.append(
                            Message(
                                role="system",
                                content="<Error>Patch failed to apply</Error>",
                            )
                        )
                        continue

                    # Verify syntax
                    syntax_error = await verify_syntax(applied_paths)
                    if syntax_error:
                        last_error = syntax_error
                        history.append(
                            Message(
                                role="system",
                                content=f"<Error>Syntax error: {syntax_error}</Error>",
                            )
                        )
                        changed_files.update(applied_paths)
                        continue

                    changed_files.update(applied_paths)

                    # Patch applied successfully, break the loop
                    logger.info(
                        "Task %s: Patch applied to %d files", task.id, len(applied_paths)
                    )
                    break

                # Final message without action - task considered complete
                if agent_response.final_message:
                    logger.info("Task %s: Agent returned final message", task.id)
                    break

            # Commit changes if any files were modified
            commit_sha: str | None = None
            if changed_files:
                add_result = await worktree_repo.add(all=True)
                if isinstance(add_result, Err):
                    logger.warning(
                        "Task %s: git add failed: %s", task.id, add_result.error
                    )

                commit_msg = f"[swarm] {task.id}: {task.objective[:50]}"
                commit_result = await worktree_repo.commit(commit_msg)
                if isinstance(commit_result, Ok):
                    commit_sha = commit_result.value
                    logger.info("Task %s: Created commit %s", task.id, commit_sha[:8])
                else:
                    logger.warning(
                        "Task %s: Commit failed: %s", task.id, commit_result.error
                    )

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                branch_name=ctx.branch_name,
                commit_sha=commit_sha,
                artifacts=[str(f.relative_to(ctx.worktree_path)) for f in changed_files],
            )

        except Exception as exc:
            logger.exception("Task %s failed with exception", task.id)
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                branch_name=ctx.branch_name,
                error_message=str(exc),
            )

    def _build_task_instruction(self, task: DAGTask) -> str:
        """Build the instruction prompt for a task.

        Args:
            task: The task to build instruction for

        Returns:
            Instruction string for the agent
        """
        files_section = ""
        if task.files_touched:
            files_section = "\n\nFiles you may need to modify:\n" + "\n".join(
                f"- {f}" for f in task.files_touched
            )

        persona_hint = "engineer" if task.persona == "engineer" else "QA tester"

        return (
            f"Task ID: {task.id}\n"
            f"Objective: {task.objective}\n"
            f"Role: You are acting as a {persona_hint}.{files_section}\n\n"
            f"Complete this task by proposing file patches. "
            f"Respond with a valid AgentResponse JSON object."
        )

    async def _prepare_prompt_for_task(
        self,
        task: DAGTask,
        instruction: str,
        history: list[Message],
        worktree_path: Path,
        last_error: str | None,
    ) -> PreparedPrompt:
        """Prepare the prompt for a task execution turn.

        Args:
            task: The task being executed
            instruction: Base instruction text
            history: Previous turns in this task
            worktree_path: Path to the worktree
            last_error: Error from last turn (if any)

        Returns:
            PreparedPrompt ready for the LLM
        """
        # Build history text from recent turns
        history_text = "\n".join(msg.content for msg in history[-3:])

        # Augment instruction with history and error context
        full_instruction = instruction
        if history_text:
            full_instruction += f"\n\nPrevious turns:\n{history_text}"
        if last_error:
            full_instruction += f"\n\nLast error to fix:\n{last_error}"

        # Build extra paths from task's files_touched
        extra_paths: list[Path] = []
        for f in task.files_touched:
            path = worktree_path / f
            if path.exists():
                extra_paths.append(path)

        return await prepare_agent_prompt(
            full_instruction,
            model=self.model,
            run_command=None,
            attach_recent=False,
            include_diff=True,
            ignore_dirs=list(self.config.ignore_dirs),
            max_file_context_chars=self.config.max_file_context_chars,
            max_command_output_chars=self.config.max_command_output_chars,
            extra_paths=extra_paths,
            workspace_override=worktree_path,
        )

    async def _execute_tool_in_worktree(
        self,
        tool_call: ToolCall,
        worktree_path: Path,
    ) -> str:
        """Execute a tool call within the worktree context.

        Args:
            tool_call: The tool call from agent response
            worktree_path: Path to execute within

        Returns:
            Tool output string
        """
        tool_name = tool_call.tool.lower().strip()

        # Handle shell command tool specially - run in worktree
        if tool_name == "shell":
            command = str(tool_call.arguments.get("command", ""))
            if not command:
                return "Error: No command provided"
            result = await run_safe_shell(command, worktree_path, "swarm.task_executor")
            if isinstance(result, Err):
                return f"Command blocked: {result.error}"
            exit_code, stdout, stderr = result.value.returncode, result.value.stdout, result.value.stderr
            output = (stdout + stderr).strip()
            if exit_code != 0:
                return f"Command failed (exit {exit_code}):\n{output}"
            return output or "Command completed successfully"

        # For read_file tool, ensure path is within worktree
        if tool_name == "read_file":
            path_arg = tool_call.arguments.get("path", "")
            if path_arg:
                target = worktree_path / str(path_arg)
                if not target.is_relative_to(worktree_path):
                    return f"Error: Path {path_arg} is outside worktree"
                try:
                    return target.read_text(encoding="utf-8")
                except Exception as exc:
                    return f"Error reading file: {exc}"

        # For other tools, use the MCP registry
        registry = get_tool_registry()
        if tool_name not in registry:
            return f"Unknown tool: {tool_call.tool}"

        try:
            return await registry[tool_name](**tool_call.arguments)
        except Exception as exc:
            return f"Tool failed: {exc}"

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
    "MergeResult",
    "ParallelSwarmController",
    "TaskResult",
    "WorktreeManager",
]
