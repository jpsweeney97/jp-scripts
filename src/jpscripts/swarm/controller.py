"""Parallel swarm controller for DAG-based task execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from jpscripts.agent.execution import (
    apply_patch_text,
    verify_syntax,
)
from jpscripts.agent.prompting import prepare_agent_prompt
from jpscripts.core.config import AppConfig
from jpscripts.core.dag import DAGGraph, DAGTask, TaskStatus, WorktreeContext
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
from jpscripts.engine import (
    Message,
    PreparedPrompt,
    ToolCall,
    parse_agent_response,
)
from jpscripts.git import AsyncRepo
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
        self.model = model or config.ai.default_model
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
                    logger.info("Task %s: Patch applied to %d files", task.id, len(applied_paths))
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
                    logger.warning("Task %s: git add failed: %s", task.id, add_result.error)

                commit_msg = f"[swarm] {task.id}: {task.objective[:50]}"
                commit_result = await worktree_repo.commit(commit_msg)
                if isinstance(commit_result, Ok):
                    commit_sha = commit_result.value
                    logger.info("Task %s: Created commit %s", task.id, commit_sha[:8])
                else:
                    logger.warning("Task %s: Commit failed: %s", task.id, commit_result.error)

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
            ignore_dirs=list(self.config.user.ignore_dirs),
            max_file_context_chars=self.config.ai.max_file_context_chars,
            max_command_output_chars=self.config.ai.max_command_output_chars,
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
            exit_code, stdout, stderr = (
                result.value.returncode,
                result.value.stdout,
                result.value.stderr,
            )
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
    "ParallelSwarmController",
]
