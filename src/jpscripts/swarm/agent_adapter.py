"""Agent adapter for swarm task execution.

This module provides the TaskExecutor protocol and SwarmAgentExecutor implementation
for decoupling agent prompting logic from the ParallelSwarmController.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError as PydanticValidationError

from jpscripts.agent import Message, PreparedPrompt, ToolCall, parse_agent_response
from jpscripts.agent.execution import apply_patch_text, verify_syntax
from jpscripts.agent.prompting import prepare_agent_prompt
from jpscripts.core.config import AppConfig
from jpscripts.core.mcp_registry import get_tool_registry
from jpscripts.core.result import Err, Ok
from jpscripts.git import AsyncRepo
from jpscripts.structures.dag import DAGTask, TaskStatus, WorktreeContext
from jpscripts.swarm.types import TaskResult
from jpscripts.system import run_safe_shell


class TaskExecutor(Protocol):
    """Protocol for task execution strategies.

    Implementations handle the actual execution of a DAGTask within a worktree,
    including LLM interaction, tool execution, and patch application.

    [invariant:typing] All types explicit; mypy --strict compliant.
    """

    async def execute(self, task: DAGTask, ctx: WorktreeContext) -> TaskResult:
        """Execute a task in the given worktree context.

        Args:
            task: The DAG task to execute
            ctx: Worktree context providing isolated workspace

        Returns:
            TaskResult with execution outcome
        """
        ...


class SwarmAgentExecutor:
    """Default task executor using LLM agent with multi-turn loop.

    This executor handles:
    - Building task instructions and prompts
    - Running multi-turn agent conversation
    - Executing tool calls within worktree
    - Applying file patches
    - Committing changes

    Attributes:
        config: Application configuration
        model: LLM model to use
        max_turns: Maximum agent turns per task

    [invariant:async-io] All I/O uses async patterns.
    """

    def __init__(
        self,
        config: AppConfig,
        model: str,
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
        max_turns: int = 5,
    ) -> None:
        """Initialize the swarm agent executor.

        Args:
            config: Application configuration
            model: LLM model to use
            fetch_response: Callback to fetch LLM responses
            max_turns: Maximum agent turns per task
        """
        self._config = config
        self._model = model
        self._fetch_response = fetch_response
        self._max_turns = max_turns

    def _build_instruction(self, task: DAGTask) -> str:
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

    async def _prepare_prompt(
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
            model=self._model,
            run_command=None,
            attach_recent=False,
            include_diff=True,
            ignore_dirs=list(self._config.user.ignore_dirs),
            max_file_context_chars=self._config.ai.max_file_context_chars,
            max_command_output_chars=self._config.ai.max_command_output_chars,
            extra_paths=extra_paths,
            workspace_override=worktree_path,
        )

    async def _execute_tool(
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

    async def execute(self, task: DAGTask, ctx: WorktreeContext) -> TaskResult:
        """Execute a task in the given worktree context.

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
        instruction = self._build_instruction(task)

        # Agent loop state
        history: list[Message] = []
        changed_files: set[Path] = set()
        last_error: str | None = None

        try:
            for _turn in range(self._max_turns):
                # Build prompt with worktree context
                prepared = await self._prepare_prompt(
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
                    tool_output = await self._execute_tool(
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


def create_agent_executor(
    config: AppConfig,
    model: str,
    fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
    max_turns: int = 5,
) -> SwarmAgentExecutor:
    """Factory for creating the default agent executor.

    Args:
        config: Application configuration
        model: LLM model to use
        fetch_response: Callback to fetch LLM responses
        max_turns: Maximum agent turns per task

    Returns:
        Configured SwarmAgentExecutor instance
    """
    return SwarmAgentExecutor(config, model, fetch_response, max_turns)


__all__ = [
    "SwarmAgentExecutor",
    "TaskExecutor",
    "create_agent_executor",
]
