"""Autonomous repair loop and execution logic.

This module provides the core repair loop functionality, including
command execution, patch application, and strategy management.

Decomposed modules:
- archive.py: Session archiving to memory store
- single_shot.py: Single-shot execution without repair loop
- turn_processing.py: Turn-level helpers (patch, tool, loop detection)
- patching.py: Patch parsing and application
- strategies.py: Repair strategy definitions
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from jpscripts.agent import ops
from jpscripts.agent.context import expand_context_paths
from jpscripts.agent.engine import AgentEngine
from jpscripts.agent.factory import build_default_middleware
from jpscripts.agent.models import (
    AgentEvent,
    AgentResponse,
    EventKind,
    Message,
    PatchFetcher,
    PreparedPrompt,
    RepairLoopConfig,
    ResponseFetcher,
    SecurityError,
    ToolCall,
)
from jpscripts.agent.ops import verify_syntax  # Re-export for backward compatibility
from jpscripts.agent.parsing import parse_agent_response
from jpscripts.agent.patching import apply_patch_text
from jpscripts.agent.prompting import prepare_agent_prompt
from jpscripts.agent.single_shot import (
    SingleShotConfig,
    SingleShotResult,
    SingleShotRunner,
)
from jpscripts.agent.strategies import (
    STRATEGY_OVERRIDE_TEXT,
    AttemptContext,
    RepairStrategy,
    StrategyConfig,
    build_repair_instruction,
    build_strategy_plan,
)
from jpscripts.agent.turn_processing import (
    LoopContext,
    append_history,
    handle_no_patch,
    handle_success_and_archive,
    process_patch,
    process_tool_call,
    setup_loop_context,
)
from jpscripts.core import security
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.mcp_registry import get_tool_registry
from jpscripts.core.result import Err, Ok

logger = get_logger(__name__)


class RepairLoopOrchestrator:
    """Orchestrates the autonomous repair loop with testable state management.

    This class encapsulates the repair loop logic, making it easier to test
    by exposing state as instance attributes and separating decision logic
    from execution.

    Attributes:
        changed_files: Set of paths modified during the repair loop.
        attempt_history: History of repair attempts for strategy selection.
        history: Conversation history for agent context.
        seen_patch_hashes: Set of patch hashes to detect duplicates.
    """

    def __init__(
        self,
        *,
        base_prompt: str,
        command: str,
        model: str | None,
        fetch_response: ResponseFetcher,
        config: RepairLoopConfig,
        app_config: AppConfig,
        workspace_root: Path,
    ) -> None:
        """Initialize the repair loop orchestrator.

        Args:
            base_prompt: The user's repair instruction.
            command: Shell command to verify fixes.
            model: LLM model ID to use.
            fetch_response: Async function to fetch LLM responses.
            config: Configuration for the repair loop.
            app_config: Application configuration (injected).
            workspace_root: Workspace root path (injected).
        """
        # Configuration (immutable)
        self.base_prompt = base_prompt
        self.command = command
        self.model = model
        self.fetch_response = fetch_response
        self.loop_config = config
        self._app_config = app_config
        self._workspace_root = workspace_root

        # State (mutable, exposed for testing)
        self.changed_files: set[Path] = set()
        self.attempt_history: list[AttemptContext] = []
        self.history: list[Message] = []
        self.seen_patch_hashes: set[str] = set()

        # Internal state (set during _setup)
        self._root: Path | None = None
        self._runtime_config: AppConfig | None = None
        self._strategies: list[StrategyConfig] = []
        self._attempt_cap: int = 0

    def _setup(self) -> None:
        """Initialize runtime state from injected configuration."""
        self._runtime_config = self._app_config
        workspace = self._workspace_root or self._app_config.user.notes_dir
        match security.validate_workspace_root(workspace):
            case Ok(root):
                self._root = root
            case Err(err):
                raise RuntimeError(f"Invalid workspace root: {err.message}")
        self._attempt_cap = max(1, self.loop_config.max_retries)
        self._strategies = build_strategy_plan(self._attempt_cap)

    async def _build_prompt(
        self,
        history_messages: Sequence[Message],
        iteration_prompt: str,
        loop_detected_flag: bool,
        temp_override: float | None,
        strategy: StrategyConfig,
        extra_paths: Sequence[Path],
    ) -> PreparedPrompt:
        """Build a prepared prompt for the agent."""
        config = self._runtime_config
        history_text = "\n".join(msg.content for msg in history_messages)
        instruction = iteration_prompt
        if history_text:
            instruction = f"{instruction}\n\nPrevious tool interactions:\n{history_text}"
        reasoning = "high" if loop_detected_flag or strategy.name == "step_back" else None
        run_cmd = self.command if strategy.name in {"fast", "deep"} else None
        return await prepare_agent_prompt(
            instruction,
            model=self.model,
            run_command=run_cmd,
            attach_recent=self.loop_config.attach_recent,
            include_diff=self.loop_config.include_diff,
            ignore_dirs=config.user.ignore_dirs,  # type: ignore[union-attr]
            max_file_context_chars=config.ai.max_file_context_chars,  # type: ignore[union-attr]
            max_command_output_chars=config.ai.max_command_output_chars,  # type: ignore[union-attr]
            reasoning_effort=reasoning,
            temperature=temp_override,
            tool_history=history_text,
            extra_paths=extra_paths,
            web_access=self.loop_config.web_access,
        )

    async def _fetch(self, prepared: PreparedPrompt) -> str:
        """Fetch a response from the LLM."""
        return await self.fetch_response(prepared)

    def _get_loop_context(self, current_error: str) -> LoopContext:
        """Set up context based on loop detection."""
        return setup_loop_context(self.attempt_history, current_error)

    async def _get_dynamic_paths(
        self,
        strategy_name: str,
        current_error: str,
    ) -> set[Path]:
        """Get dynamic context paths based on strategy."""
        assert self._root is not None
        config = self._runtime_config
        if strategy_name == "deep":
            return await expand_context_paths(
                current_error,
                self._root,
                self.changed_files,
                config.user.ignore_dirs,  # type: ignore[union-attr]
            )
        return set(self.changed_files)

    async def _run_turn_loop(
        self,
        strategy_cfg: StrategyConfig,
        current_error: str,
        loop_ctx: LoopContext,
        dynamic_paths: set[Path],
    ) -> AsyncIterator[AgentEvent]:
        """Execute the inner turn loop for agent interactions.

        Yields:
            AgentEvent objects as the turn progresses.
        """
        assert self._root is not None
        config = self._runtime_config

        for _turn in range(5):
            iteration_prompt = build_repair_instruction(
                self.base_prompt,
                current_error,
                self.attempt_history,
                self._root,
                strategy_override=loop_ctx.strategy_override,
                reasoning_hint=loop_ctx.reasoning_hint,
                strategy=strategy_cfg,
            )

            # Capture loop variables via default args for proper binding
            # tools={} disables tools (step_back), get_tool_registry() enables full tools
            # workspace_root enables governance checks for constitutional compliance
            async def prompt_builder(
                msgs: Sequence[Message],
                *,
                _prompt: str = iteration_prompt,
                _loop_detected: bool = loop_ctx.loop_detected,
                _temp: float | None = loop_ctx.temperature_override,
                _strategy: StrategyConfig = strategy_cfg,
                _paths: list[Path] = list(dynamic_paths),  # noqa: B006
            ) -> PreparedPrompt:
                return await self._build_prompt(
                    msgs,
                    _prompt,
                    _loop_detected,
                    _temp,
                    _strategy,
                    _paths,
                )

            # Build middleware using factory (dependency injection)
            middleware = build_default_middleware(
                persona="Engineer",
                workspace_root=self._root,
                governance_enabled=True,
                render_prompt=prompt_builder,
                fetch_response=self._fetch,
                parser=parse_agent_response,
            )

            engine = AgentEngine[AgentResponse](
                persona="Engineer",
                model=self.model or config.ai.default_model,  # type: ignore[union-attr]
                prompt_builder=prompt_builder,
                fetch_response=self._fetch,
                parser=parse_agent_response,
                tools={} if strategy_cfg.name == "step_back" else get_tool_registry(),
                template_root=self._root,
                workspace_root=self._root,
                middleware=middleware,
            )

            step_result = await engine.step(self.history)

            # Handle Result from engine.step()
            if isinstance(step_result, Err):
                error = step_result.error
                error_message = f"Agent step failed ({error.kind}): {error.message}"
                yield AgentEvent(
                    EventKind.VALIDATION_ERROR,
                    error_message,
                    {"error": error_message, "kind": error.kind},
                )
                append_history(
                    self.history,
                    Message(
                        role="system",
                        content=(
                            "<Turn>\nAgent thought: (invalid)\nTool output: "
                            f"{error_message}\n</Turn>"
                        ),
                    ),
                )
                current_error = error_message
                continue

            agent_response = step_result.value
            tool_call: ToolCall | None = agent_response.tool_call
            patch_text = (agent_response.file_patch or "").strip()
            thought = agent_response.thought_process

            if tool_call:
                events = await process_tool_call(engine, tool_call, thought, self.history)
                for event in events:
                    yield event
                continue

            if patch_text:
                result = await process_patch(
                    patch_text,
                    thought,
                    self._root,
                    self.seen_patch_hashes,
                    self.changed_files,
                    self.history,
                )
                for event in result.events:
                    yield event
                if result.error_message:
                    current_error = result.error_message
                if result.should_break:
                    break
                continue

            result = handle_no_patch(agent_response, thought, self.history)
            for event in result.events:
                yield event
            break

    async def _run_attempt(self, attempt: int) -> AsyncIterator[AgentEvent]:
        """Execute a single repair attempt.

        Args:
            attempt: The attempt number (0-indexed).

        Yields:
            AgentEvent objects as the attempt progresses.
            COMMAND_SUCCESS indicates the attempt succeeded.
        """
        assert self._root is not None
        config = self._runtime_config

        strategy_cfg = self._strategies[min(attempt, len(self._strategies) - 1)]
        yield AgentEvent(
            EventKind.ATTEMPT_START,
            f"Attempt {attempt + 1}/{self._attempt_cap} ({strategy_cfg.label})",
            {
                "attempt": attempt + 1,
                "max": self._attempt_cap,
                "strategy": strategy_cfg.label,
                "command": self.command,
            },
        )

        # Initial command run
        exit_code, stdout, stderr = await ops.run_agent_command(self.command, self._root)
        if exit_code == 0:
            yield AgentEvent(
                EventKind.COMMAND_SUCCESS,
                "Command succeeded. Exiting repair loop.",
                {"attempt": attempt + 1, "phase": "initial"},
            )
            return

        current_error = ops.summarize_output(
            stdout,
            stderr,
            config.ai.max_command_output_chars,  # type: ignore[union-attr]
        )
        yield AgentEvent(
            EventKind.COMMAND_FAILED,
            f"Attempt {attempt + 1} failed",
            {"attempt": attempt + 1, "error": current_error, "phase": "initial"},
        )

        # Set up loop context and dynamic paths
        loop_ctx = self._get_loop_context(current_error)
        if loop_ctx.event:
            yield loop_ctx.event
        dynamic_paths = await self._get_dynamic_paths(strategy_cfg.name, current_error)

        # Run the turn loop and yield all events
        async for event in self._run_turn_loop(
            strategy_cfg, current_error, loop_ctx, dynamic_paths
        ):
            yield event

        # Verify after turns
        exit_code, stdout, stderr = await ops.run_agent_command(self.command, self._root)
        if exit_code == 0:
            yield AgentEvent(
                EventKind.COMMAND_SUCCESS,
                "Command succeeded after applying fixes.",
                {"attempt": attempt + 1, "phase": "verification", "after_fixes": True},
            )
            return

        failure_msg = ops.summarize_output(
            stdout,
            stderr,
            config.ai.max_command_output_chars,  # type: ignore[union-attr]
        )
        yield AgentEvent(
            EventKind.COMMAND_FAILED,
            "Verification failed",
            {"attempt": attempt + 1, "error": failure_msg, "phase": "verification"},
        )

        # Record attempt history
        self.attempt_history.append(
            AttemptContext(
                iteration=attempt + 1,
                last_error=failure_msg,
                files_changed=list(self.changed_files),
                strategy=strategy_cfg.name,
            )
        )
        append_history(
            self.history,
            Message(
                role="system",
                content=f"Verification failure (attempt {attempt + 1}): {failure_msg}",
            ),
        )

    async def _verify_final(self) -> AsyncIterator[AgentEvent]:
        """Perform final verification after all attempts.

        Yields:
            AgentEvent objects. COMMAND_SUCCESS indicates success.
        """
        assert self._root is not None
        config = self._runtime_config

        yield AgentEvent(
            EventKind.COMMAND_FAILED,
            "Max retries reached. Verifying one last time...",
            {"phase": "final_verification_start"},
        )
        exit_code, stdout, stderr = await ops.run_agent_command(self.command, self._root)

        if exit_code == 0:
            yield AgentEvent(
                EventKind.COMMAND_SUCCESS,
                "Command succeeded after final verification.",
                {"phase": "final_verification"},
            )
            return

        error_msg = ops.summarize_output(
            stdout,
            stderr,
            config.ai.max_command_output_chars,  # type: ignore[union-attr]
        )
        yield AgentEvent(
            EventKind.COMMAND_FAILED,
            "Command still failing",
            {"error": error_msg, "phase": "final"},
        )

        if self.changed_files and not self.loop_config.keep_failed:
            yield AgentEvent(
                EventKind.REVERTING,
                "Reverting changes from failed attempts.",
                {"files": [str(p) for p in self.changed_files]},
            )
            await ops.revert_files(list(self.changed_files), self._root)

    async def run(self) -> AsyncIterator[AgentEvent]:
        """Execute the autonomous repair loop.

        Yields:
            AgentEvent objects as the repair progresses.
            The final event is COMPLETE with data["success"] indicating result.
        """
        self._setup()
        assert self._root is not None

        last_error: str | None = None

        for attempt in range(self._attempt_cap):
            success = False
            async for event in self._run_attempt(attempt):
                yield event
                # Track success from COMMAND_SUCCESS events
                if event.kind == EventKind.COMMAND_SUCCESS:
                    success = True
                # Track last error for archiving
                if event.kind == EventKind.COMMAND_FAILED:
                    last_error = event.data.get("error")

            if success:
                await handle_success_and_archive(
                    self.loop_config.auto_archive,
                    self.fetch_response,
                    self._app_config,
                    self.base_prompt,
                    self.command,
                    last_error
                    or (self.attempt_history[-1].last_error if self.attempt_history else None),
                    self.model,
                    self.loop_config.web_access,
                )
                yield AgentEvent(
                    EventKind.COMPLETE,
                    "Repair succeeded",
                    {"success": True},
                )
                return

        # Final verification
        final_success = False
        async for event in self._verify_final():
            yield event
            if event.kind == EventKind.COMMAND_SUCCESS:
                final_success = True

        if final_success:
            await handle_success_and_archive(
                self.loop_config.auto_archive,
                self.fetch_response,
                self._app_config,
                self.base_prompt,
                self.command,
                self.attempt_history[-1].last_error if self.attempt_history else None,
                self.model,
                self.loop_config.web_access,
            )
            yield AgentEvent(
                EventKind.COMPLETE,
                "Repair succeeded after final verification",
                {"success": True},
            )
            return

        yield AgentEvent(
            EventKind.COMPLETE,
            "Repair failed after exhausting all attempts",
            {"success": False},
        )


# SingleShotConfig, SingleShotResult, SingleShotRunner are now in single_shot.py
# Re-exported above for backward compatibility


async def run_repair_loop(
    *,
    base_prompt: str,
    command: str,
    model: str | None,
    attach_recent: bool,
    include_diff: bool,
    fetch_response: ResponseFetcher,
    app_config: AppConfig,
    workspace_root: Path,
    auto_archive: bool = True,
    max_retries: int = 3,
    keep_failed: bool = False,
    web_access: bool = False,
) -> bool:
    """Execute an autonomous repair loop (backward-compatible wrapper).

    This function wraps RepairLoopOrchestrator for backward compatibility
    with existing callers. Events are consumed internally without rendering.

    Args:
        base_prompt: The user's repair instruction.
        command: Shell command to verify fixes.
        model: LLM model ID to use.
        attach_recent: Attach recently modified files to context.
        include_diff: Include git diff in context.
        fetch_response: Async function to fetch LLM responses.
        app_config: Application configuration (injected).
        workspace_root: Workspace root path (injected).
        auto_archive: Archive successful fixes to memory.
        max_retries: Maximum repair attempts before giving up.
        keep_failed: Keep changes even if repair loop fails.
        web_access: Enable web search for context.

    Returns:
        True if the repair succeeded, False otherwise.
    """
    config = RepairLoopConfig(
        attach_recent=attach_recent,
        include_diff=include_diff,
        auto_archive=auto_archive,
        max_retries=max_retries,
        keep_failed=keep_failed,
        web_access=web_access,
    )
    orchestrator = RepairLoopOrchestrator(
        base_prompt=base_prompt,
        command=command,
        model=model,
        fetch_response=fetch_response,
        config=config,
        app_config=app_config,
        workspace_root=workspace_root,
    )
    # Consume events silently and extract final success status
    success = False
    async for event in orchestrator.run():
        if event.kind == EventKind.COMPLETE:
            success = event.data.get("success", False)
    return success


__all__ = [
    "STRATEGY_OVERRIDE_TEXT",
    "AgentEvent",
    "AttemptContext",
    "EventKind",
    "PatchFetcher",
    "RepairLoopConfig",
    "RepairLoopOrchestrator",
    "RepairStrategy",
    "ResponseFetcher",
    "SecurityError",
    "SingleShotConfig",
    "SingleShotResult",
    "SingleShotRunner",
    "StrategyConfig",
    "apply_patch_text",
    "run_repair_loop",
    "verify_syntax",
]
