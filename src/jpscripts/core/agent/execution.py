"""Autonomous repair loop and execution logic.

This module provides the core repair loop functionality, including
command execution, patch application, and strategy management.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from rich import box
from rich.panel import Panel

from jpscripts.core import security
from jpscripts.core.agent.context import expand_context_paths
from jpscripts.core.agent.patching import apply_patch_text, compute_patch_hash
from jpscripts.core.agent.prompting import prepare_agent_prompt
from jpscripts.core.agent.strategies import (
    STRATEGY_OVERRIDE_TEXT,
    AttemptContext,
    RepairStrategy,
    StrategyConfig,
    build_repair_instruction,
    build_strategy_plan,
    detect_repeated_failure,
)
from jpscripts.core.console import console, get_logger
from jpscripts.core.engine import (
    AgentEngine,
    AgentResponse,
    Message,
    PreparedPrompt,
    ToolCall,
    parse_agent_response,
)
from jpscripts.core.memory import save_memory
from jpscripts.core.result import Err, Ok
from jpscripts.core.runtime import get_runtime
from jpscripts.core.system import run_safe_shell

logger = get_logger(__name__)

_ACTIVE_ROOT: Path | None = None


class SecurityError(RuntimeError):
    """Raised when a tool invocation is considered unsafe."""


PatchFetcher = Callable[[PreparedPrompt], Awaitable[str]]
ResponseFetcher = Callable[[PreparedPrompt], Awaitable[str]]


@dataclass
class RepairLoopConfig:
    """Configuration for the autonomous repair loop.

    Groups optional parameters for run_repair_loop to reduce
    function signature complexity.
    """

    attach_recent: bool = False
    """Attach recently modified files to context."""

    include_diff: bool = True
    """Include git diff in context."""

    auto_archive: bool = True
    """Archive successful fixes to memory."""

    max_retries: int = 3
    """Maximum repair attempts before giving up."""

    keep_failed: bool = False
    """Keep changes even if repair loop fails."""

    web_access: bool = False
    """Enable web search for context."""


def _summarize_output(stdout: str, stderr: str, limit: int) -> str:
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    if not combined:
        return "Command failed without output."
    if len(combined) <= limit:
        return combined
    return _summarize_stack_trace(combined, limit)


def _summarize_stack_trace(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    lines = text.splitlines()
    if len(text) <= limit:
        return text
    if len(lines) < 4:
        return text[:limit] + "... [truncated]"

    head_keep = max(3, min(12, len(lines) // 3))
    tail_keep = max(6, min(20, len(lines) // 2))
    head_lines = lines[:head_keep]
    tail_lines = lines[-tail_keep:]
    middle_lines = lines[head_keep:-tail_keep] if tail_keep < len(lines) - head_keep else []

    middle_summary = ""
    if middle_lines:
        mid_idx = len(middle_lines) // 2
        window = middle_lines[max(0, mid_idx - 3) : min(len(middle_lines), mid_idx + 4)]
        middle_summary = (
            "\n[... middle truncated ...]\n" + "\n".join(window) + "\n[... resumes ...]\n"
        )

    assembled = "\n".join(head_lines) + middle_summary + "\n".join(tail_lines)
    if len(assembled) > limit:
        head_budget = max(limit // 3, 1)
        tail_budget = max(limit - head_budget - 40, 1)
        trimmed_head = "\n".join(lines)[:head_budget]
        trimmed_tail = "\n".join(lines)[-tail_budget:]
        return f"{trimmed_head}\n[... truncated for length ...]\n{trimmed_tail}"

    return assembled


def _append_history(history: list[Message], entry: Message, keep: int = 3) -> None:
    history.append(entry)
    if len(history) > keep:
        del history[:-keep]


async def _run_command(command: str, root: Path) -> tuple[int, str, str]:
    """Execute a shell command via centralized security validation.

    This is a thin adapter around run_safe_shell that converts the Result
    type to the (exit_code, stdout, stderr) tuple expected by run_repair_loop.

    Args:
        command: The shell command to execute
        root: The working directory

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    result = await run_safe_shell(command, root, "agent.repair_loop")
    if isinstance(result, Ok):
        return (result.value.returncode, result.value.stdout, result.value.stderr)
    # Synthetic failure for blocked/invalid commands
    return (1, "", str(result.error))


@dataclass
class _TurnResult:
    """Result of processing a single agent turn."""

    should_break: bool = False
    """Whether to break out of the turn loop."""
    error_message: str | None = None
    """Error message if turn failed."""
    applied_paths: list[Path] | None = None
    """Paths modified by patch application."""


async def _handle_success_and_archive(
    auto_archive: bool,
    fetch_response: ResponseFetcher,
    base_prompt: str,
    command: str,
    last_error: str | None,
    model: str | None,
    web_access: bool,
) -> None:
    """Handle successful command completion with optional archiving."""
    if auto_archive:
        await _archive_session_summary(
            fetch_response,
            base_prompt=base_prompt,
            command=command,
            last_error=last_error,
            model=model,
            web_access=web_access,
        )


async def _process_tool_call(
    engine: AgentEngine[AgentResponse],
    tool_call: ToolCall,
    thought: str,
    history: list[Message],
) -> None:
    """Process a tool call from the agent."""
    tool_name = tool_call.tool
    tool_args = tool_call.arguments
    console.print(
        Panel(
            f"Agent invoking {tool_name} with args {tool_args}",
            title="Tool Call",
            box=box.SIMPLE,
        )
    )
    try:
        output = await engine.execute_tool(tool_call)
    except Exception as exc:
        output = f"Tool execution failed: {exc}"
    console.print(Panel(output, title="Tool Output", box=box.SIMPLE, style="cyan"))
    history_entry = (
        "<Turn>\n"
        f"Agent thought: {thought}\n"
        f"Tool call: {tool_name}({tool_args})\n"
        f"Tool output: {output}\n"
        "</Turn>"
    )
    _append_history(history, Message(role="system", content=history_entry))


async def _process_patch(
    patch_text: str,
    thought: str,
    root: Path,
    seen_patch_hashes: set[str],
    changed_files: set[Path],
    history: list[Message],
) -> _TurnResult:
    """Process a patch from the agent."""
    patch_hash = compute_patch_hash(patch_text)
    if patch_hash in seen_patch_hashes:
        console.print("[yellow]Duplicate patch detected - skipping.[/yellow]")
        _append_history(
            history,
            Message(
                role="user",
                content="<GovernanceViolation> You proposed a patch identical to a previous failed attempt. You are looping. Try a different approach. </GovernanceViolation>",
            ),
        )
        return _TurnResult(should_break=False)

    seen_patch_hashes.add(patch_hash)
    console.print("[green]Agent proposed a fix.[/green]")
    applied_paths = await apply_patch_text(patch_text, root)
    syntax_error = await verify_syntax(applied_paths)

    if syntax_error:
        console.print(f"[red]Syntax Check Failed (Self-Correction):[/red] {syntax_error}")
        _append_history(
            history,
            Message(
                role="system",
                content=(
                    "<Turn>\n"
                    f"Agent thought: {thought}\n"
                    "Tool call: none\n"
                    f"Tool output: Syntax check failed: {syntax_error}\n"
                    "</Turn>"
                ),
            ),
        )
        changed_files.update(applied_paths)
        return _TurnResult(should_break=False, error_message=syntax_error)

    changed_files.update(applied_paths)
    return _TurnResult(should_break=True, applied_paths=applied_paths)


def _handle_no_patch(
    agent_response: AgentResponse,
    thought: str,
    history: list[Message],
) -> _TurnResult:
    """Handle when agent returns no patch."""
    message = agent_response.final_message or "Agent returned no patch content."
    console.print(f"[yellow]{message}[/yellow]")
    _append_history(
        history,
        Message(
            role="system",
            content=(
                "<Turn>\n"
                f"Agent thought: {thought}\n"
                "Tool call: none\n"
                f"Tool output: {message}\n"
                "</Turn>"
            ),
        ),
    )
    return _TurnResult(should_break=True)


@dataclass
class _LoopContext:
    """Context for loop detection and strategy adjustment."""

    loop_detected: bool
    strategy_override: str | None
    reasoning_hint: str | None
    temperature_override: float | None


def _setup_loop_context(
    attempt_history: list[AttemptContext],
    current_error: str,
) -> _LoopContext:
    """Set up context based on loop detection."""
    loop_detected = detect_repeated_failure(attempt_history, current_error)
    if loop_detected:
        console.print(
            "[yellow]Repeated failure detected; applying strategy override and higher reasoning effort.[/yellow]"
        )
    return _LoopContext(
        loop_detected=loop_detected,
        strategy_override=STRATEGY_OVERRIDE_TEXT if loop_detected else None,
        reasoning_hint=(
            "Increase temperature or reasoning effort to escape repetition."
            if loop_detected
            else None
        ),
        temperature_override=0.7 if loop_detected else None,
    )


async def _get_dynamic_paths(
    strategy_name: str,
    current_error: str,
    root: Path,
    changed_files: set[Path],
    ignore_dirs: list[str],
) -> set[Path]:
    """Get dynamic context paths based on strategy."""
    if strategy_name == "deep":
        return await expand_context_paths(current_error, root, changed_files, ignore_dirs)
    return set(changed_files)


@dataclass
class _TurnLoopResult:
    """Result of executing the agent turn loop."""

    applied_paths: list[Path]
    current_error: str


async def verify_syntax(files: list[Path]) -> str | None:
    """Verify Python syntax for changed files using py_compile.

    Args:
        files: List of file paths to verify

    Returns:
        Error message if syntax check fails, None if all files pass
    """
    py_files = [path for path in files if path.suffix == ".py"]
    if not py_files:
        return None

    for path in py_files:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "py_compile",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return "Python interpreter not found for syntax check."
        except Exception as exc:  # pragma: no cover - defensive
            return f"Syntax check failed for {path}: {exc}"

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = (
                stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            )
            return f"Syntax error in {path}: {message or 'py_compile failed'}"

    return None


async def _archive_session_summary(
    fetch_response: ResponseFetcher,
    *,
    base_prompt: str,
    command: str,
    last_error: str | None,
    model: str | None,
    web_access: bool = False,
) -> None:
    runtime = get_runtime()
    config = runtime.config
    summary_prompt = (
        "Summarize the error fixed and the solution applied in one sentence for a knowledge base.\n"
        f"Command: {command}\n"
        f"Task: {base_prompt}\n"
        f"Last error before success: {last_error or 'N/A'}"
    )
    prepared = PreparedPrompt(prompt=summary_prompt, attached_files=[])
    try:
        raw_summary = await fetch_response(prepared)
    except Exception as exc:
        logger.debug("Summary fetch failed: %s", exc)
        return

    if not raw_summary.strip():
        return

    summary_text = raw_summary.strip()
    try:
        parsed = parse_agent_response(summary_text)
        summary_text = parsed.final_message or parsed.thought_process or summary_text
    except ValidationError:
        pass

    try:
        archive_config = (
            config.model_copy(update={"use_semantic_search": False})
            if hasattr(config, "model_copy")
            else config
        )
        await asyncio.to_thread(
            save_memory, summary_text, ["auto-fix", "agent"], config=archive_config
        )
    except Exception as exc:
        logger.debug("Failed to archive repair summary: %s", exc)


async def _revert_changed_files(paths: Sequence[Path], root: Path) -> None:
    if not paths:
        return

    safe_paths: list[Path] = []
    for path in paths:
        result = await security.validate_path_safe_async(path, root)
        if isinstance(result, Err):
            logger.debug("Skipping revert for unsafe path %s: %s", path, result.error.message)
            continue
        safe_paths.append(result.value)

    if not safe_paths:
        return

    try:
        # Disable git hooks to prevent malicious hook execution during revert.
        # The -c flag must come before the subcommand.
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "checkout",
            "--",
            *[str(path) for path in safe_paths],
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return

    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug(
            "Failed to revert files after unsuccessful loop: %s", stderr.decode(errors="replace")
        )


async def run_repair_loop(
    *,
    base_prompt: str,
    command: str,
    model: str | None,
    attach_recent: bool,
    include_diff: bool,
    fetch_response: ResponseFetcher,
    auto_archive: bool = True,
    max_retries: int = 3,
    keep_failed: bool = False,
    web_access: bool = False,
) -> bool:
    """
    Execute an autonomous repair loop using AgentEngine for orchestration.
    """
    global _ACTIVE_ROOT
    runtime = get_runtime()
    config = runtime.config
    root = security.validate_workspace_root(runtime.workspace_root or config.notes_dir)
    attempt_cap = max(1, max_retries)
    strategies = build_strategy_plan(attempt_cap)
    changed_files: set[Path] = set()
    attempt_history: list[AttemptContext] = []
    history: list[Message] = []
    seen_patch_hashes: set[str] = set()
    previous_active_root = _ACTIVE_ROOT
    _ACTIVE_ROOT = root  # pyright: ignore[reportConstantRedefinition]

    async def _prompt_builder(
        history_messages: Sequence[Message],
        iteration_prompt: str,
        loop_detected_flag: bool,
        temp_override: float | None,
        strategy: StrategyConfig,
        extra_paths: Sequence[Path],
    ) -> PreparedPrompt:
        history_text = "\n".join(msg.content for msg in history_messages)
        instruction = iteration_prompt
        if history_text:
            instruction = f"{instruction}\n\nPrevious tool interactions:\n{history_text}"
        reasoning = "high" if loop_detected_flag or strategy.name == "step_back" else None
        run_cmd = command if strategy.name in {"fast", "deep"} else None
        return await prepare_agent_prompt(
            instruction,
            model=model,
            run_command=run_cmd,
            attach_recent=attach_recent,
            include_diff=include_diff,
            ignore_dirs=config.ignore_dirs,
            max_file_context_chars=config.max_file_context_chars,
            max_command_output_chars=config.max_command_output_chars,
            reasoning_effort=reasoning,
            temperature=temp_override,
            tool_history=history_text,
            extra_paths=extra_paths,
            web_access=web_access,
        )

    async def _fetch(prepared: PreparedPrompt) -> str:
        return await fetch_response(prepared)

    try:
        for attempt in range(attempt_cap):
            strategy_cfg = strategies[min(attempt, len(strategies) - 1)]
            console.print(
                f"[cyan]Attempt {attempt + 1}/{attempt_cap} ({strategy_cfg.label}): running `{command}`[/cyan]"
            )
            exit_code, stdout, stderr = await _run_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded. Exiting repair loop.[/green]")
                await _handle_success_and_archive(
                    auto_archive,
                    fetch_response,
                    base_prompt,
                    command,
                    attempt_history[-1].last_error if attempt_history else None,
                    model,
                    web_access,
                )
                return True

            current_error = _summarize_output(stdout, stderr, config.max_command_output_chars)
            console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {current_error}")

            loop_ctx = _setup_loop_context(attempt_history, current_error)
            dynamic_paths = await _get_dynamic_paths(
                strategy_cfg.name, current_error, root, changed_files, config.ignore_dirs
            )

            applied_paths: list[Path] = []
            for _turn in range(5):
                iteration_prompt = build_repair_instruction(
                    base_prompt,
                    current_error,
                    attempt_history,
                    root,
                    strategy_override=loop_ctx.strategy_override,
                    reasoning_hint=loop_ctx.reasoning_hint,
                    strategy=strategy_cfg,
                )

                # Lambda with default args from captured scope; mypy cannot infer types
                # tools=None uses unified registry from get_tool_registry()
                # workspace_root enables governance checks for constitutional compliance
                engine = AgentEngine[AgentResponse](
                    persona="Engineer",
                    model=model or config.default_model,
                    prompt_builder=lambda msgs,  # type: ignore[misc]
                    ip=iteration_prompt,
                    ld=loop_ctx.loop_detected,
                    temp=loop_ctx.temperature_override,
                    strat=strategy_cfg,
                    paths=list(dynamic_paths): _prompt_builder(msgs, ip, ld, temp, strat, paths),
                    fetch_response=_fetch,
                    parser=parse_agent_response,
                    tools={} if strategy_cfg.name == "step_back" else None,
                    template_root=root,
                    workspace_root=root,
                    governance_enabled=True,
                )

                try:
                    agent_response = await engine.step(history)
                except ValidationError as exc:
                    validation_error = f"Agent response validation failed: {exc}"
                    console.print(f"[red]{validation_error}[/red]")
                    _append_history(
                        history,
                        Message(
                            role="system",
                            content=(
                                "<Turn>\nAgent thought: (invalid)\nTool output: "
                                f"{validation_error}\n</Turn>"
                            ),
                        ),
                    )
                    current_error = validation_error
                    continue

                tool_call: ToolCall | None = agent_response.tool_call
                patch_text = (agent_response.file_patch or "").strip()
                thought = agent_response.thought_process

                if tool_call:
                    await _process_tool_call(engine, tool_call, thought, history)
                    continue

                if patch_text:
                    result = await _process_patch(
                        patch_text, thought, root, seen_patch_hashes, changed_files, history
                    )
                    if result.error_message:
                        current_error = result.error_message
                    if result.should_break:
                        if result.applied_paths:
                            applied_paths = result.applied_paths
                        break
                    continue

                _handle_no_patch(agent_response, thought, history)
                break

            exit_code, stdout, stderr = await _run_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded after applying fixes.[/green]")
                await _handle_success_and_archive(
                    auto_archive,
                    fetch_response,
                    base_prompt,
                    command,
                    current_error,
                    model,
                    web_access,
                )
                return True

            failure_msg = _summarize_output(stdout, stderr, config.max_command_output_chars)
            console.print(f"[yellow]Verification failed:[/yellow] {failure_msg}")
            attempt_history.append(
                AttemptContext(
                    iteration=attempt + 1,
                    last_error=failure_msg,
                    files_changed=list(changed_files),
                    strategy=strategy_cfg.name,
                )
            )
            _append_history(
                history,
                Message(
                    role="system",
                    content=f"Verification failure (attempt {attempt + 1}): {failure_msg}",
                ),
            )

        console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
        exit_code, stdout, stderr = await _run_command(command, root)
        if exit_code == 0:
            console.print("[green]Command succeeded after final verification.[/green]")
            await _handle_success_and_archive(
                auto_archive,
                fetch_response,
                base_prompt,
                command,
                attempt_history[-1].last_error if attempt_history else None,
                model,
                web_access,
            )
            return True

        console.print(
            f"[red]Command still failing:[/red] {_summarize_output(stdout, stderr, config.max_command_output_chars)}"
        )
        if changed_files and not keep_failed:
            console.print("[yellow]Reverting changes from failed attempts.[/yellow]")
            await _revert_changed_files(list(changed_files), root)

        return False
    finally:
        _ACTIVE_ROOT = previous_active_root  # pyright: ignore[reportConstantRedefinition]


__all__ = [
    "STRATEGY_OVERRIDE_TEXT",
    "AttemptContext",
    "PatchFetcher",
    "RepairLoopConfig",
    "RepairStrategy",
    "ResponseFetcher",
    "SecurityError",
    "StrategyConfig",
    "apply_patch_text",
    "run_repair_loop",
    "verify_syntax",
]
