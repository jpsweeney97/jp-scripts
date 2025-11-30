"""Autonomous repair loop and execution logic.

This module provides the core repair loop functionality, including
command execution, patch application, and strategy management.
"""

from __future__ import annotations

import asyncio
import hashlib
import shlex
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from rich import box
from rich.panel import Panel

from jpscripts.core import security
from jpscripts.core.agent.context import expand_context_paths
from jpscripts.core.agent.prompting import prepare_agent_prompt
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
from jpscripts.core.runtime import get_runtime

logger = get_logger(__name__)

STRATEGY_OVERRIDE_TEXT = (
    "You are stuck in a loop. Stop editing code. Analyze the error trace and the file content again. "
    "List three possible root causes before proposing a new patch."
)
_ACTIVE_ROOT: Path | None = None


class SecurityError(RuntimeError):
    """Raised when a tool invocation is considered unsafe."""


@dataclass
class AttemptContext:
    iteration: int
    last_error: str
    files_changed: list[Path]
    strategy: Literal["fast", "deep", "step_back"]


RepairStrategy = Literal["fast", "deep", "step_back"]


@dataclass(frozen=True)
class StrategyConfig:
    name: RepairStrategy
    label: str
    description: str
    system_notice: str = ""


PatchFetcher = Callable[[PreparedPrompt], Awaitable[str]]
ResponseFetcher = Callable[[PreparedPrompt], Awaitable[str]]


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


def _compute_patch_hash(patch_text: str) -> str:
    """Compute SHA256 hash of a patch for de-duplication."""
    return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


def _build_history_summary(history: Sequence[AttemptContext], root: Path) -> str:
    if not history:
        return "None yet."

    lines: list[str] = []
    for attempt in history:
        relative_files = []
        for path in attempt.files_changed:
            try:
                relative_files.append(str(path.relative_to(root)))
            except ValueError:
                relative_files.append(str(path))
        file_part = f" | files: {', '.join(relative_files)}" if relative_files else ""
        lines.append(f"Attempt {attempt.iteration}: {attempt.last_error}{file_part}")

    return "\n".join(lines)


def _detect_repeated_failure(history: Sequence[AttemptContext], current_error: str) -> bool:
    normalized_current = current_error.strip()
    if not normalized_current:
        return False
    occurrences = sum(1 for attempt in history if attempt.last_error.strip() == normalized_current)
    return occurrences + 1 >= 2


def _build_strategy_plan(attempt_cap: int) -> list[StrategyConfig]:
    base: list[StrategyConfig] = [
        StrategyConfig(
            name="fast",
            label="FAST - Immediate Context",
            description="Focus on the specific error line and immediate file context.",
        ),
        StrategyConfig(
            name="deep",
            label="DEEP - Cross-Module Analysis",
            description="Analyze imported dependencies and cross-module interactions. The error may be non-local.",
            system_notice="Context has been expanded to include imported dependencies and referenced modules.",
        ),
        StrategyConfig(
            name="step_back",
            label="STEP_BACK - Root Cause Analysis",
            description="Disregard previous assumptions. Formulate a Root Cause Analysis before writing code.",
            system_notice="Tool use is disabled for this turn. Perform Root Cause Analysis and propose a brief plan before patching.",
        ),
    ]

    if attempt_cap <= len(base):
        return base[:attempt_cap]

    tail_fill = [base[-1]] * (attempt_cap - len(base))
    return base + tail_fill


def _build_repair_instruction(
    base_prompt: str,
    current_error: str,
    history: Sequence[AttemptContext],
    root: Path,
    *,
    strategy_override: str | None = None,
    reasoning_hint: str | None = None,
    strategy: StrategyConfig,
) -> str:
    history_block = _build_history_summary(history, root)
    override_block = f"\n\nStrategy Override:\n{strategy_override}" if strategy_override else ""
    reasoning_block = (
        f"\n\nHigh reasoning effort requested: {reasoning_hint}" if reasoning_hint else ""
    )
    strategy_block = f"\n\n[Current Strategy: {strategy.label}]\n{strategy.description}"
    if strategy.system_notice:
        strategy_block += f"\n{strategy.system_notice}"
    return (
        f"{strategy_block}\n\n"
        f"{base_prompt.strip()}\n\n"
        "Autonomous repair loop in progress. Use the failure details to craft a minimal fix.\n"
        f"Current error:\n{current_error.strip()}\n\n"
        f"Previous attempts:\n{history_block}{override_block}{reasoning_block}\n\n"
        "Respond with a single JSON object that matches the AgentResponse schema. "
        "Place the unified diff in `file_patch`. Do not return Markdown or prose."
    )


async def _run_shell_command(command: str, cwd: Path) -> tuple[int, str, str]:
    """Executes command without shell interpolation."""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        logger.warning("Failed to parse shell command: %s", exc)
        return 1, "", f"Unable to parse command; simplify quoting. ({exc})"

    if not tokens:
        return 1, "", "Invalid command."

    try:
        logger.debug("Running safe command: %s", tokens)
        proc = await asyncio.create_subprocess_exec(
            *tokens,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return 1, "", str(exc)

    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _extract_patch_paths(patch_text: str, root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for raw_line in patch_text.splitlines():
        if not raw_line.startswith(("+++ ", "--- ")):
            continue
        try:
            _, path_str = raw_line.split(" ", 1)
        except ValueError:
            continue
        path_str = path_str.strip()
        if path_str in {"/dev/null", "dev/null", "a/dev/null", "b/dev/null"}:
            continue
        if path_str.startswith(("a/", "b/")):
            path_str = path_str[2:]
        try:
            candidates.add(security.validate_path(root / path_str, root))
        except PermissionError as exc:
            logger.debug("Skipped unsafe patch path %s: %s", path_str, exc)
        except Exception as exc:
            logger.debug("Failed to normalize patch path %s: %s", path_str, exc)
    return sorted(candidates)


def _write_failed_patch(patch_text: str, root: Path) -> None:
    try:
        destination = security.validate_path(root / "agent_failed_patch.diff", root)
        destination.write_text(patch_text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to persist failed patch for inspection: %s", exc)


async def _apply_patch_text(patch_text: str, root: Path) -> list[Path]:
    if not patch_text.strip():
        return []

    target_paths = _extract_patch_paths(patch_text, root)

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--whitespace=nowarn",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        proc = None

    if proc:
        stdout, stderr = await proc.communicate(patch_text.encode())
        if proc.returncode == 0:
            return target_paths
        logger.debug("git apply failed: %s", stderr.decode(errors="replace"))

    try:
        fallback = await asyncio.create_subprocess_exec(
            "patch",
            "-p1",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        _write_failed_patch(patch_text, root)
        return []

    out, err = await fallback.communicate(patch_text.encode())
    if fallback.returncode == 0:
        return target_paths

    logger.error(
        "Patch application failed: %s",
        err.decode(errors="replace") or out.decode(errors="replace"),
    )
    _write_failed_patch(patch_text, root)
    return []


async def _verify_syntax(files: list[Path]) -> str | None:
    """Verify Python syntax for changed files using py_compile."""
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
        try:
            safe_paths.append(security.validate_path(path, root))
        except PermissionError as exc:
            logger.debug("Skipping revert for unsafe path %s: %s", path, exc)

    if not safe_paths:
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
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
    strategies = _build_strategy_plan(attempt_cap)
    changed_files: set[Path] = set()
    attempt_history: list[AttemptContext] = []
    history: list[Message] = []
    seen_patch_hashes: set[str] = set()
    previous_active_root = _ACTIVE_ROOT
    _ACTIVE_ROOT = root

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
            exit_code, stdout, stderr = await _run_shell_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded. Exiting repair loop.[/green]")
                if auto_archive:
                    await _archive_session_summary(
                        fetch_response,
                        base_prompt=base_prompt,
                        command=command,
                        last_error=attempt_history[-1].last_error if attempt_history else None,
                        model=model,
                        web_access=web_access,
                    )
                return True

            current_error = _summarize_output(stdout, stderr, config.max_command_output_chars)
            console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {current_error}")

            loop_detected = _detect_repeated_failure(attempt_history, current_error)
            strategy_override = STRATEGY_OVERRIDE_TEXT if loop_detected else None
            reasoning_hint = (
                "Increase temperature or reasoning effort to escape repetition."
                if loop_detected
                else None
            )
            temperature_override = 0.7 if loop_detected else None
            if loop_detected:
                console.print(
                    "[yellow]Repeated failure detected; applying strategy override and higher reasoning effort.[/yellow]"
                )

            dynamic_paths: set[Path] = set(changed_files)
            if strategy_cfg.name == "deep":
                dynamic_paths = await expand_context_paths(
                    current_error, root, changed_files, config.ignore_dirs
                )
            elif strategy_cfg.name == "step_back":
                dynamic_paths = set(changed_files)

            applied_paths: list[Path] = []
            for _turn in range(5):
                iteration_prompt = _build_repair_instruction(
                    base_prompt,
                    current_error,
                    attempt_history,
                    root,
                    strategy_override=strategy_override,
                    reasoning_hint=reasoning_hint,
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
                    ld=loop_detected,
                    temp=temperature_override,
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
                    continue

                if patch_text:
                    # Check for duplicate patch (loop prevention)
                    patch_hash = _compute_patch_hash(patch_text)
                    if patch_hash in seen_patch_hashes:
                        console.print("[yellow]Duplicate patch detected - skipping.[/yellow]")
                        _append_history(
                            history,
                            Message(
                                role="user",
                                content="<GovernanceViolation> You proposed a patch identical to a previous failed attempt. You are looping. Try a different approach. </GovernanceViolation>",
                            ),
                        )
                        continue

                    # New patch - add to seen set
                    seen_patch_hashes.add(patch_hash)

                    console.print("[green]Agent proposed a fix.[/green]")
                    applied_paths = await _apply_patch_text(patch_text, root)
                    syntax_error = await _verify_syntax(applied_paths)
                    if syntax_error:
                        console.print(
                            f"[red]Syntax Check Failed (Self-Correction):[/red] {syntax_error}"
                        )
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
                        current_error = syntax_error
                        continue

                    changed_files.update(applied_paths)
                    break

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
                break

            exit_code, stdout, stderr = await _run_shell_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded after applying fixes.[/green]")
                if auto_archive:
                    await _archive_session_summary(
                        fetch_response,
                        base_prompt=base_prompt,
                        command=command,
                        last_error=current_error,
                        model=model,
                        web_access=web_access,
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
        exit_code, stdout, stderr = await _run_shell_command(command, root)
        if exit_code == 0:
            console.print("[green]Command succeeded after final verification.[/green]")
            if auto_archive:
                await _archive_session_summary(
                    fetch_response,
                    base_prompt=base_prompt,
                    command=command,
                    last_error=None if not attempt_history else attempt_history[-1].last_error,
                    model=model,
                    web_access=web_access,
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
        _ACTIVE_ROOT = previous_active_root


__all__ = [
    "STRATEGY_OVERRIDE_TEXT",
    "AttemptContext",
    "PatchFetcher",
    "RepairStrategy",
    "ResponseFetcher",
    "SecurityError",
    "StrategyConfig",
    "run_repair_loop",
]
