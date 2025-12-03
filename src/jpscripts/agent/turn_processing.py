"""Turn processing helpers for the repair loop.

This module contains helper functions and dataclasses used during
individual turns within the repair loop. These are extracted from
execution.py to keep the orchestrator focused on high-level flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jpscripts.agent import ops
from jpscripts.agent.archive import archive_session_summary as _archive_session_summary
from jpscripts.agent.context import expand_context_paths
from jpscripts.agent.models import (
    AgentEvent,
    AgentResponse,
    EventKind,
    Message,
    ResponseFetcher,
    ToolCall,
)
from jpscripts.agent.patching import apply_patch_text, compute_patch_hash
from jpscripts.agent.strategies import (
    STRATEGY_OVERRIDE_TEXT,
    AttemptContext,
    detect_repeated_failure,
)

if TYPE_CHECKING:
    from jpscripts.agent.engine import AgentEngine
    from jpscripts.core.config import AppConfig


# ---------------------------------------------------------------------------
# Internal Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Result of processing a single agent turn."""

    should_break: bool = False
    """Whether to break out of the turn loop."""
    error_message: str | None = None
    """Error message if turn failed."""
    applied_paths: list[Path] | None = None
    """Paths modified by patch application."""
    events: list[AgentEvent] = field(default_factory=list)
    """Events generated during this turn."""


@dataclass
class LoopContext:
    """Context for loop detection and strategy adjustment."""

    loop_detected: bool
    strategy_override: str | None
    reasoning_hint: str | None
    temperature_override: float | None
    event: AgentEvent | None = None
    """Event to emit if loop was detected."""


# ---------------------------------------------------------------------------
# History Management
# ---------------------------------------------------------------------------


def append_history(history: list[Message], entry: Message, keep: int = 3) -> None:
    """Append an entry to history, trimming to keep last N entries."""
    history.append(entry)
    if len(history) > keep:
        del history[:-keep]


# ---------------------------------------------------------------------------
# Success Handling
# ---------------------------------------------------------------------------


async def handle_success_and_archive(
    auto_archive: bool,
    fetch_response: ResponseFetcher,
    config: AppConfig,
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
            config,
            base_prompt=base_prompt,
            command=command,
            last_error=last_error,
            model=model,
            web_access=web_access,
        )


# ---------------------------------------------------------------------------
# Tool Call Processing
# ---------------------------------------------------------------------------


async def process_tool_call(
    engine: AgentEngine[AgentResponse],
    tool_call: ToolCall,
    thought: str,
    history: list[Message],
) -> list[AgentEvent]:
    """Process a tool call from the agent.

    Returns:
        List of events generated during tool processing.
    """
    events: list[AgentEvent] = []
    tool_name = tool_call.tool
    tool_args = tool_call.arguments

    events.append(
        AgentEvent(
            EventKind.TOOL_CALL,
            f"Agent invoking {tool_name}",
            {"tool_name": tool_name, "arguments": tool_args},
        )
    )

    try:
        output = await engine.execute_tool(tool_call)
    except Exception as exc:
        output = f"Tool execution failed: {exc}"

    events.append(AgentEvent(EventKind.TOOL_OUTPUT, output, {"output": output}))

    history_entry = (
        "<Turn>\n"
        f"Agent thought: {thought}\n"
        f"Tool call: {tool_name}({tool_args})\n"
        f"Tool output: {output}\n"
        "</Turn>"
    )
    append_history(history, Message(role="system", content=history_entry))

    return events


# ---------------------------------------------------------------------------
# Patch Processing
# ---------------------------------------------------------------------------


async def process_patch(
    patch_text: str,
    thought: str,
    root: Path,
    seen_patch_hashes: set[str],
    changed_files: set[Path],
    history: list[Message],
) -> TurnResult:
    """Process a patch from the agent."""
    events: list[AgentEvent] = []
    patch_hash = compute_patch_hash(patch_text)

    if patch_hash in seen_patch_hashes:
        events.append(AgentEvent(EventKind.DUPLICATE_PATCH, "Duplicate patch detected - skipping."))
        append_history(
            history,
            Message(
                role="user",
                content="<GovernanceViolation> You proposed a patch identical to a previous failed attempt. You are looping. Try a different approach. </GovernanceViolation>",
            ),
        )
        return TurnResult(should_break=False, events=events)

    seen_patch_hashes.add(patch_hash)
    events.append(AgentEvent(EventKind.PATCH_PROPOSED, "Agent proposed a fix."))
    applied_paths = await apply_patch_text(patch_text, root)
    events.append(
        AgentEvent(
            EventKind.PATCH_APPLIED,
            f"Applied patch to {len(applied_paths)} file(s)",
            {"files": [str(p) for p in applied_paths]},
        )
    )
    syntax_error = await ops.verify_syntax(applied_paths)

    if syntax_error:
        events.append(
            AgentEvent(
                EventKind.SYNTAX_ERROR,
                "Syntax check failed",
                {"error": syntax_error},
            )
        )
        append_history(
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
        return TurnResult(should_break=False, error_message=syntax_error, events=events)

    changed_files.update(applied_paths)
    return TurnResult(should_break=True, applied_paths=applied_paths, events=events)


# ---------------------------------------------------------------------------
# No-Patch Handling
# ---------------------------------------------------------------------------


def handle_no_patch(
    agent_response: AgentResponse,
    thought: str,
    history: list[Message],
) -> TurnResult:
    """Handle when agent returns no patch."""
    message = agent_response.final_message or "Agent returned no patch content."
    events = [AgentEvent(EventKind.NO_PATCH, message, {"message": message})]
    append_history(
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
    return TurnResult(should_break=True, events=events)


# ---------------------------------------------------------------------------
# Loop Detection
# ---------------------------------------------------------------------------


def setup_loop_context(
    attempt_history: list[AttemptContext],
    current_error: str,
) -> LoopContext:
    """Set up context based on loop detection."""
    loop_detected = detect_repeated_failure(attempt_history, current_error)
    event: AgentEvent | None = None
    if loop_detected:
        event = AgentEvent(
            EventKind.LOOP_DETECTED,
            "Repeated failure detected; applying strategy override and higher reasoning effort.",
        )
    return LoopContext(
        loop_detected=loop_detected,
        strategy_override=STRATEGY_OVERRIDE_TEXT if loop_detected else None,
        reasoning_hint=(
            "Increase temperature or reasoning effort to escape repetition."
            if loop_detected
            else None
        ),
        temperature_override=0.7 if loop_detected else None,
        event=event,
    )


# ---------------------------------------------------------------------------
# Dynamic Path Expansion
# ---------------------------------------------------------------------------


async def get_dynamic_paths(
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


__all__ = [
    "LoopContext",
    "TurnResult",
    "append_history",
    "get_dynamic_paths",
    "handle_no_patch",
    "handle_success_and_archive",
    "process_patch",
    "process_tool_call",
    "setup_loop_context",
]
