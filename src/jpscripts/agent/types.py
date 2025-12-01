"""Agent type definitions and data structures.

This module contains the core types used throughout the agent subsystem,
including event types, configuration dataclasses, and type aliases.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jpscripts.engine import PreparedPrompt


class SecurityError(RuntimeError):
    """Raised when a tool invocation is considered unsafe."""


# Type aliases for async fetchers
PatchFetcher = Callable[["PreparedPrompt"], Awaitable[str]]
ResponseFetcher = Callable[["PreparedPrompt"], Awaitable[str]]


class EventKind(Enum):
    """Types of events from the repair loop."""

    ATTEMPT_START = "attempt_start"
    COMMAND_SUCCESS = "command_success"
    COMMAND_FAILED = "command_failed"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    PATCH_PROPOSED = "patch_proposed"
    PATCH_APPLIED = "patch_applied"
    SYNTAX_ERROR = "syntax_error"
    DUPLICATE_PATCH = "duplicate_patch"
    LOOP_DETECTED = "loop_detected"
    VALIDATION_ERROR = "validation_error"
    NO_PATCH = "no_patch"
    REVERTING = "reverting"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Structured event from repair loop operations."""

    kind: EventKind
    message: str
    data: dict[str, Any] = field(default_factory=dict)


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


__all__ = [
    "AgentEvent",
    "EventKind",
    "PatchFetcher",
    "RepairLoopConfig",
    "ResponseFetcher",
    "SecurityError",
]
