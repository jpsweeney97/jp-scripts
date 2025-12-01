"""Repair strategy definitions and prompt construction.

This module defines the available repair strategies and provides
functions to build repair instructions based on attempt history.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

STRATEGY_OVERRIDE_TEXT = (
    "You are stuck in a loop. Stop editing code. Analyze the error trace and the file content again. "
    "List three possible root causes before proposing a new patch."
)

RepairStrategy = Literal["fast", "deep", "step_back"]


@dataclass
class AttemptContext:
    iteration: int
    last_error: str
    files_changed: list[Path]
    strategy: RepairStrategy


@dataclass(frozen=True)
class StrategyConfig:
    name: RepairStrategy
    label: str
    description: str
    system_notice: str = ""


def build_history_summary(history: Sequence[AttemptContext], root: Path) -> str:
    """Summarize previous repair attempts for context."""
    if not history:
        return "None yet."

    lines: list[str] = []
    for attempt in history:
        relative_files: list[str] = []
        for path in attempt.files_changed:
            try:
                relative_files.append(str(path.relative_to(root)))
            except ValueError:
                relative_files.append(str(path))
        file_part = f" | files: {', '.join(relative_files)}" if relative_files else ""
        lines.append(f"Attempt {attempt.iteration}: {attempt.last_error}{file_part}")

    return "\n".join(lines)


def detect_repeated_failure(history: Sequence[AttemptContext], current_error: str) -> bool:
    """Check if the current error has occurred before (loop detection)."""
    normalized_current = current_error.strip()
    if not normalized_current:
        return False
    occurrences = sum(1 for attempt in history if attempt.last_error.strip() == normalized_current)
    return occurrences + 1 >= 2


def build_strategy_plan(attempt_cap: int) -> list[StrategyConfig]:
    """Build the sequence of strategies for the repair loop."""
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


def build_repair_instruction(
    base_prompt: str,
    current_error: str,
    history: Sequence[AttemptContext],
    root: Path,
    *,
    strategy_override: str | None = None,
    reasoning_hint: str | None = None,
    strategy: StrategyConfig,
) -> str:
    """Build the complete repair instruction for the agent."""
    history_block = build_history_summary(history, root)
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


__all__ = [
    "STRATEGY_OVERRIDE_TEXT",
    "AttemptContext",
    "RepairStrategy",
    "StrategyConfig",
    "build_history_summary",
    "build_repair_instruction",
    "build_strategy_plan",
    "detect_repeated_failure",
]
