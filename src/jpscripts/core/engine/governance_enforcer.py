"""Governance enforcement for agent responses.

This module provides constitutional compliance checking for agent patches:
- Fatal violation detection (SHELL_TRUE, OS_SYSTEM, BARE_EXCEPT)
- Retry mechanism with agent feedback
- Hard-gating to prevent unsafe code
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from jpscripts.core.console import get_logger
from jpscripts.core.governance import (
    check_compliance,
    format_violations_for_agent,
    has_fatal_violations,
)
from jpscripts.core.result import ToolExecutionError

from .models import Message, PreparedPrompt

logger = get_logger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


async def enforce_governance(
    response: ResponseT,
    history: list[Message],
    prepared: PreparedPrompt,
    raw_response: str,
    workspace_root: Path,
    render_prompt: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]],
    fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
    parser: Callable[[str], ResponseT],
) -> tuple[ResponseT, PreparedPrompt, str]:
    """Check response for constitutional violations and request corrections.

    Implements hard-gating strategy:
    - Fatal violations (SHELL_TRUE, OS_SYSTEM, BARE_EXCEPT) DROP the patch
    - Non-fatal violations trigger retry with agent feedback
    - Maximum 3 retry attempts before raising ToolExecutionError

    Args:
        response: The parsed agent response
        history: Conversation history for context
        prepared: Prepared prompt issued for this attempt
        raw_response: Raw model output tied to the parsed response
        workspace_root: Root directory for compliance checking
        render_prompt: Function to render prompts from history
        fetch_response: Function to fetch model responses
        parser: Function to parse raw responses

    Returns:
        Tuple of (response, prepared_prompt, raw_response) representing the final compliant attempt

    Raises:
        ToolExecutionError: If fatal violations persist after max retries
    """
    max_retries = 3
    current_response = response
    current_history = list(history)
    current_prepared = prepared
    current_raw = raw_response

    for attempt in range(max_retries):
        # Only check responses with file patches
        if not hasattr(current_response, "file_patch"):
            return current_response, current_prepared, current_raw

        file_patch = getattr(current_response, "file_patch", None)
        if not file_patch:
            return current_response, current_prepared, current_raw

        # Check for violations
        violations = check_compliance(str(file_patch), workspace_root)
        if not violations:
            return current_response, current_prepared, current_raw

        # Log violations
        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = len(violations) - error_count
        fatal_count = sum(1 for v in violations if v.fatal)
        logger.warning(
            "Governance violations detected (attempt %d/%d): %d errors (%d fatal), %d warnings",
            attempt + 1,
            max_retries,
            error_count,
            fatal_count,
            warning_count,
        )

        # Check for fatal violations - DROP the patch
        if has_fatal_violations(violations):
            logger.error(
                "Fatal governance violation detected - patch DROPPED (attempt %d/%d)",
                attempt + 1,
                max_retries,
            )

            # Last attempt - raise error
            if attempt >= max_retries - 1:
                fatal_msgs = [
                    f"{v.type.name} at {v.file.name}:{v.line}: {v.message}"
                    for v in violations
                    if v.fatal
                ]
                raise ToolExecutionError(
                    f"Fatal governance violations after {max_retries} attempts:\n"
                    + "\n".join(fatal_msgs)
                )

            # Format feedback and inject into history for retry
            feedback = format_violations_for_agent(violations)
            governance_message = Message(
                role="system",
                content=(
                    f"<GovernanceViolation severity='FATAL'>\n"
                    f"Your patch was REJECTED and NOT APPLIED due to fatal violations.\n"
                    f"{feedback}\n"
                    f"</GovernanceViolation>"
                ),
            )
            current_history = [*current_history, governance_message]

            # Re-prompt for correction
            try:
                current_prepared = await render_prompt(current_history)
                current_raw = await fetch_response(current_prepared)
                current_response = parser(current_raw)
                continue  # Check the new response
            except Exception as exc:
                logger.warning("Governance correction failed: %s", exc)
                raise ToolExecutionError(
                    f"Governance correction failed after fatal violation: {exc}"
                ) from exc

        # Non-fatal violations: warn and return (allow patch with warnings)
        logger.warning("Non-fatal governance violations detected, proceeding with warnings")
        return current_response, current_prepared, current_raw

    # Should not reach here, but fail safely
    raise ToolExecutionError(f"Governance enforcement exceeded {max_retries} attempts")


__all__ = [
    "enforce_governance",
]
