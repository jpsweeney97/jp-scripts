"""Safety monitoring and circuit breaker logic.

This module provides:
- Token estimation using tiktoken
- Circuit breaker enforcement
- Black box crash report generation
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

from jpscripts.core import runtime
from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.runtime import CircuitBreaker

from .models import SafetyLockdownError

# Pre-warm tiktoken encoder at module import time.
# This adds ~100ms to import but avoids a blocking call during agent execution.
_TOKENIZER: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")


def _get_tokenizer() -> tiktoken.Encoding:
    """Get the tiktoken encoder (cl100k_base for GPT-4/Claude)."""
    return _TOKENIZER


def _approximate_tokens(content: str) -> int:
    """Count tokens using tiktoken for accuracy (with fallback)."""
    if not content:
        return 0
    try:
        return len(_get_tokenizer().encode(content, disallowed_special=()))
    except Exception:
        # Fallback to char/4 estimate if tiktoken fails
        return max(1, len(content) // 4)


def _estimate_token_usage(prompt_text: str, completion_text: str) -> TokenUsage:
    """Token estimate using tiktoken for circuit breaker budget tracking."""
    return TokenUsage(
        prompt_tokens=_approximate_tokens(prompt_text),
        completion_tokens=_approximate_tokens(completion_text),
    )


def _build_black_box_report(
    breaker: CircuitBreaker,
    *,
    usage: TokenUsage,
    files_touched: list[Path],
    persona: str,
    context: str,
) -> str:
    """Build a crash report for safety lockdown events."""
    file_lines = "\n".join(f"- {path}" for path in files_touched) if files_touched else "- (none)"
    reason = breaker.last_failure_reason or "Unknown"
    return (
        "=== Black Box Crash Report ===\n"
        f"Persona: {persona}\n"
        f"Context: {context}\n"
        f"Reason: {reason}\n"
        f"Prompt tokens: {usage.prompt_tokens}\n"
        f"Completion tokens: {usage.completion_tokens}\n"
        f"Cost estimate (USD): {breaker.last_cost_estimate}\n"
        f"Cost velocity (USD/min): {breaker.last_cost_velocity}\n"
        f"Max velocity allowed (USD/min): {breaker.max_cost_velocity}\n"
        f"File churn: {breaker.last_file_churn}\n"
        f"Max churn allowed: {breaker.max_file_churn}\n"
        "\nFiles touched:\n"
        f"{file_lines}"
    )


def enforce_circuit_breaker(
    *,
    usage: TokenUsage,
    files_touched: list[Path],
    persona: str,
    context: str,
) -> None:
    """Check circuit breaker and raise SafetyLockdownError if triggered.

    Args:
        usage: Token usage for this operation
        files_touched: Files modified in this operation
        persona: Agent persona for reporting
        context: Context description for reporting

    Raises:
        SafetyLockdownError: If circuit breaker is triggered
    """
    from jpscripts.core.console import get_logger

    logger = get_logger(__name__)

    breaker = runtime.get_circuit_breaker()
    if breaker.check_health(usage, files_touched):
        return

    report = _build_black_box_report(
        breaker,
        usage=usage,
        files_touched=files_touched,
        persona=persona,
        context=context,
    )
    logger.error("Circuit breaker triggered: %s", breaker.last_failure_reason)
    raise SafetyLockdownError(report)


__all__ = [
    "_approximate_tokens",
    "_build_black_box_report",
    "_estimate_token_usage",
    "_get_tokenizer",
    "enforce_circuit_breaker",
]
