"""Prompt building for evolution optimization.

This module provides:
- build_optimizer_prompt: Create LLM prompts for the Optimizer persona
"""

from __future__ import annotations

from jpscripts.analysis.complexity import TechnicalDebtScore


def build_optimizer_prompt(target: TechnicalDebtScore) -> str:
    """Build the prompt for the Optimizer persona.

    Args:
        target: The file selected for optimization with its debt metrics

    Returns:
        A prompt string for the LLM optimizer agent
    """
    reasons_text = (
        "\n".join(f"- {r}" for r in target.reasons) if target.reasons else "High complexity"
    )

    return f"""You are an Optimizer persona. Your task is to reduce technical debt in a specific file.

Target file: {target.path}
Current complexity score: {target.complexity_score:.1f}
Historical fix frequency: {target.fix_frequency}
Git churn (commit count): {target.churn}
Identified issues:
{reasons_text}

Your objectives:
1. **Reduce cyclomatic complexity** by extracting helper functions or simplifying logic
2. **Improve code clarity** with better naming and structure
3. **Add or improve type annotations** where missing
4. **Preserve all public interfaces** - do not change function signatures for public API
5. Ensure all changes pass `mypy --strict`

Constraints:
- Preserve all existing behavior (pure refactoring)
- All I/O must remain async where it currently is
- Follow existing patterns in the codebase
- Keep changes minimal and focused on complexity reduction

Emit a unified diff patch that addresses the technical debt. Focus on the most complex
functions first. If the file is large, prioritize the top 1-2 functions by complexity."""


__all__ = ["build_optimizer_prompt"]
