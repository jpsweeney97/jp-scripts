"""Evolution types and error definitions.

This module provides:
- EvolutionError: Base error for evolution operations
- EvolutionResult: Success result from evolution run
- VerificationResult: Result from test verification
- PRCreationResult: Result from PR creation
"""

from __future__ import annotations

from dataclasses import dataclass

from jpscripts.analysis.complexity import TechnicalDebtScore
from jpscripts.core.result import JPScriptsError


class EvolutionError(JPScriptsError):
    """Base error for evolution operations."""

    pass


@dataclass(frozen=True)
class VerificationResult:
    """Result of verification step."""

    test_command: str
    exit_code: int
    success: bool
    output: str | None = None


@dataclass(frozen=True)
class PRCreationResult:
    """Result of PR creation."""

    pr_url: str | None
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class EvolutionResult:
    """Result of a successful evolution run."""

    target: TechnicalDebtScore
    branch_name: str
    pr_url: str | None


__all__ = [
    "EvolutionError",
    "EvolutionResult",
    "PRCreationResult",
    "VerificationResult",
]
