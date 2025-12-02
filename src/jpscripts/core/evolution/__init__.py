"""Evolution package for autonomous code optimization.

This package provides the core logic for the `jp evolve` command,
including complexity analysis integration, optimizer prompting,
verification, and PR creation.

Public API:
    - run_evolution: Execute full evolution workflow
    - build_optimizer_prompt: Create optimizer LLM prompts
    - create_evolution_pr: Create GitHub PR for changes
    - collect_dependent_tests: Find tests affected by changes
    - run_verification: Execute verification tests
"""

from jpscripts.core.evolution.orchestrator import (
    abort_evolution,
    cleanup_branch,
    get_debt_scores,
    run_evolution,
)
from jpscripts.core.evolution.pr import create_evolution_pr
from jpscripts.core.evolution.prompting import build_optimizer_prompt
from jpscripts.core.evolution.types import (
    EvolutionError,
    EvolutionResult,
    PRCreationResult,
    VerificationResult,
)
from jpscripts.core.evolution.verification import (
    collect_dependent_tests,
    run_verification,
)

__all__ = [
    "EvolutionError",
    "EvolutionResult",
    "PRCreationResult",
    "VerificationResult",
    "abort_evolution",
    "build_optimizer_prompt",
    "cleanup_branch",
    "collect_dependent_tests",
    "create_evolution_pr",
    "get_debt_scores",
    "run_evolution",
    "run_verification",
]
