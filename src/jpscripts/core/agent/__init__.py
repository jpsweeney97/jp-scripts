"""Agent module - autonomous repair and prompt building.

This package provides the core agent functionality for autonomous code repair,
including prompt preparation, context gathering, and repair loop execution.

Public API:
    - prepare_agent_prompt: Build structured prompts for LLM agents
    - run_repair_loop: Execute autonomous repair loops
    - PreparedPrompt: Container for prepared prompts (from engine)
    - parse_agent_response: Parse JSON agent responses (from engine)
"""
from __future__ import annotations

# Re-export from engine for backwards compatibility
from jpscripts.core.engine import (
    PreparedPrompt,
    parse_agent_response,
)

# Export from prompting module
from jpscripts.core.agent.prompting import (
    AGENT_TEMPLATE_NAME,
    GOVERNANCE_ANTI_PATTERNS,
    prepare_agent_prompt,
)

# Export from execution module
from jpscripts.core.agent.execution import (
    AttemptContext,
    PatchFetcher,
    RepairStrategy,
    ResponseFetcher,
    SecurityError,
    StrategyConfig,
    run_repair_loop,
)

# Export context helpers for internal use and test patching
from jpscripts.core.agent.context import (
    build_dependency_section,
    build_file_context_section,
    collect_git_context,
    collect_git_diff,
    expand_context_paths,
    load_constitution,
    scan_recent,
)

__all__ = [
    # Re-exported from engine (backwards compatibility)
    "PreparedPrompt",
    "parse_agent_response",
    # Prompt building
    "AGENT_TEMPLATE_NAME",
    "GOVERNANCE_ANTI_PATTERNS",
    "prepare_agent_prompt",
    # Execution
    "run_repair_loop",
    "SecurityError",
    "AttemptContext",
    "StrategyConfig",
    "RepairStrategy",
    "PatchFetcher",
    "ResponseFetcher",
    # Context helpers (for internal use)
    "load_constitution",
    "collect_git_context",
    "collect_git_diff",
    "build_file_context_section",
    "build_dependency_section",
    "expand_context_paths",
    "scan_recent",
]
