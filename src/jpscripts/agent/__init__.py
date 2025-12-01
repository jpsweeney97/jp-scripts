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

# Export context helpers for internal use and test patching
from jpscripts.agent.context import (
    build_dependency_section,
    build_file_context_section,
    collect_git_context,
    collect_git_diff,
    expand_context_paths,
    load_constitution,
    scan_recent,
)

# Export from execution module
from jpscripts.agent.execution import (
    AgentEvent,
    EventKind,
    PatchFetcher,
    RepairLoopConfig,
    RepairLoopOrchestrator,
    ResponseFetcher,
    SecurityError,
    apply_patch_text,
    run_repair_loop,
    verify_syntax,
)

# Export from prompting module
from jpscripts.agent.prompting import (
    AGENT_TEMPLATE_NAME,
    GOVERNANCE_ANTI_PATTERNS,
    prepare_agent_prompt,
)

# Export from strategies module
from jpscripts.agent.strategies import (
    AttemptContext,
    RepairStrategy,
    StrategyConfig,
)

# Re-export from engine for backwards compatibility
from jpscripts.engine import (
    PreparedPrompt,
    parse_agent_response,
)

__all__ = [
    "AGENT_TEMPLATE_NAME",
    "GOVERNANCE_ANTI_PATTERNS",
    "AgentEvent",
    "AttemptContext",
    "EventKind",
    "PatchFetcher",
    "PreparedPrompt",
    "RepairLoopConfig",
    "RepairLoopOrchestrator",
    "RepairStrategy",
    "ResponseFetcher",
    "SecurityError",
    "StrategyConfig",
    "apply_patch_text",
    "build_dependency_section",
    "build_file_context_section",
    "collect_git_context",
    "collect_git_diff",
    "expand_context_paths",
    "load_constitution",
    "parse_agent_response",
    "prepare_agent_prompt",
    "run_repair_loop",
    "scan_recent",
    "verify_syntax",
]
