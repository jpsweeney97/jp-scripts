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

# Export from new agent submodules
from jpscripts.agent.circuit import enforce_circuit_breaker

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
from jpscripts.agent.engine import AgentEngine
from jpscripts.agent.middleware import (
    AgentMiddleware,
    BaseMiddleware,
    CircuitBreakerMiddleware,
    GovernanceMiddleware,
    StepContext,
    TracingMiddleware,
    run_middleware_pipeline,
)

# Export from execution module
from jpscripts.agent.execution import (
    RepairLoopOrchestrator,
    apply_patch_text,
    run_repair_loop,
)
from jpscripts.agent.governance import enforce_governance

# Export from models module (merged from types and engine)
from jpscripts.agent.models import (
    AgentEvent,
    AgentResponse,
    AgentTraceStep,
    EventKind,
    MemoryProtocol,
    Message,
    PatchFetcher,
    PreparedPrompt,
    RepairLoopConfig,
    ResponseFetcher,
    ResponseT,
    SafetyLockdownError,
    SecurityError,
    ToolCall,
)

# Export from ops module
from jpscripts.agent.ops import verify_syntax
from jpscripts.agent.parsing import parse_agent_response

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
from jpscripts.agent.tools import AUDIT_PREFIX, execute_tool, run_safe_shell
from jpscripts.agent.tracing import TraceRecorder

__all__ = [
    # Constants
    "AGENT_TEMPLATE_NAME",
    "AUDIT_PREFIX",
    "GOVERNANCE_ANTI_PATTERNS",
    # Classes
    "AgentEngine",
    "AgentEvent",
    "AgentMiddleware",
    "AgentResponse",
    "AgentTraceStep",
    "AttemptContext",
    "BaseMiddleware",
    "CircuitBreakerMiddleware",
    "GovernanceMiddleware",
    "StepContext",
    "TracingMiddleware",
    # Enums and Types
    "EventKind",
    "MemoryProtocol",
    "Message",
    "PatchFetcher",
    "PreparedPrompt",
    "RepairLoopConfig",
    "RepairLoopOrchestrator",
    "RepairStrategy",
    "ResponseFetcher",
    "ResponseT",
    "SafetyLockdownError",
    "SecurityError",
    "StrategyConfig",
    "ToolCall",
    "TraceRecorder",
    # Functions
    "apply_patch_text",
    "build_dependency_section",
    "build_file_context_section",
    "collect_git_context",
    "collect_git_diff",
    "enforce_circuit_breaker",
    "enforce_governance",
    "execute_tool",
    "expand_context_paths",
    "load_constitution",
    "parse_agent_response",
    "prepare_agent_prompt",
    "run_middleware_pipeline",
    "run_repair_loop",
    "run_safe_shell",
    "scan_recent",
    "verify_syntax",
]
