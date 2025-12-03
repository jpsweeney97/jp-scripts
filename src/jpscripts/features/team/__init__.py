"""Team management feature.

This module provides team configuration and multi-agent swarm orchestration.
"""

# Private but exported for testing
from jpscripts.features.team.model import (
    AgentTurnResponse,
    AgentUpdate,
    Objective,
    Persona,
    PlanStep,
    SpawnRequest,
    SwarmController,
    SwarmState,
    UpdateKind,
    _render_swarm_prompt,
    get_default_swarm,
    parse_agent_turn,
    parse_swarm_response,
    swarm_chat,
)

__all__ = [
    "AgentTurnResponse",
    "AgentUpdate",
    "Objective",
    "Persona",
    "PlanStep",
    "SpawnRequest",
    "SwarmController",
    "SwarmState",
    "UpdateKind",
    "_render_swarm_prompt",
    "get_default_swarm",
    "parse_agent_turn",
    "parse_swarm_response",
    "swarm_chat",
]
