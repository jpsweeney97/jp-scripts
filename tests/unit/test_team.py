from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from jpscripts.core.team import (
    AgentTurnResponse,
    Objective,
    PlanStep,
    Persona,
    SwarmState,
    _compose_prompt,
    _parse_agent_turn,
    get_default_swarm,
)
from jpscripts.core.config import AppConfig


def test_compose_prompt_includes_schema_for_architect(tmp_path: Path) -> None:
    swarm_state = SwarmState(
        objective=Objective(summary="Ship feature", constraints=[]),
        plan_steps=[PlanStep(summary="Do work")],
        current_phase="planning",
        artifacts=[],
    )
    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, log_level="INFO")
    architect = get_default_swarm()[0]
    prompt = _compose_prompt(
        architect,
        "Ship feature",
        swarm_state,
        context_log="",
        config=config,
        safe_mode=False,
        repo_root=tmp_path,
        context_files=[],
        max_file_context_chars=1000,
    )

    schema = json.dumps(AgentTurnResponse.model_json_schema(), indent=2)
    assert schema.strip() in prompt


def test_validate_swarm_output_accepts_valid_json() -> None:
    payload = AgentTurnResponse(
        swarm_state=SwarmState(
            objective=Objective(summary="Refactor"),
            plan_steps=[PlanStep(summary="Step 1", status="pending")],
            current_phase="planning",
            artifacts=[],
        ),
        next_step="engineer",
    ).model_dump_json()

    agent = SimpleNamespace(captured_raw=payload, captured_stdout="")
    parsed, error_text = _parse_agent_turn(agent)  # type: ignore[arg-type]

    assert parsed is not None
    assert error_text == ""
    assert parsed.swarm_state.objective.summary == "Refactor"
    assert parsed.next_step == "engineer"


def test_validate_swarm_output_returns_error() -> None:
    agent = SimpleNamespace(captured_raw="not-json", captured_stdout="")
    parsed, error_text = _parse_agent_turn(agent)  # type: ignore[arg-type]

    assert parsed is None
    assert "json_invalid" in error_text
