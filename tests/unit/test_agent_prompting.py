from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jpscripts.core.agent import prepare_agent_prompt
from jpscripts.core.context import GatherContextResult
from jpscripts.core.config import AppConfig
from jpscripts.core.runtime import runtime_context


@pytest.mark.asyncio
async def test_prepare_agent_prompt_includes_git_context(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"

    with patch(
        "jpscripts.core.agent._collect_git_context",
        AsyncMock(return_value=("feature/test", "abc1234", False)),
    ), patch(
        "jpscripts.core.agent.gather_context",
        AsyncMock(return_value=GatherContextResult(output="log output", files={file_path})),
    ), patch(
        "jpscripts.core.agent.smart_read_context",
        return_value="file snippet",
    ):
        config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, use_semantic_search=False)
        with runtime_context(config, workspace=tmp_path):
            prepared = await prepare_agent_prompt(
                "Do the thing",
                run_command="echo hi",
                attach_recent=False,
                include_diff=False,
                ignore_dirs=[],
                max_file_context_chars=5000,
                max_command_output_chars=1000,
            )

    prompt = json.loads(prepared.prompt)
    git_ctx = prompt["system_context"]["git_context"]
    assert git_ctx["branch"] == "feature/test"
    assert git_ctx["head"] == "abc1234"
    assert git_ctx["dirty"] is False
    assert "diagnostic" in prompt
    assert "sample.txt" in prompt["file_context"]
    assert "constitution" in prompt["system_context"]
    assert "response_contract" in prompt
    assert "tools" in prompt


@pytest.mark.asyncio
async def test_prepare_agent_prompt_marks_dirty_and_handles_empty_diff(tmp_path: Path) -> None:
    with patch(
        "jpscripts.core.agent._collect_git_context",
        AsyncMock(return_value=("main", "deadbee", True)),
    ), patch(
        "jpscripts.core.agent._collect_git_diff",
        AsyncMock(return_value=None),
    ):
        config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, use_semantic_search=False)
        with runtime_context(config, workspace=tmp_path):
            prepared = await prepare_agent_prompt(
                "Check dirty state",
                run_command=None,
                attach_recent=False,
                include_diff=True,
                ignore_dirs=[],
                max_file_context_chars=5000,
                max_command_output_chars=1000,
            )

    prompt = json.loads(prepared.prompt)
    assert prompt["system_context"]["git_context"]["dirty"] is True
    assert prompt["git_diff"] == "NO CHANGES"


@pytest.mark.asyncio
async def test_prepare_agent_prompt_includes_constitution_file(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        json.dumps({"constitution": {"invariants": [{"id": "test", "rules": ["Rule 1: Be helpful."]}]}}),
        encoding="utf-8",
    )

    with patch(
        "jpscripts.core.agent._collect_git_context",
        AsyncMock(return_value=("main", "deadbee", False)),
    ):
        config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, use_semantic_search=False)
        with runtime_context(config, workspace=tmp_path):
            prepared = await prepare_agent_prompt(
                "Honor the rules",
                run_command=None,
                attach_recent=False,
                include_diff=False,
                ignore_dirs=[],
                max_file_context_chars=5000,
                max_command_output_chars=1000,
            )

    prompt = json.loads(prepared.prompt)
    constitution = prompt["system_context"]["constitution"]
    assert constitution["invariants"][0]["rules"][0] == "Rule 1: Be helpful."
    assert "response_contract" in prompt
