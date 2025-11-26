from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jpscripts.core.agent import prepare_agent_prompt


@pytest.mark.asyncio
async def test_prepare_agent_prompt_includes_git_context(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"

    with patch(
        "jpscripts.core.agent._collect_git_context",
        AsyncMock(return_value=("feature/test", "abc1234", False)),
    ), patch(
        "jpscripts.core.agent.gather_context",
        AsyncMock(return_value=("log output", {file_path})),
    ), patch(
        "jpscripts.core.agent.read_file_context",
        return_value="file snippet",
    ):
        prepared = await prepare_agent_prompt(
            "Do the thing",
            root=tmp_path,
            run_command="echo hi",
            attach_recent=False,
            include_diff=False,
            ignore_dirs=[],
            max_file_context_chars=5000,
            max_command_output_chars=1000,
        )

    prompt = prepared.prompt
    assert "<git_context>" in prompt
    assert "<branch>feature/test</branch>" in prompt
    assert "<head>abc1234</head>" in prompt
    assert "<dirty>False</dirty>" in prompt
    assert "<diagnostic_command>" in prompt
    assert "sample.txt" in prompt


@pytest.mark.asyncio
async def test_prepare_agent_prompt_marks_dirty_and_handles_empty_diff(tmp_path: Path) -> None:
    with patch(
        "jpscripts.core.agent._collect_git_context",
        AsyncMock(return_value=("main", "deadbee", True)),
    ), patch(
        "jpscripts.core.agent._collect_git_diff",
        AsyncMock(return_value=None),
    ):
        prepared = await prepare_agent_prompt(
            "Check dirty state",
            root=tmp_path,
            run_command=None,
            attach_recent=False,
            include_diff=True,
            ignore_dirs=[],
            max_file_context_chars=5000,
            max_command_output_chars=1000,
        )

    prompt = prepared.prompt
    assert "<dirty>True</dirty>" in prompt
    assert "<git_diff>NO CHANGES</git_diff>" in prompt
