from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from jpscripts.commands import git_ops as cmd_git_ops
from jpscripts.core import git_ops as core_git_ops
from jpscripts.core.result import Ok


@pytest.mark.asyncio
async def test_fetch_repo_success() -> None:
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b""))
    mock_process.returncode = 0
    mock_process.kill = AsyncMock()

    with patch("jpscripts.commands.git_ops._has_remotes", return_value=True), patch(
        "jpscripts.commands.git_ops.asyncio.create_subprocess_exec", return_value=mock_process
    ) as mock_subproc:
        result = await cmd_git_ops._fetch_repo(Path("/tmp/repo"))

    assert result == "[green]fetched[/]"
    mock_subproc.assert_awaited_once()
    mock_process.communicate.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_repo_failure() -> None:
    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"fatal error"))
    mock_process.returncode = 1
    mock_process.kill = AsyncMock()

    with patch("jpscripts.commands.git_ops._has_remotes", return_value=True), patch(
        "jpscripts.commands.git_ops.asyncio.create_subprocess_exec", return_value=mock_process
    ):
        result = await cmd_git_ops._fetch_repo(Path("/tmp/repo"))

    assert result.startswith("[red]failed")
    mock_process.communicate.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_repo_no_remotes() -> None:
    with patch("jpscripts.commands.git_ops._has_remotes", return_value=False):
        result = await cmd_git_ops._fetch_repo(Path("/tmp/repo"))

    assert result == "[green]fetched (no remotes)[/]"


class _FakeAsyncRepo:
    def __init__(self, output: str) -> None:
        self.output = output

    async def _run_git(self, *args: str) -> Ok[str]:  # noqa: ANN204
        return Ok(self.output)


@pytest.mark.asyncio
async def test_branch_statuses_parses_ahead_behind() -> None:
    output = "main origin/main [ahead 2, behind 1]\nfeature  [ahead 1]\nlegacy origin/legacy [behind 3]\n"
    repo = _FakeAsyncRepo(output)

    summaries = (await core_git_ops.branch_statuses(repo)).unwrap()  # type: ignore[arg-type]

    assert summaries[0] == core_git_ops.BranchSummary("main", "origin/main", 2, 1, None)
    assert summaries[1] == core_git_ops.BranchSummary("feature", None, 1, 0, None)
    assert summaries[2] == core_git_ops.BranchSummary("legacy", "origin/legacy", 0, 3, None)
