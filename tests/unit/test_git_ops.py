from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jpscripts.commands import git_ops
from jpscripts.core import git as git_core


@pytest.mark.asyncio
async def test_fetch_repo_success() -> None:
    mock_repo = MagicMock()
    mock_repo.fetch = AsyncMock()

    with patch.object(git_core, "AsyncRepo") as mock_async_repo:
        mock_async_repo.open = AsyncMock(return_value=mock_repo)

        result = await git_ops._fetch_repo(Path("/tmp/repo"))

        assert result == "[green]fetched[/]"
        mock_async_repo.open.assert_awaited_once_with(Path("/tmp/repo"))
        mock_repo.fetch.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_fetch_repo_failure() -> None:
    mock_repo = MagicMock()
    mock_repo.fetch = AsyncMock(side_effect=git_core.GitOperationError("fetch failed"))

    with patch.object(git_core, "AsyncRepo") as mock_async_repo:
        mock_async_repo.open = AsyncMock(return_value=mock_repo)

        result = await git_ops._fetch_repo(Path("/tmp/repo"))

        assert result.startswith("[red]failed")
        mock_async_repo.open.assert_awaited_once_with(Path("/tmp/repo"))
        mock_repo.fetch.assert_awaited_once_with()
