from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from jpscripts.commands import git_ops

@pytest.mark.asyncio
async def test_fetch_repo_success():
    """Verify _fetch_repo calls git fetch and reports success."""
    # Use new_callable=AsyncMock to safely mock the coroutine function
    with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_exec:
        # Mock the process object returned by the shell command
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        # When the shell command is awaited, return our process mock
        mock_exec.return_value = mock_proc

        result = await git_ops._fetch_repo(Path("/tmp/repo"))

        assert "[green]fetched[/]" in result
        mock_exec.assert_called_once()

        # Verify arguments
        args, kwargs = mock_exec.call_args
        assert "git fetch --all" in args[0]
        assert kwargs["cwd"] == Path("/tmp/repo")

@pytest.mark.asyncio
async def test_fetch_repo_failure():
    """Verify _fetch_repo handles non-zero exit codes."""
    with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as mock_exec:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
        mock_proc.returncode = 1

        mock_exec.return_value = mock_proc

        result = await git_ops._fetch_repo(Path("/tmp/repo"))

        assert "[red]failed[/]" in result
