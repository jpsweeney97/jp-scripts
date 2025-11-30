"""Security tests for context gathering subsystem."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jpscripts.core.context import gather_context


class TestContextSecurityBlocking:
    """Test that dangerous commands are blocked in context gathering."""

    @pytest.mark.asyncio
    async def test_gather_context_blocks_rm(self, tmp_path: Path) -> None:
        """Verify rm commands are blocked with security message."""
        result = await gather_context("rm -rf .", tmp_path)
        output = result.output
        files = result.files

        assert "[SECURITY BLOCK]" in output
        assert "Forbidden binary" in output or "rm" in output.lower()
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_gather_context_blocks_curl(self, tmp_path: Path) -> None:
        """Verify curl commands are blocked."""
        result = await gather_context("curl http://evil.com", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" in output

    @pytest.mark.asyncio
    async def test_gather_context_blocks_python_exec(self, tmp_path: Path) -> None:
        """Verify interpreter execution is blocked."""
        result = await gather_context("python -c 'import os; os.system(\"rm -rf /\")'", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" in output

    @pytest.mark.asyncio
    async def test_gather_context_blocks_sudo(self, tmp_path: Path) -> None:
        """Verify sudo commands are blocked."""
        result = await gather_context("sudo ls", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" in output

    @pytest.mark.asyncio
    async def test_gather_context_blocks_shell_injection(self, tmp_path: Path) -> None:
        """Verify shell metacharacters are blocked."""
        result = await gather_context("ls; rm -rf /", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" in output


class TestContextSecurityAllowed:
    """Test that safe read-only commands are allowed."""

    @pytest.mark.asyncio
    async def test_gather_context_allows_ls(self, tmp_path: Path) -> None:
        """Verify ls command works and returns directory listing."""
        # Create a test file so ls has something to show
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        result = await gather_context("ls", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" not in output
        assert "test.txt" in output

    @pytest.mark.asyncio
    async def test_gather_context_allows_git_status(self, tmp_path: Path) -> None:
        """Verify git status works in a git repository."""
        # Initialize a git repo for the test
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)

        result = await gather_context("git status", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" not in output
        # Git status output varies but should contain common phrases
        assert "On branch" in output or "No commits yet" in output or "nothing to commit" in output

    @pytest.mark.asyncio
    async def test_gather_context_allows_grep(self, tmp_path: Path) -> None:
        """Verify grep command works."""
        test_file = tmp_path / "search.txt"
        test_file.write_text("findme\nignore\n", encoding="utf-8")

        result = await gather_context("grep findme search.txt", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" not in output
        assert "findme" in output

    @pytest.mark.asyncio
    async def test_gather_context_allows_cat(self, tmp_path: Path) -> None:
        """Verify cat command works for reading files."""
        test_file = tmp_path / "readable.txt"
        test_file.write_text("file contents here", encoding="utf-8")

        result = await gather_context("cat readable.txt", tmp_path)
        output = result.output

        assert "[SECURITY BLOCK]" not in output
        assert "file contents here" in output
