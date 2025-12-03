"""
Unit tests for MCP tools (search, system).

Tests cover search_codebase, find_todos, list_processes, kill_process,
and run_shell including error paths and security validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jpscripts.core.config import AIConfig, AppConfig


class TestSearchCodebase:
    """Test search_codebase MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        """Create a mock runtime context."""
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(
            ai=AIConfig(max_file_context_chars=50000),
            workspace_root=tmp_path,
        )
        return runtime

    @pytest.mark.asyncio
    async def test_search_codebase_success(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.search import search_codebase

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
            patch(
                "jpscripts.mcp.tools.search.search_core.run_ripgrep",
                return_value="file.py:1: match found",
            ) as mock_rg,
        ):
            from jpscripts.core.result import Ok

            mock_validate.return_value = Ok(tmp_path)
            result = await search_codebase("pattern", ".")

        assert "file.py:1: match found" in result
        mock_rg.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_codebase_no_matches(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.search import search_codebase

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
            patch("jpscripts.mcp.tools.search.search_core.run_ripgrep", return_value=""),
        ):
            from jpscripts.core.result import Ok

            mock_validate.return_value = Ok(tmp_path)
            result = await search_codebase("nonexistent", ".")

        assert "No matches found" in result

    @pytest.mark.asyncio
    async def test_search_codebase_path_traversal_blocked(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.search import search_codebase

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
        ):
            from jpscripts.core.result import Err, SecurityError

            mock_validate.return_value = Err(SecurityError("Path escapes workspace"))
            result = await search_codebase("pattern", "../../../etc")

        assert "Error:" in result
        assert "escapes workspace" in result


class TestFindTodos:
    """Test find_todos MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        return runtime

    @pytest.mark.asyncio
    async def test_find_todos_success(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.search import find_todos

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
            patch(
                "jpscripts.mcp.tools.search.search_core.scan_todos",
                new_callable=AsyncMock,
            ) as mock_scan,
        ):
            from dataclasses import dataclass

            from jpscripts.core.result import Ok

            @dataclass
            class MockTodo:
                type: str
                file: str
                line: int
                text: str

            mock_validate.return_value = Ok(tmp_path)
            mock_scan.return_value = [MockTodo("TODO", "file.py", 10, "Fix this")]
            result = await find_todos(".")

        assert "TODO" in result
        assert "file.py" in result
        assert "Fix this" in result

    @pytest.mark.asyncio
    async def test_find_todos_empty(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.search import find_todos

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
            patch(
                "jpscripts.mcp.tools.search.search_core.scan_todos",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from jpscripts.core.result import Ok

            mock_validate.return_value = Ok(tmp_path)
            result = await find_todos(".")

        assert result == "[]"

    @pytest.mark.asyncio
    async def test_find_todos_path_validation_error(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.search import find_todos

        with (
            patch("jpscripts.mcp.tools.search.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.search.validate_path_async",
                new_callable=AsyncMock,
            ) as mock_validate,
        ):
            from jpscripts.core.result import Err, SecurityError

            mock_validate.return_value = Err(SecurityError("Blocked"))
            result = await find_todos("/etc/passwd")

        assert "Error:" in result
        assert "Blocked" in result


class TestListProcesses:
    """Test list_processes MCP tool."""

    @pytest.mark.asyncio
    async def test_list_processes_success(self) -> None:
        from jpscripts.mcp.tools.system import list_processes

        with patch(
            "jpscripts.mcp.tools.system.system_core.find_processes",
            new_callable=AsyncMock,
        ) as mock_find:
            from dataclasses import dataclass

            from jpscripts.core.result import Ok

            @dataclass
            class MockProcess:
                pid: int
                name: str
                username: str
                cmdline: str

            mock_find.return_value = Ok([MockProcess(123, "python", "user", "python test.py")])
            result = await list_processes()

        assert "123" in result
        assert "python" in result
        assert "user" in result

    @pytest.mark.asyncio
    async def test_list_processes_with_filter(self) -> None:
        from jpscripts.mcp.tools.system import list_processes

        with patch(
            "jpscripts.mcp.tools.system.system_core.find_processes",
            new_callable=AsyncMock,
        ) as mock_find:
            from jpscripts.core.result import Ok

            mock_find.return_value = Ok([])
            result = await list_processes(name_filter="python")

        assert "No matching processes found" in result
        mock_find.assert_called_once_with("python", None)

    @pytest.mark.asyncio
    async def test_list_processes_error(self) -> None:
        from jpscripts.mcp.tools.system import list_processes

        with patch(
            "jpscripts.mcp.tools.system.system_core.find_processes",
            new_callable=AsyncMock,
        ) as mock_find:
            from jpscripts.core.result import Err, SystemResourceError

            mock_find.return_value = Err(SystemResourceError("Permission denied"))
            result = await list_processes()

        assert "Error listing processes" in result
        assert "Permission denied" in result

    @pytest.mark.asyncio
    async def test_list_processes_truncates_long_list(self) -> None:
        from jpscripts.mcp.tools.system import list_processes

        with patch(
            "jpscripts.mcp.tools.system.system_core.find_processes",
            new_callable=AsyncMock,
        ) as mock_find:
            from dataclasses import dataclass

            from jpscripts.core.result import Ok

            @dataclass
            class MockProcess:
                pid: int
                name: str
                username: str
                cmdline: str

            # Create 60 processes to trigger truncation
            procs = [MockProcess(i, f"proc{i}", "user", "cmd") for i in range(60)]
            mock_find.return_value = Ok(procs)
            result = await list_processes()

        assert "... and 10 more" in result


class TestKillProcess:
    """Test kill_process MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        return runtime

    @pytest.mark.asyncio
    async def test_kill_process_success(self, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.system import kill_process

        with (
            patch("jpscripts.mcp.tools.system.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.system.system_core.kill_process_async",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            from jpscripts.core.result import Ok

            mock_kill.return_value = Ok("terminated")
            result = await kill_process(123)

        assert "Process 123" in result
        assert "terminated" in result

    @pytest.mark.asyncio
    async def test_kill_process_error(self, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.system import kill_process

        with (
            patch("jpscripts.mcp.tools.system.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.system.system_core.kill_process_async",
                new_callable=AsyncMock,
            ) as mock_kill,
        ):
            from jpscripts.core.result import Err, SystemResourceError

            mock_kill.return_value = Err(SystemResourceError("No such process"))
            result = await kill_process(99999)

        assert "Error killing process" in result
        assert "No such process" in result


class TestRunShell:
    """Test run_shell MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        return runtime

    @pytest.mark.asyncio
    async def test_run_shell_success(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.system import run_shell

        with (
            patch("jpscripts.mcp.tools.system.get_runtime", return_value=mock_runtime),
            patch(
                "jpscripts.mcp.tools.system.run_safe_shell",
                new_callable=AsyncMock,
                return_value="output: success",
            ) as mock_run,
        ):
            result = await run_shell("ls -la")

        assert "success" in result
        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_shell_empty_command(self, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.system import run_shell

        with patch("jpscripts.mcp.tools.system.get_runtime", return_value=mock_runtime):
            result = await run_shell("   ")

        assert "Invalid command argument" in result

    @pytest.mark.asyncio
    async def test_run_shell_whitespace_command(self, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.system import run_shell

        with patch("jpscripts.mcp.tools.system.get_runtime", return_value=mock_runtime):
            result = await run_shell("")

        assert "Invalid command argument" in result
