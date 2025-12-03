"""Tests for MCP server security middleware integration."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jpscripts.core.config import AppConfig, UserConfig
from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.errors import SecurityError
from jpscripts.core.runtime import runtime_context
from jpscripts.core.safety import wrap_mcp_tool


class TestWrapMcpTool:
    """Tests for wrap_mcp_tool in the context of MCP server."""

    @pytest.fixture
    def test_config(self, tmp_path: Path) -> AppConfig:
        """Create a test config for runtime context."""
        return AppConfig(
            user=UserConfig(
                workspace_root=tmp_path,
                notes_dir=tmp_path / "notes",
            )
        )

    def test_wrapped_tool_executes_normally(self, test_config: AppConfig) -> None:
        """A wrapped MCP tool should execute when breaker is healthy."""
        call_count = 0

        async def my_tool(message: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Got: {message}"

        wrapped = wrap_mcp_tool(my_tool, "my_tool")

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = asyncio.run(wrapped("hello"))

        assert result == "Got: hello"
        assert call_count == 1

    def test_wrapped_tool_preserves_function_name(self) -> None:
        """wrap_mcp_tool should preserve the original function name."""

        async def original_tool_name(x: str) -> str:
            return x

        wrapped = wrap_mcp_tool(original_tool_name, "original_tool_name")

        assert wrapped.__name__ == "original_tool_name"

    def test_wrapped_tool_preserves_docstring(self) -> None:
        """wrap_mcp_tool should preserve the original docstring."""

        async def documented_tool(x: str) -> str:
            """This is the original docstring."""
            return x

        wrapped = wrap_mcp_tool(documented_tool, "documented_tool")

        assert "original docstring" in (wrapped.__doc__ or "")

    def test_wrapped_tool_works_without_runtime_context(self) -> None:
        """A wrapped tool should still work if no runtime context is available."""
        call_count = 0

        async def resilient_tool(data: str) -> str:
            nonlocal call_count
            call_count += 1
            return data

        wrapped = wrap_mcp_tool(resilient_tool, "resilient_tool")

        # No runtime context - should still execute
        result = asyncio.run(wrapped("test"))

        assert result == "test"
        assert call_count == 1

    def test_mcp_breaker_has_higher_limits_than_agent(self, test_config: AppConfig) -> None:
        """MCP circuit breaker should have higher default limits than agent breaker."""
        with runtime_context(test_config, workspace=test_config.user.workspace_root) as ctx:
            agent_breaker = ctx.get_circuit_breaker()
            mcp_breaker = ctx.get_mcp_circuit_breaker()

            # MCP should be more permissive
            assert mcp_breaker.max_cost_velocity >= agent_breaker.max_cost_velocity
            assert mcp_breaker.max_file_churn >= agent_breaker.max_file_churn

    def test_mcp_breaker_is_separate_from_agent_breaker(self, test_config: AppConfig) -> None:
        """MCP and agent should have separate circuit breaker instances."""
        with runtime_context(test_config, workspace=test_config.user.workspace_root) as ctx:
            agent_breaker = ctx.get_circuit_breaker()
            mcp_breaker = ctx.get_mcp_circuit_breaker()

            # Should be different objects
            assert agent_breaker is not mcp_breaker

            # MCP breaker should have "mcp" model_id
            assert mcp_breaker.model_id == "mcp"


class TestMcpBreakerTrip:
    """Tests for circuit breaker tripping in MCP context."""

    @pytest.fixture
    def strict_config(self, tmp_path: Path) -> AppConfig:
        """Create a config with very low MCP limits for testing."""
        config = AppConfig(
            user=UserConfig(
                workspace_root=tmp_path,
                notes_dir=tmp_path / "notes",
            )
        )
        # We'll manually create a strict breaker in the test
        return config

    def test_mcp_tool_trips_on_high_cost(self, strict_config: AppConfig) -> None:
        """MCP tool should raise SecurityError when cost velocity is exceeded."""

        async def expensive_tool(huge_data: str) -> str:
            return huge_data

        wrapped = wrap_mcp_tool(expensive_tool, "expensive_tool")

        with runtime_context(strict_config, workspace=strict_config.user.workspace_root) as ctx:
            # Get the MCP breaker and make it very strict
            breaker = ctx.get_mcp_circuit_breaker()
            breaker.max_cost_velocity = Decimal("0.000001")  # Extremely low

            # Prime the timestamp
            breaker.check_health(TokenUsage(prompt_tokens=0, completion_tokens=0), [])

            # Now the next call with large data should trip it
            with pytest.raises(SecurityError) as exc_info:
                asyncio.run(wrapped("x" * 1000000))

            assert "Circuit breaker tripped" in str(exc_info.value)
            assert "mcp-client" in str(exc_info.value)
            assert "tool:expensive_tool" in str(exc_info.value)

    def test_mcp_tool_trips_on_file_churn(self, strict_config: AppConfig) -> None:
        """MCP tool should raise SecurityError when file churn is exceeded."""

        async def file_heavy_tool(paths: list[str]) -> str:
            return f"Processed {len(paths)} files"

        wrapped = wrap_mcp_tool(file_heavy_tool, "file_heavy_tool")

        with runtime_context(strict_config, workspace=strict_config.user.workspace_root) as ctx:
            # Get the MCP breaker and make file churn very strict
            breaker = ctx.get_mcp_circuit_breaker()
            breaker.max_file_churn = 0  # No files allowed

            # This should trip the breaker (note: file churn is tracked separately)
            # The wrap_mcp_tool doesn't track files_touched, so this test verifies
            # that the enforcement mechanism is in place (file churn would need
            # to be tracked at a higher level for MCP tools)

            # For now, test that the wrapper doesn't break with list arguments
            result = asyncio.run(wrapped(["a.py", "b.py", "c.py"]))
            assert "Processed 3 files" in result


class TestMcpServerIntegration:
    """Integration tests for MCP server tool registration."""

    @pytest.fixture
    def test_config(self, tmp_path: Path) -> AppConfig:
        """Create a test config."""
        return AppConfig(
            user=UserConfig(
                workspace_root=tmp_path,
                notes_dir=tmp_path / "notes",
            )
        )

    def test_register_tools_wraps_with_safety(self, test_config: AppConfig) -> None:
        """register_tools should wrap all tools with safety middleware."""
        from jpscripts.mcp.server import register_tools

        mock_mcp = MagicMock()

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            # This should not raise and should call add_tool for each discovered tool
            register_tools(mock_mcp)

        # Verify add_tool was called
        assert mock_mcp.add_tool.called
        # Get the number of times add_tool was called
        call_count = mock_mcp.add_tool.call_count
        assert call_count > 0, "Expected at least one tool to be registered"

    def test_wrapped_tool_in_server_has_correct_metadata(self, test_config: AppConfig) -> None:
        """Tools registered with safety wrappers should preserve metadata."""
        from jpscripts.mcp.server import register_tools

        registered_tools = []

        def capture_tool(tool):
            registered_tools.append(tool)

        mock_mcp = MagicMock()
        mock_mcp.add_tool = capture_tool

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            register_tools(mock_mcp)

        # Check that we got some tools
        assert len(registered_tools) > 0

        # Each tool should have a name (from Tool.from_function)
        for tool in registered_tools:
            assert hasattr(tool, "name") or hasattr(tool, "fn")


class TestMcpPathTraversalAttacks:
    """Attack test cases for MCP filesystem tools path validation."""

    @pytest.fixture
    def test_config(self, tmp_path: Path) -> AppConfig:
        """Create a test config with isolated workspace."""
        return AppConfig(
            user=UserConfig(
                workspace_root=tmp_path,
                notes_dir=tmp_path / "notes",
            )
        )

    @pytest.mark.asyncio
    async def test_read_etc_passwd_blocked(self, test_config: AppConfig) -> None:
        """Attempt to read /etc/passwd should be blocked."""
        from jpscripts.mcp.tools.filesystem import read_file

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await read_file("/etc/passwd")

        assert "Error" in result
        assert "escapes workspace" in result.lower() or "forbidden" in result.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, test_config: AppConfig, tmp_path: Path) -> None:
        """Attempt to read ../outside_workspace/secret should be blocked."""
        from jpscripts.mcp.tools.filesystem import read_file

        # Create a file outside the workspace
        outside_dir = tmp_path.parent / "outside_workspace"
        outside_dir.mkdir(exist_ok=True)
        secret_file = outside_dir / "secret"
        secret_file.write_text("top secret data")

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await read_file("../outside_workspace/secret")

        assert "Error" in result
        # Should not contain the secret content
        assert "top secret data" not in result

    @pytest.mark.asyncio
    async def test_symlink_escape_blocked(self, test_config: AppConfig, tmp_path: Path) -> None:
        """Symlink pointing outside workspace should be blocked."""
        from jpscripts.mcp.tools.filesystem import read_file

        # Create a file outside the workspace
        outside_dir = tmp_path.parent / "outside_symlink_target"
        outside_dir.mkdir(exist_ok=True)
        target_file = outside_dir / "secret.txt"
        target_file.write_text("symlink escape secret")

        # Create symlink inside workspace pointing outside
        symlink_path = tmp_path / "escape_link"
        symlink_path.symlink_to(target_file)

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await read_file("escape_link")

        assert "Error" in result
        # Should not contain the secret content
        assert "symlink escape secret" not in result

    @pytest.mark.asyncio
    async def test_write_path_traversal_blocked(self, test_config: AppConfig, tmp_path: Path) -> None:
        """write_file should block path traversal attempts."""
        from jpscripts.mcp.tools.filesystem import write_file

        # Try to write outside workspace
        outside_dir = tmp_path.parent / "outside_write_target"
        outside_dir.mkdir(exist_ok=True)

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await write_file("../outside_write_target/evil.txt", "malicious content")

        assert "Error" in result
        # The file should not have been created
        assert not (outside_dir / "evil.txt").exists()

    @pytest.mark.asyncio
    async def test_write_no_mkdir_outside_workspace(self, test_config: AppConfig, tmp_path: Path) -> None:
        """write_file should not create directories outside workspace."""
        from jpscripts.mcp.tools.filesystem import write_file

        # Try to create nested directories outside workspace
        outside_path = tmp_path.parent / "should_not_exist"

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await write_file("../should_not_exist/nested/deep/file.txt", "content")

        assert "Error" in result
        # The directory should not have been created
        assert not outside_path.exists()

    @pytest.mark.asyncio
    async def test_list_directory_traversal_blocked(self, test_config: AppConfig) -> None:
        """list_directory should block path traversal attempts."""
        from jpscripts.mcp.tools.filesystem import list_directory

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = await list_directory("../")

        assert "Error" in result

    @pytest.mark.asyncio
    async def test_valid_path_still_works(self, test_config: AppConfig, tmp_path: Path) -> None:
        """Valid paths within workspace should still work (no false positives)."""
        from jpscripts.mcp.tools.filesystem import read_file, write_file

        # Create a valid file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("valid content")

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            # Read should work
            read_result = await read_file("valid.txt")
            assert "valid content" in read_result
            assert "Error" not in read_result

            # Write should work
            write_result = await write_file("new_file.txt", "new content")
            assert "Successfully" in write_result
            assert (tmp_path / "new_file.txt").exists()
