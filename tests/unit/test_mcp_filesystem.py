"""
Unit tests for MCP filesystem tools.

Tests cover read_file, write_file, list_directory, and apply_patch
including error paths, rate limiting, and security validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jpscripts.core.config import AppConfig


class TestReadFile:
    """Test read_file MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        """Create a mock runtime context."""
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        runtime.config.max_file_context_chars = 50000
        runtime.dry_run = False
        return runtime

    @pytest.mark.asyncio
    async def test_read_file_success(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import read_file

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file(str(test_file))

        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import read_file

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file(str(tmp_path / "nonexistent.txt"))

        assert "Error:" in result
        assert "does not exist" in result

    @pytest.mark.asyncio
    async def test_read_file_directory_error(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import read_file

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file(str(tmp_path))

        assert "Error:" in result
        assert "not a file" in result

    @pytest.mark.asyncio
    async def test_read_file_rate_limited(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import read_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch(
                    "jpscripts.mcp.tools.filesystem._file_rate_limiter.time_until_available",
                    return_value=5.0,
                ):
                    result = await read_file(str(test_file))

        assert "Rate limit exceeded" in result


class TestWriteFile:
    """Test write_file MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        runtime.dry_run = False
        return runtime

    @pytest.mark.asyncio
    async def test_write_file_new(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import write_file

        test_file = tmp_path / "new_file.txt"

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await write_file(str(test_file), "New content")

        assert "Successfully wrote" in result
        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_file_overwrite_required(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.filesystem import write_file

        test_file = tmp_path / "existing.txt"
        test_file.write_text("Original")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await write_file(str(test_file), "New content", overwrite=False)

        assert "Error:" in result
        assert "already exists" in result
        assert test_file.read_text() == "Original"

    @pytest.mark.asyncio
    async def test_write_file_overwrite_allowed(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.filesystem import write_file

        test_file = tmp_path / "existing.txt"
        test_file.write_text("Original")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await write_file(str(test_file), "New content", overwrite=True)

        assert "Successfully wrote" in result
        assert test_file.read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_file_dry_run(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import write_file

        mock_runtime.dry_run = True
        test_file = tmp_path / "dry_run.txt"

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await write_file(str(test_file), "Content")

        assert "Simulated write" in result
        assert "dry-run" in result
        assert not test_file.exists()


class TestListDirectory:
    """Test list_directory MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        return runtime

    @pytest.mark.asyncio
    async def test_list_directory_success(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import list_directory

        # Create some test files and dirs
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("content")
        (tmp_path / "subdir").mkdir()

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await list_directory(str(tmp_path))

        assert "f: file1.txt" in result
        assert "f: file2.py" in result
        assert "d: subdir" in result

    @pytest.mark.asyncio
    async def test_list_directory_not_found(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import list_directory

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await list_directory(str(tmp_path / "nonexistent"))

        assert "Error:" in result
        assert "does not exist" in result

    @pytest.mark.asyncio
    async def test_list_directory_file_error(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import list_directory

        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await list_directory(str(test_file))

        assert "Error:" in result
        assert "not a directory" in result


class TestReadFilePaged:
    """Test read_file_paged MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        return runtime

    @pytest.mark.asyncio
    async def test_read_file_paged_offset(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import read_file_paged

        test_file = tmp_path / "large.txt"
        test_file.write_text("0123456789" * 100)

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file_paged(str(test_file), offset=10, limit=20)

        assert len(result) == 20
        assert result.startswith("0123456789")

    @pytest.mark.asyncio
    async def test_read_file_paged_invalid_offset(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.filesystem import read_file_paged

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file_paged(str(test_file), offset=-1)

        assert "Error:" in result
        assert "offset must be non-negative" in result

    @pytest.mark.asyncio
    async def test_read_file_paged_invalid_limit(
        self, tmp_path: Path, mock_runtime: MagicMock
    ) -> None:
        from jpscripts.mcp.tools.filesystem import read_file_paged

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await read_file_paged(str(test_file), limit=0)

        assert "Error:" in result
        assert "limit must be positive" in result


class TestApplyPatch:
    """Test apply_patch MCP tool."""

    @pytest.fixture
    def mock_runtime(self, tmp_path: Path) -> MagicMock:
        runtime = MagicMock()
        runtime.workspace_root = tmp_path
        runtime.config = AppConfig(workspace_root=tmp_path)
        runtime.dry_run = False
        return runtime

    @pytest.mark.asyncio
    async def test_apply_patch_empty(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import apply_patch

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await apply_patch(str(test_file), "   ")

        assert "Error" in result or "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_apply_patch_to_directory(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import apply_patch

        diff = """--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old
+new"""

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await apply_patch(str(tmp_path), diff)

        assert "Error" in result or "directory" in result.lower()

    @pytest.mark.asyncio
    async def test_apply_patch_rate_limited(self, tmp_path: Path, mock_runtime: MagicMock) -> None:
        from jpscripts.mcp.tools.filesystem import apply_patch

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with patch("jpscripts.mcp.tools.filesystem.get_runtime", return_value=mock_runtime):
            with patch(
                "jpscripts.mcp.tools.filesystem._file_rate_limiter.acquire",
                new_callable=AsyncMock,
                return_value=False,
            ):
                with patch(
                    "jpscripts.mcp.tools.filesystem._file_rate_limiter.time_until_available",
                    return_value=5.0,
                ):
                    result = await apply_patch(str(test_file), "diff content")

        assert "Rate limit exceeded" in result


class TestPatchHelpers:
    """Test patch utility functions."""

    def test_normalize_patch_path(self) -> None:
        from jpscripts.mcp.tools.filesystem import _normalize_patch_path

        assert _normalize_patch_path("a/foo/bar.py") == "foo/bar.py"
        assert _normalize_patch_path("b/foo/bar.py") == "foo/bar.py"
        assert _normalize_patch_path("foo/bar.py") == "foo/bar.py"
        assert _normalize_patch_path("  a/test.py  ") == "test.py"

    def test_extract_patch_targets(self) -> None:
        from jpscripts.mcp.tools.filesystem import _extract_patch_targets

        diff = """--- a/src/foo.py
+++ b/src/foo.py
@@ -1 +1 @@
-old
+new"""
        targets = _extract_patch_targets(diff)
        assert "src/foo.py" in targets

    def test_extract_patch_targets_dev_null(self) -> None:
        from jpscripts.mcp.tools.filesystem import _extract_patch_targets

        diff = """--- /dev/null
+++ b/new_file.py
@@ -0,0 +1 @@
+new content"""
        targets = _extract_patch_targets(diff)
        assert "/dev/null" not in targets
        assert "new_file.py" in targets

    def test_detect_strip_level(self) -> None:
        from jpscripts.mcp.tools.filesystem import _detect_strip_level

        diff_with_prefix = """--- a/file.py
+++ b/file.py"""
        assert _detect_strip_level(diff_with_prefix) == 1

        diff_without_prefix = """--- file.py
+++ file.py"""
        assert _detect_strip_level(diff_without_prefix) == 0
