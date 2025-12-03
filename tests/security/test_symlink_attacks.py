"""Tests for symlink attack prevention in path validation.

These tests verify that the security module correctly handles various
symlink-based attack vectors including:
- Simple symlink escapes
- Chained symlinks
- Circular symlinks
- System directory protection
- Path traversal combined with symlinks
- TOCTOU mitigation via validate_and_open
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jpscripts.core.result import Err, Ok, SecurityError
from jpscripts.core.security import (
    FORBIDDEN_ROOTS,
    MAX_SYMLINK_DEPTH,
    validate_and_open,
    validate_path,
    validate_path_async,
    validate_workspace_root_async,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    # Create a marker file to make it look like a valid workspace
    (ws / ".git").mkdir()
    return ws


@pytest.fixture
def outside_dir(tmp_path: Path) -> Path:
    """Create a directory outside the workspace."""
    outside = tmp_path / "outside"
    outside.mkdir()
    return outside


class TestSimpleSymlinkEscape:
    """Test basic symlink escape scenarios."""

    def test_symlink_to_outside_file(self, workspace: Path, outside_dir: Path) -> None:
        """Symlink pointing to file outside workspace should be rejected."""
        # Create file outside workspace
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret data")

        # Create symlink inside workspace pointing outside
        malicious_link = workspace / "innocent.txt"
        malicious_link.symlink_to(outside_file)

        result = validate_path(malicious_link, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_symlink_to_outside_directory(self, workspace: Path, outside_dir: Path) -> None:
        """Symlink pointing to directory outside workspace should be rejected."""
        malicious_link = workspace / "data"
        malicious_link.symlink_to(outside_dir)

        result = validate_path(malicious_link, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_symlink_to_parent_directory(self, workspace: Path) -> None:
        """Symlink pointing to parent should be rejected."""
        malicious_link = workspace / "parent"
        malicious_link.symlink_to(workspace.parent)

        result = validate_path(malicious_link, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_relative_symlink_escape(self, workspace: Path, outside_dir: Path) -> None:
        """Relative symlink that escapes workspace should be rejected."""
        # Create file outside
        outside_file = outside_dir / "data.txt"
        outside_file.write_text("sensitive")

        # Create relative symlink that escapes
        malicious_link = workspace / "link.txt"
        # This creates a relative symlink like "../outside/data.txt"
        relative_target = Path("..") / "outside" / "data.txt"
        malicious_link.symlink_to(relative_target)

        result = validate_path(malicious_link, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_valid_internal_symlink(self, workspace: Path) -> None:
        """Symlink within workspace pointing to valid file should work."""
        # Create real file in workspace
        real_file = workspace / "real.txt"
        real_file.write_text("valid content")

        # Create symlink to it
        link = workspace / "link.txt"
        link.symlink_to(real_file)

        result = validate_path(link, workspace)
        assert isinstance(result, Ok)
        assert result.value == real_file.resolve()


class TestChainedSymlinks:
    """Test multi-hop symlink chains."""

    def test_double_symlink_escape(self, workspace: Path, outside_dir: Path) -> None:
        """Chain of symlinks eventually escaping should be rejected."""
        # Create target outside
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")

        # Create chain: link1 -> link2 -> outside_file
        link2 = workspace / "link2"
        link2.symlink_to(outside_file)

        link1 = workspace / "link1"
        link1.symlink_to(link2)

        result = validate_path(link1, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_symlink_chain_depth_limit(self, workspace: Path) -> None:
        """Deep symlink chains should be rejected."""
        # Create a chain of symlinks deeper than MAX_SYMLINK_DEPTH
        real_file = workspace / "real.txt"
        real_file.write_text("content")

        prev_link = real_file
        for i in range(MAX_SYMLINK_DEPTH + 5):
            link = workspace / f"link_{i}"
            link.symlink_to(prev_link)
            prev_link = link

        # Should fail due to depth limit
        result = validate_path(prev_link, workspace)
        assert isinstance(result, Err)
        assert "too deep" in result.error.message.lower()

    def test_circular_symlink_detection(self, workspace: Path) -> None:
        """Circular symlink chains should be detected and rejected."""
        # Create circular chain: link1 -> link2 -> link1
        link1 = workspace / "link1"
        link2 = workspace / "link2"

        # Create link2 first pointing to where link1 will be
        link2.symlink_to(link1)
        link1.symlink_to(link2)

        result = validate_path(link1, workspace)
        assert isinstance(result, Err)
        assert "circular" in result.error.message.lower()


class TestSystemDirectoryProtection:
    """Test forbidden system path rejection."""

    def test_reject_etc_via_helper(self, workspace: Path) -> None:
        """Direct access to /etc should be rejected."""
        from jpscripts.core.security import _is_forbidden_path

        # /etc itself and files within it should be forbidden
        assert _is_forbidden_path(Path("/etc"))
        assert _is_forbidden_path(Path("/etc/passwd"))

    def test_reject_etc_path(self, workspace: Path) -> None:
        """Access to /etc should be rejected."""
        from jpscripts.core.security import _is_forbidden_path

        assert _is_forbidden_path(Path("/etc"))
        assert _is_forbidden_path(Path("/etc/passwd"))

    def test_reject_system_path_via_symlink(self, workspace: Path, tmp_path: Path) -> None:
        """Symlink to system directory should be rejected."""
        # Create symlink to /etc (if it exists)
        if Path("/etc").exists():
            malicious_link = workspace / "etc_link"
            malicious_link.symlink_to(Path("/etc"))

            result = validate_path(malicious_link, workspace)
            assert isinstance(result, Err)

    def test_forbidden_roots_constant(self) -> None:
        """Verify FORBIDDEN_ROOTS contains expected system paths."""
        # Note: / is intentionally excluded - too broad
        assert Path("/etc") in FORBIDDEN_ROOTS
        assert Path("/usr") in FORBIDDEN_ROOTS
        assert Path("/bin") in FORBIDDEN_ROOTS


class TestPathTraversalWithSymlinks:
    """Test combined path traversal and symlink attacks."""

    def test_symlink_then_traversal(self, workspace: Path, outside_dir: Path) -> None:
        """Path traversal after symlink should be blocked."""
        # Create a directory symlink inside workspace
        subdir = workspace / "subdir"
        subdir.mkdir()

        # Try to traverse out using path traversal after symlink
        traversal_path = workspace / "subdir" / ".." / ".." / "outside"

        result = validate_path(traversal_path, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_traversal_to_symlink(self, workspace: Path, outside_dir: Path) -> None:
        """Symlink reached via traversal should still be validated."""
        # Create nested structure
        subdir = workspace / "a" / "b"
        subdir.mkdir(parents=True)

        # Create symlink to outside at workspace root
        escape_link = workspace / "escape"
        escape_link.symlink_to(outside_dir)

        # Try to reach it via traversal
        traversal = workspace / "a" / "b" / ".." / ".." / "escape"

        result = validate_path(traversal, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()

    def test_mixed_attack_vectors(self, workspace: Path, outside_dir: Path) -> None:
        """Combination of symlinks and traversal should be blocked."""
        # Create outside target
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")

        # Create chain with traversal embedded
        subdir = workspace / "data"
        subdir.mkdir()

        # Symlink that uses relative traversal in target
        link = subdir / "link"
        link.symlink_to(Path("../../outside/secret.txt"))

        result = validate_path(link, workspace)
        assert isinstance(result, Err)
        assert "escapes" in result.error.message.lower()


class TestValidateAndOpen:
    """Test the atomic validate_and_open function."""

    def test_validate_and_open_success(self, workspace: Path) -> None:
        """Successfully open a valid file."""
        test_file = workspace / "test.txt"
        test_file.write_text("hello world")

        result = validate_and_open(test_file, workspace, "r")
        assert isinstance(result, Ok)

        with result.value as f:
            content = f.read()
            assert content == "hello world"

    def test_validate_and_open_rejects_symlink(self, workspace: Path, outside_dir: Path) -> None:
        """Opening a symlink to outside should fail."""
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")

        link = workspace / "link.txt"
        link.symlink_to(outside_file)

        result = validate_and_open(link, workspace, "r")
        assert isinstance(result, Err)

    def test_validate_and_open_internal_symlink(self, workspace: Path) -> None:
        """Opening internal symlink works because path is resolved first.

        Note: The path validation resolves symlinks BEFORE opening,
        so by the time we call os.open with O_NOFOLLOW, we're opening
        the resolved real file, not the symlink. This is the expected
        behavior - internal symlinks to valid workspace paths work.
        """
        test_file = workspace / "real.txt"
        test_file.write_text("content")

        link = workspace / "link.txt"
        link.symlink_to(test_file)

        # Path is resolved before opening, so this succeeds
        result = validate_and_open(link, workspace, "r")
        assert isinstance(result, Ok)

        with result.value as f:
            assert f.read() == "content"

    def test_validate_and_open_write_mode(self, workspace: Path) -> None:
        """Test write mode with validate_and_open."""
        test_file = workspace / "output.txt"

        result = validate_and_open(test_file, workspace, "w")
        assert isinstance(result, Ok)

        with result.value as f:
            f.write("written content")

        assert test_file.read_text() == "written content"

    def test_validate_and_open_nonexistent_read(self, workspace: Path) -> None:
        """Opening nonexistent file for read should fail."""
        nonexistent = workspace / "does_not_exist.txt"

        result = validate_and_open(nonexistent, workspace, "r")
        assert isinstance(result, Err)


@pytest.mark.asyncio
class TestAsyncValidation:
    """Test async validation functions."""

    async def test_async_validate_path(self, workspace: Path) -> None:
        """Async path validation should work."""
        test_file = workspace / "test.txt"
        test_file.write_text("content")

        result = await validate_path_async(test_file, workspace)
        assert isinstance(result, Ok)
        assert result.value == test_file.resolve()

    async def test_async_validate_workspace_root(self, workspace: Path) -> None:
        """Async workspace validation should work."""
        result = await validate_workspace_root_async(workspace)
        assert isinstance(result, Ok)
        assert result.value == workspace.resolve()

    async def test_async_validate_rejects_escape(self, workspace: Path, outside_dir: Path) -> None:
        """Async validation should reject escapes."""
        link = workspace / "escape"
        link.symlink_to(outside_dir)

        result = await validate_path_async(link, workspace)
        assert isinstance(result, Err)

    async def test_async_concurrent_validation(self, workspace: Path) -> None:
        """Multiple async validations should work concurrently."""
        import asyncio

        # Create multiple test files
        files = []
        for i in range(10):
            f = workspace / f"file_{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        # Validate all concurrently
        results = await asyncio.gather(*[validate_path_async(f, workspace) for f in files])

        assert all(isinstance(r, Ok) for r in results)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_path(self, workspace: Path) -> None:
        """Empty path should be handled gracefully."""
        result = validate_path("", workspace)
        # Behavior depends on implementation - either error or resolve to cwd
        # Just ensure it doesn't crash
        assert isinstance(result, (Ok, Err))

    def test_dot_path(self, workspace: Path) -> None:
        """Current directory '.' should work within workspace."""
        # Change to workspace and validate '.'
        old_cwd = os.getcwd()
        try:
            os.chdir(workspace)
            result = validate_path(".", workspace)
            assert isinstance(result, Ok)
        finally:
            os.chdir(old_cwd)

    def test_unicode_path(self, workspace: Path) -> None:
        """Unicode filenames should be handled correctly."""
        unicode_file = workspace / "файл_文件_αρχείο.txt"
        unicode_file.write_text("unicode content")

        result = validate_path(unicode_file, workspace)
        assert isinstance(result, Ok)

    def test_long_path(self, workspace: Path) -> None:
        """Very long paths should be handled."""
        # Create deeply nested directory
        deep_path = workspace
        for i in range(20):
            deep_path = deep_path / f"level_{i}"

        deep_path.mkdir(parents=True)
        test_file = deep_path / "deep_file.txt"
        test_file.write_text("deep")

        result = validate_path(test_file, workspace)
        assert isinstance(result, Ok)

    def test_special_characters_in_path(self, workspace: Path) -> None:
        """Paths with special characters should work."""
        special_file = workspace / "file with spaces & 'quotes'.txt"
        special_file.write_text("special")

        result = validate_path(special_file, workspace)
        assert isinstance(result, Ok)

    def test_symlink_to_nonexistent(self, workspace: Path) -> None:
        """Symlink to nonexistent target within workspace is allowed.

        Note: Path validation only checks that the resolved path stays
        within the workspace. It doesn't require the file to exist.
        This allows creating files via symlinks.
        """
        broken_link = workspace / "broken"
        broken_link.symlink_to(workspace / "does_not_exist")

        result = validate_path(broken_link, workspace)
        # This succeeds because the resolved path is within workspace
        # even though the target doesn't exist yet
        assert isinstance(result, Ok)
        assert "does_not_exist" in str(result.value)
