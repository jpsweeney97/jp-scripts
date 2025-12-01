"""Extended security tests for comprehensive coverage of security.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from jpscripts.core.result import Err, Ok
from jpscripts.core.security import (
    MAX_SYMLINK_DEPTH,
    PathValidationError,
    WorkspaceValidationError,
    _is_forbidden_path,
    _resolve_with_limit,
    is_path_safe,
    validate_path,
    validate_path_safe,
    validate_path_safe_async,
    validate_workspace_root,
    validate_workspace_root_safe,
)


class TestForbiddenPath:
    """Tests for _is_forbidden_path function."""

    def test_etc_is_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/etc")) is True

    def test_etc_child_is_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/etc/passwd")) is True

    def test_usr_is_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/usr")) is True

    def test_bin_is_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/bin")) is True

    def test_root_is_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/root")) is True

    def test_normal_path_is_not_forbidden(self, tmp_path: Path) -> None:
        assert _is_forbidden_path(tmp_path) is False

    def test_home_is_not_forbidden(self) -> None:
        assert _is_forbidden_path(Path.home()) is False

    def test_tmp_is_not_forbidden(self) -> None:
        assert _is_forbidden_path(Path("/tmp")) is False


class TestResolveWithLimit:
    """Tests for _resolve_with_limit function."""

    def test_regular_file_resolves(self, tmp_path: Path) -> None:
        regular_file = tmp_path / "file.txt"
        regular_file.write_text("content", encoding="utf-8")

        result = _resolve_with_limit(regular_file)
        assert isinstance(result, Ok)
        assert result.value == regular_file.resolve()

    def test_single_symlink_resolves(self, tmp_path: Path) -> None:
        target = tmp_path / "target.txt"
        target.write_text("content", encoding="utf-8")

        link = tmp_path / "link.txt"
        link.symlink_to(target)

        result = _resolve_with_limit(link)
        assert isinstance(result, Ok)
        assert result.value == target.resolve()

    def test_deep_symlink_chain_rejected(self, tmp_path: Path) -> None:
        """Create a chain deeper than MAX_SYMLINK_DEPTH."""
        # Create a chain of symlinks exceeding the limit
        target = tmp_path / "target.txt"
        target.write_text("content", encoding="utf-8")

        current = target
        for i in range(MAX_SYMLINK_DEPTH + 2):
            link = tmp_path / f"link_{i}.txt"
            link.symlink_to(current)
            current = link

        result = _resolve_with_limit(current)
        assert isinstance(result, Err)
        assert "too deep" in str(result.error).lower()

    def test_circular_symlink_detected(self, tmp_path: Path) -> None:
        """Test circular symlink detection."""
        link_a = tmp_path / "link_a"
        link_b = tmp_path / "link_b"

        # Create circular reference
        link_a.symlink_to(link_b)
        link_b.symlink_to(link_a)

        result = _resolve_with_limit(link_a)
        assert isinstance(result, Err)
        assert "circular" in str(result.error).lower()

    def test_nonexistent_path_resolves(self, tmp_path: Path) -> None:
        """Non-existent paths can still resolve (for path validation)."""
        nonexistent = tmp_path / "does_not_exist.txt"
        result = _resolve_with_limit(nonexistent)
        # Non-existent paths resolve to their canonical form
        assert isinstance(result, Ok)


class TestValidatePathSafe:
    """Tests for validate_path_safe function (Result-based API)."""

    def test_valid_path_in_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file.txt"
        target.write_text("content", encoding="utf-8")

        result = validate_path_safe(target, workspace)
        assert isinstance(result, Ok)
        assert result.value == target.resolve()

    def test_relative_path_in_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file.txt"
        target.write_text("content", encoding="utf-8")

        # Relative paths are joined with workspace root
        result = validate_path_safe(workspace / "file.txt", workspace)
        assert isinstance(result, Ok)

    def test_traversal_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_path_safe("../../../etc/passwd", workspace)
        assert isinstance(result, Err)

    def test_symlink_escape_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        malicious = workspace / "escape.txt"
        malicious.symlink_to(outside)

        result = validate_path_safe(malicious, workspace)
        assert isinstance(result, Err)

    def test_absolute_path_outside_workspace_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_path_safe("/etc/passwd", workspace)
        assert isinstance(result, Err)


class TestValidatePathSafeAsync:
    """Tests for async path validation."""

    @pytest.mark.asyncio
    async def test_valid_path_in_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file.txt"
        target.write_text("content", encoding="utf-8")

        result = await validate_path_safe_async(target, workspace)
        assert isinstance(result, Ok)
        assert result.value == target.resolve()

    @pytest.mark.asyncio
    async def test_traversal_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = await validate_path_safe_async("../../../etc/passwd", workspace)
        assert isinstance(result, Err)


class TestIsPathSafe:
    """Tests for is_path_safe boolean helper."""

    def test_valid_path_returns_true(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file.txt"
        target.write_text("content", encoding="utf-8")

        assert is_path_safe(target, workspace) is True

    def test_traversal_returns_false(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        assert is_path_safe("../../../etc/passwd", workspace) is False

    def test_symlink_escape_returns_false(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        malicious = workspace / "escape.txt"
        malicious.symlink_to(outside)

        assert is_path_safe(malicious, workspace) is False


class TestValidateWorkspaceRoot:
    """Tests for workspace root validation."""

    def test_valid_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_workspace_root(workspace)
        assert result == workspace.resolve()

    def test_missing_workspace_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"

        with pytest.raises(WorkspaceValidationError):
            validate_workspace_root(missing)

    def test_file_as_workspace_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(WorkspaceValidationError):
            validate_workspace_root(file_path)


class TestValidateWorkspaceRootSafe:
    """Tests for Result-based workspace validation."""

    def test_valid_workspace_returns_ok(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_workspace_root_safe(workspace)
        assert isinstance(result, Ok)
        assert result.value == workspace.resolve()

    def test_missing_workspace_returns_err(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"

        result = validate_workspace_root_safe(missing)
        assert isinstance(result, Err)

    def test_file_as_workspace_returns_err(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("content", encoding="utf-8")

        result = validate_workspace_root_safe(file_path)
        assert isinstance(result, Err)


class TestEdgeCases:
    """Edge case tests for security module."""

    def test_empty_string_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Empty string expands to cwd which may not be in workspace
        # This should be blocked if cwd != workspace
        result = validate_path_safe("", workspace)
        # Empty string resolves to cwd, which is outside workspace in tests
        assert isinstance(result, Err)

    def test_dot_path_outside_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # "." resolves to cwd which is outside workspace in tests
        result = validate_path_safe(".", workspace)
        assert isinstance(result, Err)

    def test_workspace_itself_is_valid(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # The workspace directory itself is valid
        result = validate_path_safe(workspace, workspace)
        assert isinstance(result, Ok)

    def test_double_dot_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_path_safe("..", workspace)
        assert isinstance(result, Err)

    def test_path_with_null_byte_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Null byte injection attempt
        with pytest.raises((ValueError, OSError)):
            validate_path("file\x00.txt", workspace)

    def test_unicode_path_handled(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "файл.txt"  # Russian for "file"
        target.write_text("content", encoding="utf-8")

        result = validate_path_safe(target, workspace)
        assert isinstance(result, Ok)

    def test_path_with_spaces(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "file with spaces.txt"
        target.write_text("content", encoding="utf-8")

        result = validate_path_safe(target, workspace)
        assert isinstance(result, Ok)

    def test_nested_symlinks_within_workspace(self, tmp_path: Path) -> None:
        """Symlinks within workspace that don't escape are valid."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        subdir = workspace / "subdir"
        subdir.mkdir()

        target = subdir / "target.txt"
        target.write_text("content", encoding="utf-8")

        link = workspace / "link.txt"
        link.symlink_to(target)

        result = validate_path_safe(link, workspace)
        assert isinstance(result, Ok)
        assert result.value == target.resolve()


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_workspace_validation_error_is_permission_error(self) -> None:
        exc = WorkspaceValidationError("test message")
        assert isinstance(exc, PermissionError)

    def test_path_validation_error_is_permission_error(self) -> None:
        exc = PathValidationError("test message")
        assert isinstance(exc, PermissionError)

    def test_exception_with_context(self) -> None:
        exc = WorkspaceValidationError("test", context={"key": "value"})
        assert exc.context == {"key": "value"}
