from __future__ import annotations

import os
from pathlib import Path

import pytest

from jpscripts.core.context import HARD_FILE_CONTEXT_LIMIT, read_file_context
from jpscripts.core.security import WorkspaceValidationError, cache_workspace_root, validate_path


def test_validate_path_blocks_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(PermissionError):
        validate_path("../../../../etc/passwd", workspace)


def test_validate_path_blocks_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    malicious_link = workspace / "link.txt"
    malicious_link.symlink_to(outside_file)

    with pytest.raises(PermissionError):
        validate_path(malicious_link, workspace)


def test_cache_workspace_root_requires_existing_dir(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(WorkspaceValidationError):
        cache_workspace_root(missing_root)


def test_validate_path_uses_cached_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_workspace_root(workspace)

    target = workspace / "child" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data", encoding="utf-8")

    assert validate_path(target, None) == target.resolve()


def test_read_file_context_caps_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    large_file = workspace / "big.txt"
    # Write ~50MB without holding the whole string in memory at once
    chunk = "x" * 1024 * 1024
    with large_file.open("w", encoding="utf-8") as fh:
        for _ in range(50):
            fh.write(chunk)

    content = read_file_context(large_file, max_chars=HARD_FILE_CONTEXT_LIMIT * 10)

    assert content is not None
    assert len(content) == HARD_FILE_CONTEXT_LIMIT
    assert content == "x" * HARD_FILE_CONTEXT_LIMIT
