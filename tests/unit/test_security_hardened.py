from __future__ import annotations

import os
from pathlib import Path

import pytest

from jpscripts.core.context import HARD_CONTEXT_CAP, read_file_context
from jpscripts.core.security import WorkspaceValidationError, validate_path, validate_workspace_root


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
        validate_workspace_root(missing_root)


def test_validate_path_requires_valid_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = workspace / "child" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data", encoding="utf-8")

    assert validate_path(target, workspace) == target.resolve()


def test_validate_path_raises_on_invalid_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    with pytest.raises(WorkspaceValidationError):
        validate_path("file.txt", workspace)


def test_read_file_context_caps_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    large_file = workspace / "big.txt"
    # Write ~50MB without holding the whole string in memory at once
    chunk = "x" * 1024 * 1024
    with large_file.open("w", encoding="utf-8") as fh:
        for _ in range(50):
            fh.write(chunk)

    content = read_file_context(large_file, max_chars=HARD_CONTEXT_CAP * 10)

    assert content is not None
    assert len(content) == HARD_CONTEXT_CAP
    assert content == "x" * HARD_CONTEXT_CAP
