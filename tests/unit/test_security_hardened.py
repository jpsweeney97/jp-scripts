from __future__ import annotations

from pathlib import Path

import pytest

from jpscripts.core.context import DEFAULT_MODEL_CONTEXT_LIMIT, read_file_context
from jpscripts.core.result import Err, Ok
from jpscripts.core.security import validate_path, validate_workspace_root


def test_validate_path_blocks_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = validate_path("../../../../etc/passwd", workspace)
    assert isinstance(result, Err)


def test_validate_path_blocks_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    malicious_link = workspace / "link.txt"
    malicious_link.symlink_to(outside_file)

    result = validate_path(malicious_link, workspace)
    assert isinstance(result, Err)


def test_cache_workspace_root_requires_existing_dir(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    result = validate_workspace_root(missing_root)
    assert isinstance(result, Err)


def test_validate_path_requires_valid_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = workspace / "child" / "file.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("data", encoding="utf-8")

    result = validate_path(target, workspace)
    assert isinstance(result, Ok)
    assert result.value == target.resolve()


def test_validate_path_returns_err_on_invalid_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    # Workspace doesn't exist, so validation fails
    result = validate_path("file.txt", workspace)
    assert isinstance(result, Err)


def test_read_file_context_caps_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    large_file = workspace / "big.txt"
    # Write ~50MB without holding the whole string in memory at once
    chunk = "x" * 1024 * 1024
    with large_file.open("w", encoding="utf-8") as fh:
        for _ in range(50):
            fh.write(chunk)

    content = read_file_context(large_file, max_chars=DEFAULT_MODEL_CONTEXT_LIMIT * 10)

    assert content is not None
    assert len(content) == DEFAULT_MODEL_CONTEXT_LIMIT
    assert content == "x" * DEFAULT_MODEL_CONTEXT_LIMIT
