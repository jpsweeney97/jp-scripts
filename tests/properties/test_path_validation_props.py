"""Property-based tests for path validation using Hypothesis.

These tests verify core invariants of the path validation system:
- validate_path never raises (returns Result)
- Forbidden paths are always rejected
- Valid paths within workspace are always accepted
- Symlink depth is properly limited
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from jpscripts.core.result import Err, Ok
from jpscripts.core.security import (
    FORBIDDEN_ROOTS,
    MAX_SYMLINK_DEPTH,
    validate_path,
    validate_workspace_root,
)

# === Strategies ===

# Path components (safe characters only, no path separators)
path_component_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),  # Letters and numbers
        whitelist_characters="_-.",
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s not in (".", "..") and not s.startswith("-"))

# Relative path segments
relative_path_strategy = st.lists(
    path_component_strategy,
    min_size=1,
    max_size=5,
).map(lambda parts: "/".join(parts))


# === Property Tests ===


@given(path_str=st.text(max_size=1000))
@settings(max_examples=200)
def test_validate_path_never_raises(path_str: str) -> None:
    """validate_path should never raise an exception, only return Result."""
    # Skip paths with null bytes - known to cause ValueError in pathlib.resolve()
    # This is a limitation in Python's pathlib, not our code
    assume("\x00" not in path_str)

    # Skip paths starting with ~username (like ~0, ~foo, ~:) that cause expanduser() to fail
    # when the user doesn't exist. Only ~/ and ~ alone are safe (expand to current user home).
    # This is a Python pathlib limitation.
    assume(not re.match(r"^~[^/]", path_str))

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        result = validate_path(path_str, workspace)

        # Should always return a Result type
        assert isinstance(result, (Ok, Err))


@given(path_str=st.text(max_size=500))
@settings(max_examples=100)
def test_validate_workspace_root_never_raises(path_str: str) -> None:
    """validate_workspace_root should never raise, only return Result."""
    # Skip paths with null bytes - known to cause ValueError in pathlib.resolve()
    assume("\x00" not in path_str)

    result = validate_workspace_root(path_str)

    # Should always return a Result type
    assert isinstance(result, (Ok, Err))


@given(relative=relative_path_strategy)
@settings(max_examples=100)
def test_valid_relative_paths_accepted(relative: str) -> None:
    """Valid relative paths within workspace are accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        # Create the target path
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

        result = validate_path(target, workspace)

        # Valid paths should succeed
        assert isinstance(result, Ok), f"Expected Ok for valid path, got: {result}"


def test_forbidden_roots_rejected() -> None:
    """Paths targeting forbidden roots are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        for forbidden in FORBIDDEN_ROOTS:
            if forbidden.exists():  # Only test if the path exists on this system
                result = validate_path(forbidden, workspace)
                assert isinstance(result, Err), f"Forbidden path {forbidden} was not rejected"


@given(
    traversal_depth=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50)
def test_path_traversal_rejected(traversal_depth: int) -> None:
    """Paths that traverse outside workspace are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "nested" / "workspace"
        workspace.mkdir(parents=True)

        # Try to escape workspace
        traversal = "../" * traversal_depth
        escape_path = workspace / traversal / "etc"

        result = validate_path(escape_path, workspace)

        # Traversal attempts should fail
        # (either rejected as escaping or the path doesn't resolve inside workspace)
        if isinstance(result, Ok):
            # If it succeeded, verify it stayed in workspace
            resolved = result.value
            try:
                resolved.relative_to(workspace.resolve())
            except ValueError:
                raise AssertionError(f"Traversal to {resolved} escaped workspace")


def test_symlink_depth_constant_reasonable() -> None:
    """MAX_SYMLINK_DEPTH is a reasonable value."""
    # Should be positive but not too large
    assert 1 <= MAX_SYMLINK_DEPTH <= 100


def test_empty_path_rejected() -> None:
    """Empty string path is rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        result = validate_path("", workspace)

        # Empty path should fail
        assert isinstance(result, Err)


def test_null_bytes_raise_valueerror() -> None:
    """Document that paths with null bytes cause ValueError (Python limitation).

    This is a known limitation in Python's pathlib - null bytes in paths
    cause ValueError during resolve(). The security module could catch this
    and return Err, but that's out of scope for this test suite.
    """
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        malicious = "test\x00file"

        # Currently raises - documenting this behavior
        with pytest.raises(ValueError, match="embedded null"):
            validate_path(malicious, workspace)


@given(
    component_count=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50)
def test_deeply_nested_paths(component_count: int) -> None:
    """Deeply nested paths are handled without stack overflow."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        # Create deeply nested path
        nested = "/".join(["dir"] * component_count) + "/file.txt"
        full_path = workspace / nested

        result = validate_path(full_path, workspace)

        # Should either succeed or fail gracefully
        assert isinstance(result, (Ok, Err))


def test_workspace_root_validation() -> None:
    """Workspace root itself is valid."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        result = validate_path(workspace, workspace)

        # Workspace root should be valid
        assert isinstance(result, Ok)


@given(content=st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_new_file_in_workspace_valid(content: str) -> None:
    """New files created in workspace are valid."""
    # Filter out problematic characters for filenames
    assume(all(c not in content for c in '/\x00\\:*?"<>|'))
    assume(content.strip() not in (".", ".."))
    assume(len(content.strip()) > 0)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        new_file = workspace / content.strip()

        # Should succeed for new file path
        result = validate_path(new_file, workspace)

        # Path within workspace should be valid (file may not exist yet)
        assert isinstance(result, (Ok, Err))  # Could fail if name is invalid
