"""Tests for memory content integrity and hash-based drift detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from jpscripts.core.config import AppConfig
from jpscripts.core.memory import (
    _compute_file_hash,
    _fallback_path,
    _load_entries,
    _resolve_store,
    prune_memory,
    save_memory,
)


def _test_config(tmp_path: Path) -> AppConfig:
    """Create a test config with semantic search disabled."""
    store_path = tmp_path / "memory.lance"
    return AppConfig(
        workspace_root=str(tmp_path),
        memory_store=str(store_path),
        use_semantic_search=False,
    )


class TestHashCapture:
    """Test that content hashes are captured on save."""

    def test_compute_file_hash_returns_md5(self, tmp_path: Path) -> None:
        """Verify hash computation returns consistent MD5."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')", encoding="utf-8")

        hash1 = _compute_file_hash(test_file)
        hash2 = _compute_file_hash(test_file)

        assert hash1 is not None
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_compute_file_hash_returns_none_for_missing(self, tmp_path: Path) -> None:
        """Verify missing files return None."""
        missing = tmp_path / "nonexistent.py"
        assert _compute_file_hash(missing) is None

    def test_save_memory_captures_hash(self, tmp_path: Path) -> None:
        """Verify save_memory sets content_hash when source_path provided."""
        source_file = tmp_path / "source.py"
        source_file.write_text("def foo(): pass", encoding="utf-8")

        config = _test_config(tmp_path)

        entry = save_memory(
            "A memory about source.py",
            tags=["test"],
            config=config,
            source_path=str(source_file),
        )

        assert entry.content_hash is not None
        assert entry.content_hash == _compute_file_hash(source_file)

    def test_save_memory_no_hash_without_source_path(self, tmp_path: Path) -> None:
        """Verify save_memory does not set content_hash when no source_path."""
        config = _test_config(tmp_path)

        entry = save_memory(
            "A general memory",
            tags=["general"],
            config=config,
        )

        assert entry.content_hash is None


class TestPruneDrift:
    """Test that prune removes entries with drifted content."""

    def test_prune_removes_drifted_entry(self, tmp_path: Path) -> None:
        """Verify entries with hash mismatch are pruned."""
        stable_file = tmp_path / "stable.py"
        drift_file = tmp_path / "drift.py"

        stable_file.write_text("# stable", encoding="utf-8")
        drift_file.write_text("# original", encoding="utf-8")

        config = _test_config(tmp_path)

        # Save memories for both files
        save_memory(
            "Stable memory",
            tags=["stable"],
            config=config,
            source_path=str(stable_file),
        )
        save_memory(
            "Drift memory",
            tags=["drift"],
            config=config,
            source_path=str(drift_file),
        )

        # Modify drift file to cause hash mismatch
        drift_file.write_text("# modified content", encoding="utf-8")

        # Prune and verify
        pruned_count = prune_memory(config)

        # Load remaining entries
        jsonl_path = _fallback_path(_resolve_store(config))
        remaining = _load_entries(jsonl_path)

        assert pruned_count == 1
        assert len(remaining) == 1
        assert remaining[0].tags == ["stable"]

    def test_prune_keeps_entries_without_hash(self, tmp_path: Path) -> None:
        """Verify entries without content_hash are kept (backward compat)."""
        config = _test_config(tmp_path)

        # Save memory without source_path (no hash)
        entry = save_memory(
            "General memory",
            tags=["general"],
            config=config,
        )

        assert entry.content_hash is None

        # Prune should keep it
        pruned_count = prune_memory(config)

        jsonl_path = _fallback_path(_resolve_store(config))
        remaining = _load_entries(jsonl_path)

        assert pruned_count == 0
        assert len(remaining) == 1

    def test_prune_removes_missing_file_entry(self, tmp_path: Path) -> None:
        """Verify entries with missing source files are pruned."""
        temp_file = tmp_path / "temp.py"
        temp_file.write_text("# temporary", encoding="utf-8")

        config = _test_config(tmp_path)

        # Save memory with source_path
        save_memory(
            "Temp memory",
            tags=["temp"],
            config=config,
            source_path=str(temp_file),
        )

        # Delete the source file
        temp_file.unlink()

        # Prune and verify
        pruned_count = prune_memory(config)

        jsonl_path = _fallback_path(_resolve_store(config))
        remaining = _load_entries(jsonl_path)

        assert pruned_count == 1
        assert len(remaining) == 0

    def test_prune_keeps_unchanged_file_entry(self, tmp_path: Path) -> None:
        """Verify entries with unchanged files are kept."""
        stable_file = tmp_path / "stable.py"
        stable_file.write_text("# unchanged", encoding="utf-8")

        config = _test_config(tmp_path)

        # Save memory with source_path
        save_memory(
            "Stable memory",
            tags=["stable"],
            config=config,
            source_path=str(stable_file),
        )

        # Do NOT modify the file

        # Prune and verify
        pruned_count = prune_memory(config)

        jsonl_path = _fallback_path(_resolve_store(config))
        remaining = _load_entries(jsonl_path)

        assert pruned_count == 0
        assert len(remaining) == 1
        assert remaining[0].tags == ["stable"]
