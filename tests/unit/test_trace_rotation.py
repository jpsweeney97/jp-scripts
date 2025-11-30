"""Tests for TraceRecorder rotation and cleanup."""

from __future__ import annotations

import gzip
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jpscripts.core.engine import TraceRecorder


class TestTraceRotation:
    """Test trace file rotation when size limit is exceeded."""

    def test_rotation_compresses_large_file(self, tmp_path: Path) -> None:
        """When trace file exceeds 10MB, it should be compressed and truncated."""
        recorder = TraceRecorder(tmp_path, trace_id="test-trace")

        # Create a file just under the limit
        content = "x" * (TraceRecorder.MAX_TRACE_SIZE - 100)
        recorder._path.write_text(content)

        # Write one more line - should NOT trigger rotation yet
        recorder._write_line('{"step": 1}')
        assert recorder._path.exists()
        assert not list(tmp_path.glob("*.jsonl.gz"))

        # Now make file exceed limit
        content = "x" * TraceRecorder.MAX_TRACE_SIZE
        recorder._path.write_text(content)

        # Write again - should trigger rotation
        recorder._write_line('{"step": 2}')

        # Original file should be truncated (only has new line)
        assert recorder._path.read_text() == '{"step": 2}\n'

        # Should have a compressed archive
        archives = list(tmp_path.glob("*.jsonl.gz"))
        assert len(archives) == 1

        # Archive should contain the original content
        with gzip.open(archives[0], "rt") as f:
            archived = f.read()
        assert len(archived) == TraceRecorder.MAX_TRACE_SIZE

    def test_rotation_filename_format(self, tmp_path: Path) -> None:
        """Verify archive filename includes trace_id and timestamp."""
        recorder = TraceRecorder(tmp_path, trace_id="my-trace-id")

        # Create oversized file
        content = "x" * (TraceRecorder.MAX_TRACE_SIZE + 1)
        recorder._path.write_text(content)

        recorder._write_line('{"data": "test"}')

        archives = list(tmp_path.glob("*.jsonl.gz"))
        assert len(archives) == 1

        archive_name = archives[0].name
        assert archive_name.startswith("my-trace-id.")
        assert archive_name.endswith(".jsonl.gz")

    def test_no_rotation_when_under_limit(self, tmp_path: Path) -> None:
        """Files under size limit should not be rotated."""
        recorder = TraceRecorder(tmp_path, trace_id="small")

        recorder._write_line('{"step": 1}')
        recorder._write_line('{"step": 2}')

        assert not list(tmp_path.glob("*.jsonl.gz"))
        content = recorder._path.read_text()
        assert '{"step": 1}' in content
        assert '{"step": 2}' in content


class TestTraceCleanup:
    """Test cleanup of old trace archives."""

    def test_cleanup_removes_old_archives(self, tmp_path: Path) -> None:
        """Archives older than 30 days should be removed."""
        recorder = TraceRecorder(tmp_path, trace_id="cleanup-test")

        # Create an old archive (>30 days)
        old_archive = tmp_path / "old-trace.20200101_120000.jsonl.gz"
        with gzip.open(old_archive, "wt") as f:
            f.write("old data")

        # Set its mtime to 31 days ago
        old_time = datetime.now(UTC) - timedelta(days=31)
        os.utime(old_archive, (old_time.timestamp(), old_time.timestamp()))

        # Create a recent archive
        recent_archive = tmp_path / "recent-trace.jsonl.gz"
        with gzip.open(recent_archive, "wt") as f:
            f.write("recent data")

        # Trigger cleanup
        recorder._cleanup_old_archives()

        # Old archive should be deleted
        assert not old_archive.exists()

        # Recent archive should remain
        assert recent_archive.exists()

    def test_cleanup_keeps_recent_archives(self, tmp_path: Path) -> None:
        """Archives less than 30 days old should be kept."""
        recorder = TraceRecorder(tmp_path, trace_id="keep-test")

        # Create several archives with various ages
        for days_ago in [5, 15, 29]:
            archive = tmp_path / f"trace-{days_ago}d.jsonl.gz"
            with gzip.open(archive, "wt") as f:
                f.write(f"data from {days_ago} days ago")

            old_time = datetime.now(UTC) - timedelta(days=days_ago)
            os.utime(archive, (old_time.timestamp(), old_time.timestamp()))

        recorder._cleanup_old_archives()

        # All should still exist (< 30 days)
        for days_ago in [5, 15, 29]:
            archive = tmp_path / f"trace-{days_ago}d.jsonl.gz"
            assert archive.exists(), f"Archive from {days_ago} days ago should exist"

    def test_cleanup_ignores_non_gz_files(self, tmp_path: Path) -> None:
        """Non-.gz files should not be affected by cleanup."""
        recorder = TraceRecorder(tmp_path, trace_id="ignore-test")

        # Create a non-gz file
        non_gz = tmp_path / "important.jsonl"
        non_gz.write_text("important data")

        # Set its mtime to 31 days ago
        old_time = datetime.now(UTC) - timedelta(days=31)
        os.utime(non_gz, (old_time.timestamp(), old_time.timestamp()))

        recorder._cleanup_old_archives()

        # Should still exist
        assert non_gz.exists()


class TestTraceRecorderInit:
    """Test TraceRecorder initialization and directory handling."""

    def test_creates_trace_directory(self, tmp_path: Path) -> None:
        """Creates trace directory if it doesn't exist."""
        trace_dir = tmp_path / "new" / "trace" / "dir"
        assert not trace_dir.exists()

        recorder = TraceRecorder(trace_dir, trace_id="init-test")

        assert trace_dir.exists()
        assert recorder.trace_dir == trace_dir.expanduser()

    def test_uses_existing_directory(self, tmp_path: Path) -> None:
        """Uses existing directory without error."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        recorder = TraceRecorder(tmp_path, trace_id="existing-test")
        assert recorder.trace_dir == tmp_path.expanduser()

    def test_path_property(self, tmp_path: Path) -> None:
        """Path property returns correct file path."""
        recorder = TraceRecorder(tmp_path, trace_id="path-test")
        expected = tmp_path / "path-test.jsonl"
        assert recorder.path == expected
