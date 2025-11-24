from __future__ import annotations

import os
from datetime import datetime

from jpscripts.commands.nav import _human_time, _scan_recent


def test_human_time_formats_timestamp():
    timestamp = datetime(2024, 1, 2, 15, 30).timestamp()
    assert _human_time(timestamp) == "2024-01-02 15:30"


def test_scan_recent_sorts_and_ignores(tmp_path):
    base = datetime.now().timestamp()

    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    os.utime(keep, (base + 10, base + 10))

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "file.txt"
    nested_file.write_text("nested")
    os.utime(nested_file, (base, base))

    ignored_dir = tmp_path / "node_modules"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "ignore.js"
    ignored_file.write_text("ignore")

    entries = _scan_recent(tmp_path, max_depth=2, include_dirs=False, ignore_dirs={"node_modules"})
    names = [entry.path.name for entry in entries]

    assert names[0] == "keep.txt"
    assert "file.txt" in names
    assert "ignore.js" not in names
