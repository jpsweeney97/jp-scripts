from __future__ import annotations

from pathlib import Path

from jpscripts.core.context import read_file_context


def test_read_file_context_truncates(tmp_path: Path):
    path = tmp_path / "file.txt"
    path.write_text("abcd" * 100, encoding="utf-8")

    result = read_file_context(path, max_chars=10)
    assert result == "abcdabcdab"


def test_read_file_context_handles_binary(tmp_path: Path):
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\x00\xfe")

    result = read_file_context(path, max_chars=10)
    assert result is None
