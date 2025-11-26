from __future__ import annotations

import ast
import json
from pathlib import Path

from jpscripts.core.context import get_file_skeleton, read_file_context, smart_read_context


def test_read_file_context_truncates(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("abcd" * 100, encoding="utf-8")

    result = read_file_context(path, max_chars=10)
    assert result == "abcdabcdab"


def test_read_file_context_handles_binary(tmp_path: Path) -> None:
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\x00\xfe")

    result = read_file_context(path, max_chars=10)
    assert result is None


def test_smart_read_context_aligns_to_definition(tmp_path: Path) -> None:
    source = (
        "def first():\n"
        "    return 'a'\n"
        "\n"
        "def second():\n"
        "    return 'b'\n"
    )
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")

    snippet = smart_read_context(path, max_chars=200)

    assert "def first" in snippet
    assert "def second" in snippet
    ast.parse(snippet)


def test_get_file_skeleton_replaces_long_bodies(tmp_path: Path) -> None:
    source = (
        "def big():\n"
        '    """docstring"""\n'
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    return a + b + c + d\n"
    )
    path = tmp_path / "skeleton.py"
    path.write_text(source, encoding="utf-8")

    skeleton = get_file_skeleton(path)

    assert "def big" in skeleton
    assert "pass" in skeleton or "..." in skeleton
    assert "return a + b + c + d" not in skeleton
    ast.parse(skeleton)


def test_smart_read_context_structured_json(tmp_path: Path) -> None:
    payload = {"a": 1, "b": 2}
    text = json.dumps(payload, indent=2)
    path = tmp_path / "data.json"
    path.write_text(text, encoding="utf-8")

    snippet = smart_read_context(path, max_chars=len(text) - 2)

    assert len(snippet) <= len(text) - 2
    if snippet:
        json.loads(snippet)
