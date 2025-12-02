"""Tests for core/search module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jpscripts.net.search import (
    TodoEntry,
    _ensure_rg,
    get_ripgrep_cmd,
    run_ripgrep,
    scan_todos,
)


class TestEnsureRg:
    """Tests for _ensure_rg function."""

    def test_returns_path_when_found(self) -> None:
        """Returns binary path when ripgrep is found."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            result = _ensure_rg()
            assert result == "/usr/bin/rg"

    def test_raises_when_not_found(self) -> None:
        """Raises RuntimeError when ripgrep is not found."""
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(RuntimeError, match=r"ripgrep.*not found"),
        ):
            _ensure_rg()


class TestRunRipgrep:
    """Tests for run_ripgrep function."""

    def test_basic_search(self, tmp_path: Path) -> None:
        """Basic search returns matching lines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar\nhello again\n")

        with patch("shutil.which", return_value="/usr/bin/rg"):
            # Mock subprocess to simulate rg output
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["hello world\nhello again\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            with patch("subprocess.Popen", return_value=mock_proc):
                result = run_ripgrep("hello", tmp_path)
                assert "hello" in result

    def test_with_context(self, tmp_path: Path) -> None:
        """Search with context includes -C flag."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["match\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            captured_cmd = []

            def capture_popen(cmd, **kwargs):
                captured_cmd.extend(cmd)
                return mock_proc

            with patch("subprocess.Popen", side_effect=capture_popen):
                run_ripgrep("test", tmp_path, context=3)
                assert "-C3" in captured_cmd

    def test_with_line_numbers(self, tmp_path: Path) -> None:
        """Search with line numbers includes --line-number flag."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["1:match\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            captured_cmd = []

            def capture_popen(cmd, **kwargs):
                captured_cmd.extend(cmd)
                return mock_proc

            with patch("subprocess.Popen", side_effect=capture_popen):
                run_ripgrep("test", tmp_path, line_number=True)
                assert "--line-number" in captured_cmd

    def test_with_follow(self, tmp_path: Path) -> None:
        """Search with follow includes --follow flag."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["match\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            captured_cmd = []

            def capture_popen(cmd, **kwargs):
                captured_cmd.extend(cmd)
                return mock_proc

            with patch("subprocess.Popen", side_effect=capture_popen):
                run_ripgrep("test", tmp_path, follow=True)
                assert "--follow" in captured_cmd

    def test_with_pcre2(self, tmp_path: Path) -> None:
        """Search with pcre2 includes --pcre2 flag."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["match\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            captured_cmd = []

            def capture_popen(cmd, **kwargs):
                captured_cmd.extend(cmd)
                return mock_proc

            with patch("subprocess.Popen", side_effect=capture_popen):
                run_ripgrep("test", tmp_path, pcre2=True)
                assert "--pcre2" in captured_cmd

    def test_with_extra_args(self, tmp_path: Path) -> None:
        """Search with extra_args passes them through."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["match\n", ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            captured_cmd = []

            def capture_popen(cmd, **kwargs):
                captured_cmd.extend(cmd)
                return mock_proc

            with patch("subprocess.Popen", side_effect=capture_popen):
                run_ripgrep("test", tmp_path, extra_args=["--type", "py"])
                assert "--type" in captured_cmd
                assert "py" in captured_cmd

    def test_ripgrep_error(self, tmp_path: Path) -> None:
        """Ripgrep error raises RuntimeError."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["", ""]
            mock_proc.stderr.read.return_value = "error: invalid pattern"
            mock_proc.returncode = 2
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            with (
                patch("subprocess.Popen", return_value=mock_proc),
                pytest.raises(RuntimeError, match="ripgrep error"),
            ):
                run_ripgrep("test", tmp_path)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError raises RuntimeError."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            with (
                patch("subprocess.Popen", side_effect=FileNotFoundError()),
                pytest.raises(RuntimeError, match="execution failed"),
            ):
                run_ripgrep("test", tmp_path)

    def test_truncation_with_max_chars(self, tmp_path: Path) -> None:
        """Output is truncated when exceeding max_chars."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            # Simulate large output that exceeds max_chars
            mock_proc = MagicMock()
            mock_proc.stdout.read.side_effect = ["a" * 5000, "b" * 5000, ""]
            mock_proc.stderr.read.return_value = ""
            mock_proc.returncode = 0
            mock_proc.terminate = MagicMock()
            mock_proc.wait = MagicMock()
            mock_proc.__enter__ = MagicMock(return_value=mock_proc)
            mock_proc.__exit__ = MagicMock(return_value=False)

            with patch("subprocess.Popen", return_value=mock_proc):
                result = run_ripgrep("test", tmp_path, max_chars=100)
                assert "[truncated]" in result
                mock_proc.terminate.assert_called_once()


class TestGetRipgrepCmd:
    """Tests for get_ripgrep_cmd function."""

    def test_basic_cmd(self, tmp_path: Path) -> None:
        """Basic command includes pattern and path."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            cmd = get_ripgrep_cmd("pattern", tmp_path)
            assert "/usr/bin/rg" in cmd
            assert "--color=always" in cmd
            assert "pattern" in cmd
            assert str(tmp_path) in cmd

    def test_with_all_options(self, tmp_path: Path) -> None:
        """Command includes all options when set."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            cmd = get_ripgrep_cmd(
                "pattern",
                tmp_path,
                context=2,
                line_number=True,
                follow=True,
                pcre2=True,
            )
            assert "-C2" in cmd
            assert "--line-number" in cmd
            assert "--follow" in cmd
            assert "--pcre2" in cmd

    def test_expands_user_path(self, tmp_path: Path) -> None:
        """Path with ~ is expanded."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            # Create a path that would need expansion
            user_path = Path("~/test")
            cmd = get_ripgrep_cmd("pattern", user_path)
            # The result should be the expanded path
            assert "~" not in cmd[-1] or str(Path.home()) in cmd[-1]


class TestTodoEntry:
    """Tests for TodoEntry dataclass."""

    def test_fields(self) -> None:
        """TodoEntry stores all fields correctly."""
        entry = TodoEntry(
            file="src/main.py",
            line=42,
            type="TODO",
            text="# TODO: implement this",
        )
        assert entry.file == "src/main.py"
        assert entry.line == 42
        assert entry.type == "TODO"
        assert entry.text == "# TODO: implement this"

    def test_different_types(self) -> None:
        """TodoEntry works with different marker types."""
        for marker in ["TODO", "FIXME", "HACK", "BUG"]:
            entry = TodoEntry(file="test.py", line=1, type=marker, text=f"# {marker}")
            assert entry.type == marker


class TestScanTodos:
    """Tests for scan_todos async function."""

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path: Path) -> None:
        """No matches returns empty list."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(return_value=b"")
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert result == []

    @pytest.mark.asyncio
    async def test_parses_json_output(self, tmp_path: Path) -> None:
        """Parses ripgrep JSON output correctly."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            # Create JSON match output
            json_match = {
                "type": "match",
                "data": {
                    "path": {"text": "src/main.py"},
                    "line_number": 10,
                    "submatches": [{"match": {"text": "TODO"}}],
                    "lines": {"text": "# TODO: fix this bug"},
                },
            }
            json_line = json.dumps(json_match).encode() + b"\n"

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=[json_line, b""])
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert len(result) == 1
                assert result[0].file == "src/main.py"
                assert result[0].line == 10
                assert result[0].type == "TODO"
                assert "fix this bug" in result[0].text

    @pytest.mark.asyncio
    async def test_handles_multiple_matches(self, tmp_path: Path) -> None:
        """Handles multiple matches across files."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            matches = [
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "file1.py"},
                        "line_number": 5,
                        "submatches": [{"match": {"text": "TODO"}}],
                        "lines": {"text": "# TODO: first"},
                    },
                },
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "file2.py"},
                        "line_number": 10,
                        "submatches": [{"match": {"text": "FIXME"}}],
                        "lines": {"text": "# FIXME: second"},
                    },
                },
            ]
            json_lines = [json.dumps(m).encode() + b"\n" for m in matches]
            json_lines.append(b"")  # End of stream

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=json_lines)
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert len(result) == 2
                assert result[0].type == "TODO"
                assert result[1].type == "FIXME"

    @pytest.mark.asyncio
    async def test_skips_non_match_events(self, tmp_path: Path) -> None:
        """Non-match JSON events are skipped."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            events = [
                {"type": "begin", "data": {"path": {"text": "file.py"}}},
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "file.py"},
                        "line_number": 1,
                        "submatches": [{"match": {"text": "TODO"}}],
                        "lines": {"text": "# TODO: actual match"},
                    },
                },
                {"type": "end", "data": {"path": {"text": "file.py"}}},
                {"type": "summary", "data": {"elapsed_total": {"secs": 0}}},
            ]
            json_lines = [json.dumps(e).encode() + b"\n" for e in events]
            json_lines.append(b"")

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=json_lines)
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                # Only the match event should be included
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON lines are skipped."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            lines = [
                b"not valid json\n",
                b"",
            ]

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=lines)
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert result == []

    @pytest.mark.asyncio
    async def test_handles_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines are skipped."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            lines = [
                b"\n",
                b"  \n",
                b"",
            ]

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=lines)
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert result == []

    @pytest.mark.asyncio
    async def test_default_tag_when_no_submatch(self, tmp_path: Path) -> None:
        """Uses 'TODO' as default when submatch is empty."""
        with patch("shutil.which", return_value="/usr/bin/rg"):
            json_match = {
                "type": "match",
                "data": {
                    "path": {"text": "file.py"},
                    "line_number": 1,
                    "submatches": [],  # Empty submatches
                    "lines": {"text": "# some comment"},
                },
            }
            json_line = json.dumps(json_match).encode() + b"\n"

            mock_proc = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=[json_line, b""])
            mock_proc.wait = AsyncMock()

            with patch(
                "asyncio.create_subprocess_exec",
                return_value=mock_proc,
            ):
                result = await scan_todos(tmp_path)
                assert len(result) == 1
                assert result[0].type == "TODO"  # Default


class TestSearchIntegration:
    """Integration tests using actual ripgrep if available."""

    @pytest.fixture
    def has_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        import shutil

        return shutil.which("rg") is not None

    def test_real_ripgrep_search(self, tmp_path: Path, has_ripgrep: bool) -> None:
        """Test with real ripgrep if available."""
        if not has_ripgrep:
            pytest.skip("ripgrep not installed")

        # Create test files
        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: implement feature\nprint('hello')\n")

        result = run_ripgrep("TODO", tmp_path, line_number=True)
        assert "TODO" in result

    @pytest.mark.asyncio
    async def test_real_scan_todos(self, tmp_path: Path, has_ripgrep: bool) -> None:
        """Test scan_todos with real ripgrep if available."""
        if not has_ripgrep:
            pytest.skip("ripgrep not installed")

        # Create test files
        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: first task\n# FIXME: bug here\n")

        result = await scan_todos(tmp_path)
        assert len(result) >= 2
        types = {e.type for e in result}
        assert "TODO" in types
        assert "FIXME" in types
