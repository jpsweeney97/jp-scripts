"""Tests for constitutional governance compliance checking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jpscripts.core.governance import (
    ViolationType,
    Violation,
    check_source_compliance,
    format_violations_for_agent,
)


class TestViolationDetection:
    """Test AST-based violation detection."""

    def test_detects_bare_except(self, tmp_path: Path) -> None:
        """Bare except: clauses should be flagged."""
        source = """\
def foo():
    try:
        x = 1
    except:
        pass
    return x
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert any(v.type == ViolationType.BARE_EXCEPT for v in violations)

    def test_detects_shell_true(self, tmp_path: Path) -> None:
        """subprocess with shell=True should be flagged."""
        source = """\
import subprocess
def run():
    subprocess.run("ls -la", shell=True)
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert any(v.type == ViolationType.SHELL_TRUE for v in violations)

    def test_detects_os_system(self, tmp_path: Path) -> None:
        """os.system() should be flagged."""
        source = """\
import os
def run():
    os.system("ls")
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert any(v.type == ViolationType.OS_SYSTEM for v in violations)

    def test_clean_code_no_violations(self, tmp_path: Path) -> None:
        """Well-written code should have no violations."""
        source = """\
import asyncio
async def run():
    try:
        result = await asyncio.to_thread(lambda: 42)
    except ValueError as e:
        print(e)
    return result
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        # Should not have bare_except or os_system violations
        assert not any(v.type == ViolationType.BARE_EXCEPT for v in violations)
        assert not any(v.type == ViolationType.OS_SYSTEM for v in violations)

    def test_detects_sync_subprocess_in_async(self, tmp_path: Path) -> None:
        """subprocess.run in async context without asyncio.to_thread should be flagged."""
        source = """\
import subprocess
async def run():
    result = subprocess.run(["ls"])
    return result
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert any(v.type == ViolationType.SYNC_SUBPROCESS for v in violations)

    def test_sync_subprocess_outside_async_is_ok(self, tmp_path: Path) -> None:
        """subprocess.run outside async context is allowed."""
        source = """\
import subprocess
def run():
    result = subprocess.run(["ls"])
    return result
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert not any(v.type == ViolationType.SYNC_SUBPROCESS for v in violations)


class TestFormatViolations:
    """Test formatting violations for agent prompts."""

    def test_formats_multiple_violations(self) -> None:
        """Multiple violations should be formatted clearly."""
        violations = [
            Violation(
                type=ViolationType.BARE_EXCEPT,
                file=Path("test.py"),
                line=10,
                column=0,
                message="Use specific exception types",
                suggestion="Catch specific exceptions",
                severity="error",
            ),
            Violation(
                type=ViolationType.OS_SYSTEM,
                file=Path("test.py"),
                line=15,
                column=0,
                message="Never use os.system()",
                suggestion="Use subprocess",
                severity="error",
            ),
        ]

        formatted = format_violations_for_agent(violations)

        assert "BARE_EXCEPT" in formatted
        assert "OS_SYSTEM" in formatted
        assert "test.py" in formatted
        assert "10" in formatted
        assert "15" in formatted

    def test_empty_violations_returns_empty(self) -> None:
        """No violations should return empty string."""
        formatted = format_violations_for_agent([])
        assert formatted == ""

    def test_separates_errors_and_warnings(self) -> None:
        """Errors and warnings should be in separate sections."""
        violations = [
            Violation(
                type=ViolationType.OS_SYSTEM,
                file=Path("test.py"),
                line=10,
                column=0,
                message="Never use os.system()",
                suggestion="Use subprocess",
                severity="error",
            ),
            Violation(
                type=ViolationType.SYNC_OPEN,
                file=Path("test.py"),
                line=20,
                column=0,
                message="Sync open in async",
                suggestion="Use aiofiles",
                severity="warning",
            ),
        ]

        formatted = format_violations_for_agent(violations)

        assert "Errors (must fix)" in formatted
        assert "Warnings (should fix)" in formatted


class TestMultipleViolations:
    """Test detection of multiple violations in the same file."""

    def test_detects_multiple_violations(self, tmp_path: Path) -> None:
        """Multiple violations in one file should all be detected."""
        source = """\
import os
import subprocess

def bad_code():
    try:
        os.system("echo hello")
    except:
        subprocess.run("ls", shell=True)
"""
        violations = check_source_compliance(source, tmp_path / "test.py")

        assert any(v.type == ViolationType.OS_SYSTEM for v in violations)
        assert any(v.type == ViolationType.BARE_EXCEPT for v in violations)
        assert any(v.type == ViolationType.SHELL_TRUE for v in violations)
        assert len(violations) >= 3

    def test_detects_sync_open_in_async(self, tmp_path: Path) -> None:
        """Synchronous open() in async context should be flagged as warning."""
        source = """\
async def read_file():
    with open("test.txt") as f:
        return f.read()
"""
        violations = check_source_compliance(source, tmp_path / "test.py")
        assert any(v.type == ViolationType.SYNC_OPEN for v in violations)
