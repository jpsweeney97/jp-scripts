"""
Tokenized command validation for secure shell execution (pre-flight).

This module provides robust command validation using proper tokenization
instead of regex-based pattern matching, preventing common bypass techniques.
At runtime, isolation is enforced by sandboxes (e.g., Docker); this validation
serves as an early rejection layer to avoid obviously dangerous commands.

Usage:
    from jpscripts.core.command_validation import validate_command, CommandVerdict

    verdict, reason = validate_command("ls -la", workspace_root)
    if verdict == CommandVerdict.ALLOWED:
        # Safe to execute
        pass
    else:
        # Block with reason
        print(f"Blocked: {reason}")
"""

from __future__ import annotations

import shlex
from enum import Enum, auto
from pathlib import Path
from typing import FrozenSet


class CommandVerdict(Enum):
    """Result of command validation."""

    ALLOWED = auto()
    BLOCKED_FORBIDDEN = auto()
    BLOCKED_NOT_ALLOWLISTED = auto()
    BLOCKED_PATH_ESCAPE = auto()
    BLOCKED_DANGEROUS_FLAG = auto()
    BLOCKED_METACHAR = auto()
    BLOCKED_UNPARSEABLE = auto()


# Binaries that are explicitly forbidden - these can cause damage
FORBIDDEN_BINARIES: FrozenSet[str] = frozenset({
    # Destructive file operations
    "rm", "rmdir", "unlink", "shred",
    "mv", "cp",  # Can overwrite files
    "dd",  # Can destroy disks
    # Permission/ownership changes
    "chmod", "chown", "chgrp", "chattr",
    # Privilege escalation
    "sudo", "su", "doas", "pkexec",
    # Interpreters (can execute arbitrary code)
    "python", "python2", "python3", "python3.8", "python3.9", "python3.10", "python3.11", "python3.12",
    "perl", "ruby", "node", "nodejs", "php", "lua",
    "sh", "bash", "zsh", "fish", "csh", "tcsh", "ksh", "dash",
    "awk", "gawk", "mawk", "nawk",
    "sed",  # Can modify files
    # System modification
    "mkfs", "mount", "umount", "fdisk", "parted",
    "systemctl", "service", "init",
    "useradd", "userdel", "usermod", "groupadd", "groupdel",
    "passwd", "chpasswd",
    # Network (potential exfiltration)
    "curl", "wget", "nc", "netcat", "ncat", "socat",
    "ssh", "scp", "rsync", "ftp", "sftp",
    # Package managers
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "pip", "npm", "yarn",
})

# Binaries that are explicitly allowed - read-only operations
ALLOWED_BINARIES: FrozenSet[str] = frozenset({
    # Directory listing
    "ls", "dir", "tree", "exa", "lsd",
    # File reading
    "cat", "head", "tail", "less", "more",
    "bat", "batcat",  # Modern cat alternatives
    # Search
    "grep", "egrep", "fgrep", "rg", "ripgrep", "ag",
    "find", "fd", "fdfind", "locate",
    # Git (read operations only - validated separately)
    "git",
# Utilities
"wc", "sort", "uniq", "cut", "tr", "column",
"pwd", "realpath", "dirname", "basename",
"which", "whereis", "type", "command",
"file", "stat", "du", "df",
"env", "printenv", "echo",
"date", "cal",
"true", "false",
# JPScripts CLI - required for recursive protocol execution (e.g., jp verify-protocol -> jp status-all)
"jp",
# JSON processing
"jq", "yq",
# Testing commands
"test", "[",
})

# Git subcommands that are safe (read-only)
ALLOWED_GIT_SUBCOMMANDS: FrozenSet[str] = frozenset({
    "status", "diff", "log", "show", "branch",
    "ls-files", "ls-tree", "ls-remote",
    "rev-parse", "rev-list",
    "describe", "shortlog",
    "blame", "annotate",
    "tag", "stash",  # list only - push/pop blocked
    "config", "--version", "version",
    "remote",  # show only
    "for-each-ref",
    "cat-file",
    "name-rev", "merge-base",
})

# Git subcommands that are dangerous
FORBIDDEN_GIT_SUBCOMMANDS: FrozenSet[str] = frozenset({
    "push", "pull", "fetch",
    "commit", "add", "rm", "mv",
    "reset", "revert", "rebase", "merge",
    "checkout", "switch", "restore",
    "clean", "gc", "prune",
    "filter-branch", "filter-repo",
    "submodule",
    "clone", "init",
})

# Flags that are dangerous regardless of command
DANGEROUS_FLAGS: FrozenSet[str] = frozenset({
    "--exec", "-exec",
    "--delete", "-delete",
    "-rf", "-fr",  # rm -rf pattern
    "--no-preserve-root",
    "-i",  # interactive - won't work anyway
    "--interactive",
})

# These flags are only dangerous for specific commands
CONTEXT_DANGEROUS_FLAGS: dict[str, FrozenSet[str]] = {
    "rm": frozenset({"-f", "--force", "-r", "-R", "--recursive"}),
    "cp": frozenset({"-f", "--force"}),
    "mv": frozenset({"-f", "--force"}),
}

# Shell metacharacters that indicate command chaining/injection
# These are checked BEFORE shlex parsing to catch obvious injection attempts
SHELL_METACHARS: FrozenSet[str] = frozenset({
    "|",   # Pipe
    ";",   # Command separator
    "&",   # Background/AND
    "`",   # Command substitution
    ">>",  # Append redirect
    ">",   # Output redirect (single)
    "<",   # Input redirect
    "<<",  # Here doc
})

# These are only dangerous at the start of tokens (after parsing)
TOKEN_METACHARS: FrozenSet[str] = frozenset({
    "$(",  # Command substitution (in parsed tokens)
})


def _get_binary_name(token: str) -> str:
    """Extract the binary name from a command token."""
    # Handle full paths like /usr/bin/rm
    path = Path(token)
    return path.name.lower()


def _check_path_escape(token: str, workspace_root: Path) -> bool:
    """Check if a path token would escape the workspace."""
    # Skip flags
    if token.startswith("-"):
        return False

    # Check for explicit traversal
    if ".." in token:
        return True

    # Check absolute paths
    try:
        candidate = Path(token).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
            workspace_resolved = workspace_root.resolve()
            if not str(resolved).startswith(str(workspace_resolved)):
                return True
    except Exception:
        pass  # Non-path arguments are fine

    return False


def _check_shell_metachars_in_tokens(tokens: list[str]) -> str | None:
    """Check for shell metacharacters in parsed tokens.

    After shlex parsing, metacharacters should only appear inside
    quoted strings (which become single tokens). We check each token
    for dangerous patterns.

    Returns the offending metachar if found, None otherwise.
    """
    for token in tokens:
        # Check if any token starts with command substitution
        for meta in TOKEN_METACHARS:
            if token.startswith(meta):
                return meta

    return None


def validate_command(
    command: str,
    workspace_root: Path,
    *,
    strict_git: bool = True,
) -> tuple[CommandVerdict, str]:
    """
    Validate a command using tokenized analysis (pre-flight).

    This function parses the command into tokens and validates each
    component, preventing common bypass techniques like:
    - Full path binaries (/bin/rm)
    - Interpreter wrapping (python -c "...")
    - Shell metacharacters (cmd1; cmd2)
    - Path traversal (cat ../../etc/passwd)

    Args:
        command: The command string to validate
        workspace_root: The workspace root directory (paths must stay within)
        strict_git: If True, validate git subcommands strictly

    Returns:
        Tuple of (verdict, reason) where reason explains the decision
    """
    # Handle empty or whitespace-only commands
    command = command.strip()
    if not command:
        return CommandVerdict.BLOCKED_FORBIDDEN, "Empty command"

    # Check for obvious metacharacters before parsing
    for meta in SHELL_METACHARS:
        if meta in command:
            return CommandVerdict.BLOCKED_METACHAR, f"Shell metacharacter detected: {meta!r}"

    # Tokenize using shell parsing rules
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return CommandVerdict.BLOCKED_UNPARSEABLE, f"Unparseable command: {e}"

    if not tokens:
        return CommandVerdict.BLOCKED_FORBIDDEN, "Empty command after parsing"

    # Get the binary name (strip any path prefix)
    binary = _get_binary_name(tokens[0])

    # Check forbidden list first (highest priority)
    if binary in FORBIDDEN_BINARIES:
        return CommandVerdict.BLOCKED_FORBIDDEN, f"Forbidden binary: {binary}"

    # Check allowlist
    if binary not in ALLOWED_BINARIES:
        return CommandVerdict.BLOCKED_NOT_ALLOWLISTED, f"Binary not in allowlist: {binary}"

    # Special handling for git - validate subcommand
    if binary == "git" and strict_git:
        if len(tokens) < 2:
            return CommandVerdict.ALLOWED, "OK (git with no subcommand)"

        subcommand = tokens[1].lower().lstrip("-")

        # Check forbidden first
        if subcommand in FORBIDDEN_GIT_SUBCOMMANDS:
            return CommandVerdict.BLOCKED_FORBIDDEN, f"Git subcommand not allowed: {subcommand}"

        # Check allowed
        if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
            return CommandVerdict.BLOCKED_NOT_ALLOWLISTED, f"Git subcommand not in allowlist: {subcommand}"

    # Check for dangerous flags (universal)
    for token in tokens[1:]:
        if token in DANGEROUS_FLAGS:
            return CommandVerdict.BLOCKED_DANGEROUS_FLAG, f"Dangerous flag: {token}"

    # Check for context-dependent dangerous flags
    context_flags = CONTEXT_DANGEROUS_FLAGS.get(binary, frozenset())
    for token in tokens[1:]:
        if token in context_flags:
            return CommandVerdict.BLOCKED_DANGEROUS_FLAG, f"Dangerous flag for {binary}: {token}"

    # Validate all path-like arguments stay within workspace
    for token in tokens[1:]:
        if _check_path_escape(token, workspace_root):
            return CommandVerdict.BLOCKED_PATH_ESCAPE, f"Path escapes workspace: {token}"

    # Final check for command substitution patterns in tokens
    found_meta = _check_shell_metachars_in_tokens(tokens)
    if found_meta:
        return CommandVerdict.BLOCKED_METACHAR, f"Shell metacharacter in token: {found_meta!r}"

    return CommandVerdict.ALLOWED, "OK"


def is_command_safe(command: str, workspace_root: Path) -> bool:
    """Convenience function returning True if command is safe."""
    verdict, _ = validate_command(command, workspace_root)
    return verdict == CommandVerdict.ALLOWED


__all__ = [
    "CommandVerdict",
    "validate_command",
    "is_command_safe",
    "FORBIDDEN_BINARIES",
    "ALLOWED_BINARIES",
    "ALLOWED_GIT_SUBCOMMANDS",
    "FORBIDDEN_GIT_SUBCOMMANDS",
]
