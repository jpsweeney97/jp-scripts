"""Command execution and sandboxing utilities.

Provides:
- SandboxProtocol for command isolation
- LocalSandbox and DockerSandbox implementations
- run_safe_shell for validated command execution
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)


@dataclass(slots=True)
class CommandResult:
    """Result of a shell command execution."""

    returncode: int
    stdout: str
    stderr: str


class SandboxProtocol(Protocol):
    """Protocol for command execution sandboxes."""

    async def run_command(
        self,
        tokens: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> Result[CommandResult, SystemResourceError]: ...


class LocalSandbox:
    """Execute commands directly on the local system."""

    async def run_command(
        self,
        tokens: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> Result[CommandResult, SystemResourceError]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *tokens,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            return Err(
                SystemResourceError("Command not found", context={"error": str(exc), "cmd": tokens})
            )
        except Exception as exc:  # pragma: no cover - defensive
            return Err(
                SystemResourceError(
                    "Failed to start command", context={"error": str(exc), "cmd": tokens}
                )
            )

        stdout_bytes, stderr_bytes = await proc.communicate()
        return Ok(
            CommandResult(
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode(errors="replace"),
                stderr=stderr_bytes.decode(errors="replace"),
            )
        )


class DockerSandbox:
    """Execute commands in a Docker container."""

    def __init__(self, image: str, workspace_root: Path) -> None:
        self.image = image
        self.workspace_root = workspace_root

    async def run_command(
        self,
        tokens: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> Result[CommandResult, SystemResourceError]:
        docker_binary = shutil.which("docker")
        if not docker_binary:
            return Err(
                SystemResourceError("Docker binary not found", context={"image": self.image})
            )

        mount_root = self.workspace_root.expanduser().resolve()
        workdir = "/workspace"
        try:
            rel = cwd.resolve().relative_to(mount_root)
            workdir = str(Path("/workspace") / rel)
        except Exception:
            pass

        docker_env: list[str] = []
        for key, value in (env or {}).items():
            docker_env.extend(["-e", f"{key}={value}"])

        user_flag = f"{os.getuid()}:{os.getgid()}"

        docker_cmd = [
            docker_binary,
            "run",
            "--rm",
            "-v",
            f"{mount_root}:/workspace",
            "-w",
            workdir,
            "-u",
            user_flag,
            *docker_env,
            self.image,
            *tokens,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return Err(
                SystemResourceError("Docker binary not found", context={"image": self.image})
            )
        except Exception as exc:
            return Err(
                SystemResourceError(
                    "Failed to start docker command", context={"error": str(exc), "cmd": docker_cmd}
                )
            )

        stdout_bytes, stderr_bytes = await proc.communicate()
        return Ok(
            CommandResult(
                returncode=proc.returncode or 0,
                stdout=stdout_bytes.decode(errors="replace"),
                stderr=stderr_bytes.decode(errors="replace"),
            )
        )


def get_sandbox(config: AppConfig | None) -> SandboxProtocol:
    """Get the appropriate sandbox based on configuration."""
    if config and config.infra.use_docker_sandbox:
        return DockerSandbox(config.infra.docker_image, config.user.workspace_root)
    return LocalSandbox()


async def run_safe_shell(
    command: str,
    root: Path,
    audit_prefix: str,
    config: AppConfig | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Result[CommandResult, SystemResourceError]:
    """Validate and execute a shell command asynchronously using the sandbox.

    Args:
        command: Shell command string to execute
        root: Working directory for the command
        audit_prefix: Prefix for audit logging
        config: Optional app config for sandbox settings
        env: Optional environment variables

    Returns:
        Result containing CommandResult on success, SystemResourceError on failure
    """
    verdict, reason = validate_command(command, root)
    if verdict != CommandVerdict.ALLOWED:
        return Err(
            SystemResourceError(
                "Command blocked by policy", context={"reason": reason, "command": command}
            )
        )

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return Err(
            SystemResourceError(
                "Failed to parse command", context={"command": command, "error": str(exc)}
            )
        )

    if not tokens:
        return Err(SystemResourceError("Invalid command", context={"command": command}))

    runner = get_sandbox(config)
    run_result = await runner.run_command(tokens, root, env=env)
    if isinstance(run_result, Err):
        logger.warning("%s.run_safe_shell failure: %s", audit_prefix, run_result.error)
        return run_result
    return run_result


__all__ = [
    "CommandResult",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxProtocol",
    "get_sandbox",
    "run_safe_shell",
]
