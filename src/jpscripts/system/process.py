"""Process management utilities.

Provides:
- ProcessInfo dataclass for process metadata
- find_processes for discovering running processes
- kill_process_async and kill_process for terminating processes
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import psutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError
from jpscripts.core.runtime import get_runtime

from .execution import DockerSandbox, get_sandbox

logger = get_logger(__name__)


@dataclass
class ProcessInfo:
    """Information about a running process."""

    pid: int
    username: str
    name: str
    cmdline: str

    @property
    def label(self) -> str:
        """Human-readable label for display."""
        return f"{self.pid} - {self.name} ({self.username})"


def _format_cmdline(proc: psutil.Process) -> str:
    """Format process command line, falling back to name on error."""
    try:
        return " ".join(proc.cmdline()) or proc.name()
    except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
        return proc.name()


async def find_processes(
    name_filter: str | None = None, port_filter: int | None = None
) -> Result[list[ProcessInfo], SystemResourceError]:
    """Find processes matching optional name and port filters.

    Args:
        name_filter: Optional substring to match against command line
        port_filter: Optional port number to match

    Returns:
        Result containing list of ProcessInfo on success
    """

    def _collect() -> list[ProcessInfo]:
        matches: list[ProcessInfo] = []
        for proc in psutil.process_iter(["pid", "name", "username"]):
            try:
                if port_filter is not None:
                    has_port = False
                    for conn in proc.connections(kind="inet"):
                        if conn.laddr.port == port_filter or (
                            conn.raddr and conn.raddr.port == port_filter
                        ):
                            has_port = True
                            break
                    if not has_port:
                        continue

                cmd = _format_cmdline(proc)
                if name_filter and name_filter.lower() not in cmd.lower():
                    continue

                matches.append(
                    ProcessInfo(
                        pid=proc.pid,
                        username=proc.username(),
                        name=proc.name(),
                        cmdline=cmd,
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return sorted(matches, key=lambda p: p.pid)

    try:
        matches = await asyncio.to_thread(_collect)
    except Exception as exc:
        return Err(
            SystemResourceError("Failed to enumerate processes", context={"error": str(exc)})
        )

    return Ok(matches)


async def kill_process_async(pid: int, force: bool = False) -> Result[str, SystemResourceError]:
    """Terminate a process by PID.

    Args:
        pid: Process ID to terminate
        force: If True, use SIGKILL; otherwise use SIGTERM

    Returns:
        Result containing status string on success
    """
    runtime = get_runtime()
    config = runtime.config
    dry_run = config.user.dry_run
    if dry_run:
        logger.info("Did not kill PID %s (dry-run)", pid)
        return Ok("dry-run")

    runner = get_sandbox(config)
    if isinstance(runner, DockerSandbox):
        return Err(
            SystemResourceError(
                "Killing host processes is not supported in docker sandbox mode.",
                context={"pid": pid},
            )
        )

    def _terminate() -> Result[str, SystemResourceError]:
        try:
            process = psutil.Process(pid)
            if force:
                process.kill()
                return Ok("killed")
            process.terminate()
            return Ok("terminated")
        except psutil.NoSuchProcess:
            return Err(SystemResourceError("Process not found", context={"pid": pid}))
        except psutil.AccessDenied:
            return Err(
                SystemResourceError("Permission denied to kill process", context={"pid": pid})
            )
        except Exception as exc:
            return Err(
                SystemResourceError(
                    "Failed to kill process", context={"pid": pid, "error": str(exc)}
                )
            )

    return await asyncio.to_thread(_terminate)


def kill_process(pid: int, force: bool = False) -> Result[str, SystemResourceError]:
    """Synchronous wrapper for kill_process_async."""
    return asyncio.run(kill_process_async(pid, force))


__all__ = [
    "ProcessInfo",
    "find_processes",
    "kill_process",
    "kill_process_async",
]
