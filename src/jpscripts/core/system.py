"""System utilities and process management.

Provides low-level system operations:
    - Process discovery and management
    - Port-to-process mapping
    - Safe shell command execution
    - HTTP server utilities
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import os
import shlex
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError
from jpscripts.core.runtime import get_runtime

logger = get_logger(__name__)


@dataclass
class ProcessInfo:
    pid: int
    username: str
    name: str
    cmdline: str

    @property
    def label(self) -> str:
        return f"{self.pid} - {self.name} ({self.username})"


@dataclass(slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SandboxProtocol(Protocol):
    async def run_command(
        self,
        tokens: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> Result[CommandResult, SystemResourceError]: ...


class LocalSandbox:
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
    if config and config.infra.use_docker_sandbox:
        return DockerSandbox(config.infra.docker_image, config.user.workspace_root)
    return LocalSandbox()


def _format_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()) or proc.name()
    except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
        return proc.name()


async def find_processes(
    name_filter: str | None = None, port_filter: int | None = None
) -> Result[list[ProcessInfo], SystemResourceError]:
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
    return asyncio.run(kill_process_async(pid, force))


async def run_safe_shell(
    command: str,
    root: Path,
    audit_prefix: str,
    config: AppConfig | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Result[CommandResult, SystemResourceError]:
    """Validate and execute a shell command asynchronously using the sandbox."""
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


async def get_audio_devices() -> Result[list[str], SystemResourceError]:
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))

    try:
        proc = await asyncio.create_subprocess_exec(
            switch_cmd,
            "-a",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))
    except Exception as exc:
        return Err(SystemResourceError("Failed to list audio devices", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return Err(
            SystemResourceError(
                "SwitchAudioSource failed",
                context={"stderr": stderr.decode().strip(), "returncode": proc.returncode},
            )
        )

    return Ok([line.strip() for line in stdout.decode().splitlines() if line.strip()])


async def set_audio_device(device_name: str) -> Result[None, SystemResourceError]:
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))

    try:
        proc = await asyncio.create_subprocess_exec(
            switch_cmd,
            "-s",
            device_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("SwitchAudioSource binary not found"))
    except Exception as exc:
        return Err(
            SystemResourceError("Failed to switch audio device", context={"error": str(exc)})
        )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode().strip() or stdout.decode().strip()
        return Err(
            SystemResourceError(
                "Failed to switch device", context={"stderr": message, "device": device_name}
            )
        )

    return Ok(None)


async def get_ssh_hosts(config_path: Path | None = None) -> Result[list[str], SystemResourceError]:
    target = config_path or Path.home() / ".ssh" / "config"
    if not target.exists():
        return Ok([])

    def _read_hosts() -> list[str]:
        hosts: list[str] = []
        content = target.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("host "):
                entries = line.split()[1:]
                hosts.extend([h for h in entries if h != "*"])
        return sorted(hosts)

    try:
        hosts = await asyncio.to_thread(_read_hosts)
    except OSError as exc:
        return Err(
            SystemResourceError(
                "Failed to read SSH config", context={"path": str(target), "error": str(exc)}
            )
        )

    return Ok(hosts)


async def run_temp_server(directory: Path, port: int) -> Result[None, SystemResourceError]:
    if not directory.is_dir():
        return Err(
            SystemResourceError(
                "Serve directory is not a folder", context={"directory": str(directory)}
            )
        )

    def _serve() -> None:
        handler: Callable[..., http.server.SimpleHTTPRequestHandler] = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(directory)
        )
        httpd = http.server.ThreadingHTTPServer(("", port), handler)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()

    try:
        await asyncio.to_thread(_serve)
    except OSError as exc:
        return Err(
            SystemResourceError(
                "Failed to start HTTP server",
                context={"directory": str(directory), "error": str(exc)},
            )
        )

    return Ok(None)


async def search_brew(query: str | None) -> Result[list[str], SystemResourceError]:
    """Async wrapper for `brew search`. Executes command without shell interpolation."""
    brew = shutil.which("brew")
    if not brew:
        return Err(SystemResourceError("Homebrew is required."))

    args = [brew, "search"]
    if query:
        args.append(query)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("Homebrew is required."))
    except Exception as exc:
        return Err(SystemResourceError("Failed to start brew search", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return Err(
            SystemResourceError("brew search failed", context={"stderr": stderr.decode().strip()})
        )

    return Ok([line.strip() for line in stdout.decode().splitlines() if line.strip()])


async def get_brew_info(name: str) -> Result[str, SystemResourceError]:
    """Async wrapper for `brew info`. Executes command without shell interpolation."""
    brew = shutil.which("brew")
    if not brew:
        return Err(SystemResourceError("Homebrew is required."))

    try:
        proc = await asyncio.create_subprocess_exec(
            brew,
            "info",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("Homebrew is required."))
    except Exception as exc:
        return Err(SystemResourceError("Failed to start brew info", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return Err(
            SystemResourceError("brew info failed", context={"stderr": stderr.decode().strip()})
        )

    return Ok(stdout.decode().strip())
