from __future__ import annotations

import asyncio
import functools
import http.server
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import psutil
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

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


def _format_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()) or proc.name()
    except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess):
        return proc.name()


async def find_processes(name_filter: str | None = None, port_filter: int | None = None) -> Result[list[ProcessInfo], SystemResourceError]:
    def _collect() -> list[ProcessInfo]:
        matches: list[ProcessInfo] = []
        for proc in psutil.process_iter(["pid", "name", "username"]):
            try:
                if port_filter is not None:
                    has_port = False
                    for conn in proc.connections(kind="inet"):
                        if conn.laddr.port == port_filter or (conn.raddr and conn.raddr.port == port_filter):
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
        return Err(SystemResourceError("Failed to enumerate processes", context={"error": str(exc)}))

    return Ok(matches)


async def kill_process_async(pid: int, force: bool = False, config: AppConfig | None = None) -> Result[str, SystemResourceError]:
    dry_run = config.dry_run if config is not None else False
    if dry_run:
        logger.info("Did not kill PID %s (dry-run)", pid)
        return Ok("dry-run")

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
            return Err(SystemResourceError("Permission denied to kill process", context={"pid": pid}))
        except Exception as exc:
            return Err(SystemResourceError("Failed to kill process", context={"pid": pid, "error": str(exc)}))

    return await asyncio.to_thread(_terminate)


def kill_process(pid: int, force: bool = False, config: AppConfig | None = None) -> Result[str, SystemResourceError]:
    return asyncio.run(kill_process_async(pid, force, config))


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
        return Err(SystemResourceError("Failed to switch audio device", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode().strip() or stdout.decode().strip()
        return Err(SystemResourceError("Failed to switch device", context={"stderr": message, "device": device_name}))

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
        return Err(SystemResourceError("Failed to read SSH config", context={"path": str(target), "error": str(exc)}))

    return Ok(hosts)


async def run_temp_server(directory: Path, port: int) -> Result[None, SystemResourceError]:
    if not directory.is_dir():
        return Err(SystemResourceError("Serve directory is not a folder", context={"directory": str(directory)}))

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
        return Err(SystemResourceError("Failed to start HTTP server", context={"directory": str(directory), "error": str(exc)}))

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
        return Err(SystemResourceError("brew search failed", context={"stderr": stderr.decode().strip()}))

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
        return Err(SystemResourceError("brew info failed", context={"stderr": stderr.decode().strip()}))

    return Ok(stdout.decode().strip())
