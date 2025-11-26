from __future__ import annotations

import asyncio
import functools
import http.server
import shutil
import socketserver
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil  # type: ignore[import-untyped]
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

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

async def find_processes(name_filter: str | None = None, port_filter: int | None = None) -> list[ProcessInfo]:
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

    return await asyncio.to_thread(_collect)


async def kill_process_async(pid: int, force: bool = False, config: AppConfig | None = None) -> str:
    dry_run = config.dry_run if config is not None else False
    if dry_run:
        logger.info("Did not kill PID %s (dry-run)", pid)
        return "dry-run"

    def _terminate() -> str:
        try:
            p = psutil.Process(pid)
            if force:
                p.kill()
                return "killed"
            p.terminate()
            return "terminated"
        except psutil.NoSuchProcess:
            return "not found"
        except psutil.AccessDenied:
            return "permission denied"

    return await asyncio.to_thread(_terminate)


def kill_process(pid: int, force: bool = False, config: AppConfig | None = None) -> str:
    return asyncio.run(kill_process_async(pid, force, config))


def get_audio_devices() -> list[str]:
    # ... (Keep existing implementation) ...
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        raise RuntimeError("SwitchAudioSource binary not found")

    proc = subprocess.run([switch_cmd, "-a"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SwitchAudioSource failed: {proc.stderr}")

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

def set_audio_device(device_name: str) -> None:
    # ... (Keep existing implementation) ...
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        raise RuntimeError("SwitchAudioSource binary not found")

    proc = subprocess.run([switch_cmd, "-s", device_name], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to switch device: {proc.stderr}")

def get_ssh_hosts(config_path: Path | None = None) -> list[str]:
    # ... (Keep existing implementation) ...
    target = config_path or Path.home() / ".ssh" / "config"
    if not target.exists():
        return []

    hosts: list[str] = []
    try:
        content = target.read_text(encoding="utf-8")
    except OSError:
        return []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("host "):
            entries = line.split()[1:]
            hosts.extend([h for h in entries if h != "*"])

    return sorted(hosts)

def run_temp_server(directory: Path, port: int) -> None:
    # ... (Keep existing implementation) ...
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("", port), handler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

# --- Async Homebrew Wrappers ---

async def search_brew(query: str | None) -> list[str]:
    """Async wrapper for `brew search`."""
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError("Homebrew is required.")

    args = [brew, "search"]
    if query:
        args.append(query)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"brew search failed: {stderr.decode().strip()}")

    return [line.strip() for line in stdout.decode().splitlines() if line.strip()]

async def get_brew_info(name: str) -> str:
    """Async wrapper for `brew info`."""
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError("Homebrew is required.")

    proc = await asyncio.create_subprocess_exec(
        brew, "info", name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(f"brew info failed: {stderr.decode().strip()}")

    return stdout.decode().strip()
