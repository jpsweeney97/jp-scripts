from __future__ import annotations

import http.server
import shutil
import socketserver
import subprocess
from dataclasses import dataclass
from pathlib import Path

import psutil

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

def find_processes(name_filter: str | None = None, port_filter: int | None = None) -> list[ProcessInfo]:
    """
    Scan for processes matching the given name substring or listening on a specific port.
    Returns a list of ProcessInfo objects.
    """
    matches: list[ProcessInfo] = []

    # We iterate once to be efficient
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            # 1. Check Port (if requested)
            if port_filter is not None:
                has_port = False
                for conn in proc.connections(kind="inet"):
                    if conn.laddr.port == port_filter or (conn.raddr and conn.raddr.port == port_filter):
                        has_port = True
                        break
                if not has_port:
                    continue

            # 2. Check Name (if requested)
            cmd = _format_cmdline(proc)
            if name_filter and name_filter.lower() not in cmd.lower():
                continue

            matches.append(
                ProcessInfo(
                    pid=proc.pid,
                    username=proc.username(),
                    name=proc.name(),
                    cmdline=cmd
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return sorted(matches, key=lambda p: p.pid)

def kill_process(pid: int, force: bool = False) -> str:
    """
    Kill a process by PID. Returns a status string ('killed', 'terminated', etc).
    """
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

def get_audio_devices() -> list[str]:
    """List available audio output devices via SwitchAudioSource."""
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        raise RuntimeError("SwitchAudioSource binary not found")

    proc = subprocess.run([switch_cmd, "-a"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SwitchAudioSource failed: {proc.stderr}")

    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

def set_audio_device(device_name: str) -> None:
    """Set the audio output device."""
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        raise RuntimeError("SwitchAudioSource binary not found")

    proc = subprocess.run([switch_cmd, "-s", device_name], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to switch device: {proc.stderr}")

def get_ssh_hosts(config_path: Path | None = None) -> list[str]:
    """Parse ssh config for Host entries."""
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
        # Rudimentary parsing for 'Host alias'
        if line.lower().startswith("host "):
            entries = line.split()[1:]
            hosts.extend([h for h in entries if h != "*"])

    return sorted(hosts)

def run_temp_server(directory: Path, port: int) -> None:
    """Blocking call to run a simple HTTP server."""
    handler = http.server.SimpleHTTPRequestHandler

    # We use partial application to pass the directory
    def handler_factory(*args, **kwargs):
        return handler(*args, directory=str(directory), **kwargs)

    class _ThreadingServer(socketserver.ThreadingMixIn, http.server.ThreadingHTTPServer):
        daemon_threads = True

    with _ThreadingServer(("", port), handler_factory) as httpd:
        httpd.serve_forever()
