from __future__ import annotations

import http.server
import shutil
import socket
import socketserver
import subprocess
import threading
from pathlib import Path
from typing import Iterable

import psutil
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console


def _run_fzf(lines: list[str], prompt: str) -> str | None:
    proc = subprocess.run(
        ["fzf", "--prompt", prompt],
        input="\n".join(lines),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _format_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()) or proc.name()
    except psutil.Error:
        return proc.name()


def _kill_process(pid: int, force: bool) -> str:
    try:
        p = psutil.Process(pid)
        p.kill() if force else p.terminate()
        return "killed" if force else "terminated"
    except psutil.NoSuchProcess:
        return "not found"
    except psutil.AccessDenied:
        return "permission denied"


def process_kill(
    name: str = typer.Option("", "--name", "-n", help="Filter processes containing this substring."),
    port: int | None = typer.Option(None, "--port", "-p", help="Filter processes listening on a port."),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interactively select and kill a process using psutil."""
    use_fzf = shutil.which("fzf") and not no_fzf

    matches: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            cmd = _format_cmdline(proc)
            if name and name.lower() not in cmd.lower():
                continue
            if port is not None:
                has_port = any(
                    conn.laddr.port == port or (conn.raddr and conn.raddr.port == port)
                    for conn in proc.connections(kind="inet")
                )
                if not has_port:
                    continue
            matches.append(proc)
        except psutil.Error:
            continue

    if not matches:
        console.print("[yellow]No matching processes found.[/yellow]")
        return

    lines = [f"{p.pid}\t{p.username()}\t{_format_cmdline(p)}" for p in matches]

    if use_fzf:
        selection = _run_fzf(lines, prompt="kill> ")
        if not selection:
            return
        pid = int(selection.split("\t", 1)[0])
        result = _kill_process(pid, force)
        console.print(f"[green]{result}[/green] process {pid}")
        return

    table = Table(title="Processes", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("PID", style="cyan", no_wrap=True)
    table.add_column("User", style="white", no_wrap=True)
    table.add_column("Command", style="white")
    for proc in matches[:40]:
        table.add_row(str(proc.pid), proc.username(), _format_cmdline(proc))
    console.print(table)
    console.print(Panel("Re-run with fzf installed for interactive selection.", style="yellow"))


def port_kill(
    port: int = typer.Argument(..., help="Port to search for."),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Find processes bound to a port and kill one."""
    use_fzf = shutil.which("fzf") and not no_fzf

    matches: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            for conn in proc.connections(kind="inet"):
                if conn.laddr.port == port or (conn.raddr and conn.raddr.port == port):
                    matches.append(proc)
                    break
        except psutil.Error:
            continue

    if not matches:
        console.print(f"[yellow]No processes found on port {port}.[/yellow]")
        return

    lines = [f"{p.pid}\t{p.username()}\t{_format_cmdline(p)}" for p in matches]
    if use_fzf:
        selection = _run_fzf(lines, prompt="port-kill> ")
        if not selection:
            return
        pid = int(selection.split("\t", 1)[0])
        result = _kill_process(pid, force)
        console.print(f"[green]{result}[/green] process {pid}")
        return

    table = Table(title=f"Processes on port {port}", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("PID", style="cyan", no_wrap=True)
    table.add_column("User", style="white", no_wrap=True)
    table.add_column("Command", style="white")
    for proc in matches[:40]:
        table.add_row(str(proc.pid), proc.username(), _format_cmdline(proc))
    console.print(table)
    console.print(Panel("Re-run with fzf installed for interactive selection.", style="yellow"))


def brew_explorer(
    query: str = typer.Option("", "--query", "-q", help="Optional search term for brew search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Search brew formulas/casks and show info."""
    if not shutil.which("brew"):
        console.print("[red]Homebrew is required for brew-explorer.[/red]")
        raise typer.Exit(code=1)

    search_cmd = ["brew", "search"]
    if query:
        search_cmd.append(query)

    search = subprocess.run(search_cmd, capture_output=True, text=True)
    if search.returncode != 0:
        console.print(f"[red]brew search failed:[/red] {search.stderr}")
        raise typer.Exit(code=1)

    items = [line.strip() for line in search.stdout.splitlines() if line.strip()]
    if not items:
        console.print("[yellow]No results from brew search.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = None
    if use_fzf:
        selection = _run_fzf(items, prompt="brew> ")
    else:
        table = Table(title="brew search", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Name", style="cyan")
        for item in items[:30]:
            table.add_row(item)
        console.print(table)
        console.print(Panel("fzf not available; re-run with fzf for interactive selection.", style="yellow"))
        return

    if not selection:
        return

    info = subprocess.run(["brew", "info", selection], capture_output=True, text=True)
    if info.stdout:
        console.print(Panel(info.stdout.strip(), title=f"brew info {selection}", box=box.SIMPLE))
    if info.returncode != 0:
        console.print(f"[red]brew info failed:[/red] {info.stderr}")


def audioswap(no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available.")) -> None:
    """Switch audio output device using SwitchAudioSource."""
    switch_cmd = shutil.which("SwitchAudioSource")
    if not switch_cmd:
        console.print("[red]SwitchAudioSource binary not found.[/red]")
        raise typer.Exit(code=1)

    list_proc = subprocess.run([switch_cmd, "-a"], capture_output=True, text=True)
    if list_proc.returncode != 0:
        console.print(f"[red]SwitchAudioSource failed:[/red] {list_proc.stderr}")
        raise typer.Exit(code=1)

    devices = [line.strip() for line in list_proc.stdout.splitlines() if line.strip()]
    if not devices:
        console.print("[yellow]No audio devices found.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    target = _run_fzf(devices, prompt="audio> ") if use_fzf else devices[0]
    if not target:
        return

    set_proc = subprocess.run([switch_cmd, "-s", target], capture_output=True, text=True)
    if set_proc.returncode == 0:
        console.print(f"[green]Switched to[/green] {target}")
    else:
        console.print(f"[red]Failed to switch device:[/red] {set_proc.stderr}")


def ssh_open(
    host: str | None = typer.Option(None, "--host", "-h", help="Host alias to connect to. If omitted, opens fzf picker."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-pick an SSH host from ~/.ssh/config and connect."""
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        console.print("[red]~/.ssh/config not found.[/red]")
        raise typer.Exit(code=1)

    hosts: list[str] = []
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("host "):
            entries = line.split()[1:]
            hosts.extend([h for h in entries if h != "*"])

    if not hosts:
        console.print("[yellow]No host entries found in ~/.ssh/config.[/yellow]")
        return

    if host and host not in hosts:
        console.print(f"[red]Host {host} not found in ~/.ssh/config[/red]")
        raise typer.Exit(code=1)

    use_fzf = shutil.which("fzf") and not no_fzf
    target = host
    if not target:
        target = _run_fzf(hosts, prompt="ssh> ") if use_fzf else hosts[0]

    if not target:
        return

    console.print(f"[green]Connecting to[/green] {target} ...")
    subprocess.run(["ssh", target])


def tmpserver(
    directory: Path = typer.Option(Path("."), "--dir", "-d", help="Directory to serve."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start a simple HTTP server."""
    directory = directory.expanduser()
    if not directory.is_dir():
        console.print(f"[red]{directory} is not a directory.[/red]")
        raise typer.Exit(code=1)

    handler = http.server.SimpleHTTPRequestHandler

    class _ThreadingServer(socketserver.ThreadingMixIn, http.server.ThreadingHTTPServer):
        daemon_threads = True

    def serve() -> None:
        handler_factory = lambda *args, **kwargs: handler(*args, directory=str(directory), **kwargs)  # noqa: E731
        with _ThreadingServer(("", port), handler_factory) as httpd:
            console.print(f"[green]Serving {directory} on port {port}[/green]")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping server...[/yellow]")
