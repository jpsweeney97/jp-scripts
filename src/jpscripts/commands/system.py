from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

# Import the new core logic
from jpscripts.core import system as system_core
from jpscripts.core.console import console
from jpscripts.core.ui import fzf_select


def _select_process(matches: list[system_core.ProcessInfo], use_fzf: bool, prompt: str) -> int | None:
    """Helper to handle the UI selection of a process."""
    if not matches:
        console.print("[yellow]No matching processes found.[/yellow]")
        return None

    if use_fzf:
        # Format for FZF: "PID\tUSER\tCMD"
        lines = [f"{p.pid}\t{p.username}\t{p.cmdline}" for p in matches]
        selection = fzf_select(lines, prompt=prompt)
        if not isinstance(selection, str) or not selection:
            return None
        return int(selection.split("\t", 1)[0])

    # Fallback to Table
    table = Table(title="Processes", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("PID", style="cyan", no_wrap=True)
    table.add_column("User", style="white", no_wrap=True)
    table.add_column("Command", style="white")

    for proc in matches[:40]:
        table.add_row(str(proc.pid), proc.username, proc.cmdline)

    console.print(table)
    console.print(Panel("Re-run with fzf installed for interactive selection.", style="yellow"))
    return None


def process_kill(
    name: str = typer.Option("", "--name", "-n", help="Filter processes containing this substring."),
    port: int | None = typer.Option(None, "--port", "-p", help="Filter processes listening on a port."),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interactively select and kill a process."""
    # LOGIC: Delegate to core
    matches = system_core.find_processes(name_filter=name, port_filter=port)

    use_fzf = shutil.which("fzf") and not no_fzf
    pid = _select_process(matches, use_fzf, prompt="kill> ")

    if pid:
        # ACTION: Delegate to core
        result = system_core.kill_process(pid, force)
        color = "green" if result in ("killed", "terminated") else "red"
        console.print(f"[{color}]{result}[/{color}] process {pid}")


def port_kill(
    port: int = typer.Argument(..., help="Port to search for."),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Find processes bound to a port and kill one."""
    # LOGIC: Delegate to core
    matches = system_core.find_processes(port_filter=port)

    use_fzf = shutil.which("fzf") and not no_fzf
    pid = _select_process(matches, use_fzf, prompt=f"port-kill ({port})> ")

    if pid:
        # ACTION: Delegate to core
        result = system_core.kill_process(pid, force)
        color = "green" if result in ("killed", "terminated") else "red"
        console.print(f"[{color}]{result}[/{color}] process {pid}")


def audioswap(no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available.")) -> None:
    """Switch audio output device using SwitchAudioSource."""
    try:
        devices = system_core.get_audio_devices()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if not devices:
        console.print("[yellow]No audio devices found.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = fzf_select(devices, prompt="audio> ") if use_fzf else devices[0]
    target = selection if isinstance(selection, str) else None

    if not target:
        return

    try:
        system_core.set_audio_device(target)
        console.print(f"[green]Switched to[/green] {target}")
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")


def ssh_open(
    host: str | None = typer.Option(None, "--host", "-h", help="Host alias to connect to. If omitted, opens fzf picker."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-pick an SSH host from ~/.ssh/config and connect."""
    hosts = system_core.get_ssh_hosts()

    if not hosts:
        console.print("[yellow]No host entries found in ~/.ssh/config.[/yellow]")
        return

    if host and host not in hosts:
        console.print(f"[red]Host {host} not found in ~/.ssh/config[/red]")
        raise typer.Exit(code=1)

    use_fzf = shutil.which("fzf") and not no_fzf
    target = host
    if not target:
        selection = fzf_select(hosts, prompt="ssh> ") if use_fzf else hosts[0]
        target = selection if isinstance(selection, str) else None

    if not target:
        return

    console.print(f"[green]Connecting to[/green] {target} ...")
    # NOTE: We keep subprocess here because ssh requires taking over the terminal TTY
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

    console.print(f"[green]Serving {directory} on port {port}[/green]")

    def serve_wrapper() -> None:
        system_core.run_temp_server(directory, port)

    thread = threading.Thread(target=serve_wrapper, daemon=True)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping server...[/yellow]")


def brew_explorer(
    query: str = typer.Option("", "--query", "-q", help="Optional search term for brew search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Search brew formulas/casks and show info."""
    # NOTE: This logic remains here for now as core/system.py does not yet implement brew support.
    # Ideally, move this to system_core.search_brew(query) in the future.

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
        selection = fzf_select(items, prompt="brew> ")
    else:
        table = Table(title="brew search", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Name", style="cyan")
        for item in items[:30]:
            table.add_row(item)
        console.print(table)
        console.print(Panel("fzf not available; re-run with fzf for interactive selection.", style="yellow"))
        return

    if not isinstance(selection, str) or not selection:
        return

    info = subprocess.run(["brew", "info", selection], capture_output=True, text=True)
    if info.stdout:
        console.print(Panel(info.stdout.strip(), title=f"brew info {selection}", box=box.SIMPLE))
    if info.returncode != 0:
        console.print(f"[red]brew info failed:[/red] {info.stderr}")
