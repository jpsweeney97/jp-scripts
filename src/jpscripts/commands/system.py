from __future__ import annotations

import asyncio
import shutil
import signal
import sys
from pathlib import Path
from typing import TypeVar

import psutil
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.commands.ui import fzf_select_async
from jpscripts.core import git as git_core
from jpscripts.core import system as system_core
from jpscripts.core.console import console
from jpscripts.core.result import Err, JPScriptsError, Ok, Result, SystemResourceError

T = TypeVar("T")


def _fzf_select(lines: list[str], prompt: str) -> str | list[str] | None:
    """Run fzf selection without blocking the main thread."""
    return asyncio.run(fzf_select_async(lines, prompt=prompt))


def _select_process(
    matches: list[system_core.ProcessInfo], use_fzf: bool, prompt: str
) -> int | None:
    """Helper to handle the UI selection of a process."""
    if not matches:
        console.print("[yellow]No matching processes found.[/yellow]")
        return None

    if use_fzf:
        # Format for FZF: "PID\tUSER\tCMD"
        lines = [f"{p.pid}\t{p.username}\t{p.cmdline}" for p in matches]
        selection = _fzf_select(lines, prompt=prompt)
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


def _unwrap_result(result: Result[T, JPScriptsError] | Result[T, SystemResourceError]) -> T:
    match result:
        case Ok(value):
            return value
        case Err(err):
            message = err.message if hasattr(err, "message") else str(err)
            console.print(f"[red]{message}[/red]")
            raise typer.Exit(code=1)


def process_kill(
    ctx: typer.Context,
    name: str = typer.Option(
        "", "--name", "-n", help="Filter processes containing this substring."
    ),
    port: int | None = typer.Option(
        None, "--port", "-p", help="Filter processes listening on a port."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interactively select and kill a process."""
    # LOGIC: Delegate to core
    matches = _unwrap_result(
        asyncio.run(system_core.find_processes(name_filter=name, port_filter=port))
    )

    use_fzf = bool(shutil.which("fzf")) and not no_fzf
    pid = _select_process(matches, use_fzf, prompt="kill> ")

    if pid:
        # ACTION: Delegate to core
        result = _unwrap_result(asyncio.run(system_core.kill_process_async(pid, force)))
        color = "green" if result in ("killed", "terminated") else "red"
        console.print(f"[{color}]{result}[/{color}] process {pid}")


def port_kill(
    ctx: typer.Context,
    port: int = typer.Argument(..., help="Port to search for."),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill (SIGKILL)."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Find processes bound to a port and kill one."""
    # LOGIC: Delegate to core
    matches = _unwrap_result(asyncio.run(system_core.find_processes(port_filter=port)))

    use_fzf = bool(shutil.which("fzf")) and not no_fzf
    pid = _select_process(matches, use_fzf, prompt=f"port-kill ({port})> ")

    if pid:
        # ACTION: Delegate to core
        result = _unwrap_result(asyncio.run(system_core.kill_process_async(pid, force)))
        color = "green" if result in ("killed", "terminated") else "red"
        console.print(f"[{color}]{result}[/{color}] process {pid}")


def audioswap(
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Switch audio output device using SwitchAudioSource."""
    devices = _unwrap_result(asyncio.run(system_core.get_audio_devices()))

    if not devices:
        console.print("[yellow]No audio devices found.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = _fzf_select(devices, prompt="audio> ") if use_fzf else devices[0]
    target = selection if isinstance(selection, str) else None

    if not target:
        return

    _unwrap_result(asyncio.run(system_core.set_audio_device(target)))
    console.print(f"[green]Switched to[/green] {target}")


def ssh_open(
    host: str | None = typer.Option(
        None, "--host", "-h", help="Host alias to connect to. If omitted, opens fzf picker."
    ),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Fuzzy-pick an SSH host from ~/.ssh/config and connect."""
    hosts = _unwrap_result(asyncio.run(system_core.get_ssh_hosts()))

    if not hosts:
        console.print("[yellow]No host entries found in ~/.ssh/config.[/yellow]")
        return

    if host and host not in hosts:
        console.print(f"[red]Host {host} not found in ~/.ssh/config[/red]")
        raise typer.Exit(code=1)

    use_fzf = shutil.which("fzf") and not no_fzf
    target = host
    if not target:
        selection = _fzf_select(hosts, prompt="ssh> ") if use_fzf else hosts[0]
        target = selection if isinstance(selection, str) else None

    if not target:
        return

    console.print(f"[green]Connecting to[/green] {target} ...")
    if not shutil.which("ssh"):
        console.print("[red]ssh binary not found on PATH.[/red]")
        raise typer.Exit(code=1)
    try:
        exit_code = asyncio.run(_run_ssh(target))
        if exit_code != 0:
            console.print(f"[red]ssh exited with code {exit_code}[/red]")
    except FileNotFoundError:
        console.print("[red]ssh binary not found on PATH.[/red]")
        raise typer.Exit(code=1)


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

    try:
        _unwrap_result(asyncio.run(system_core.run_temp_server(directory, port)))
    except KeyboardInterrupt:
        console.print("[yellow]Stopping server...[/yellow]")


def brew_explorer(
    query: str = typer.Option("", "--query", "-q", help="Optional search term for brew search."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Search brew formulas/casks and show info."""

    async def run_search() -> list[str]:
        with console.status("Searching Homebrew...", spinner="dots"):
            match await system_core.search_brew(query):
                case Err(err):
                    console.print(f"[red]{err.message}[/red]")
                    raise typer.Exit(code=1)
                case Ok(items):
                    return items

    items = asyncio.run(run_search())

    if not items:
        console.print("[yellow]No results from brew search.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = None
    if use_fzf:
        selection = _fzf_select(items, prompt="brew> ")
    else:
        table = Table(title="brew search", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Name", style="cyan")
        for item in items[:30]:
            table.add_row(item)
        console.print(table)
        console.print(
            Panel("fzf not available; re-run with fzf for interactive selection.", style="yellow")
        )
        return

    if not isinstance(selection, str) or not selection:
        return

    async def run_info() -> str | None:
        with console.status(f"Fetching info for {selection}...", spinner="dots"):
            match await system_core.get_brew_info(selection):
                case Err(err):
                    console.print(f"[red]{err.message}[/red]")
                    raise typer.Exit(code=1)
                case Ok(info):
                    return info

    info = asyncio.run(run_info())
    if info:
        console.print(Panel(info, title=f"brew info {selection}", box=box.SIMPLE))


def update() -> None:
    """Update jpscripts in editable installs, or guide pipx users."""
    project_root = Path(__file__).resolve().parents[3]
    src_path = project_root / "src" / "jpscripts"

    if not src_path.exists():
        console.print(
            "[yellow]Detected pipx or wheel install. Run `pipx upgrade jpscripts`.[/yellow]"
        )
        return

    async def _run_update() -> None:
        try:
            with console.status("Upgrading God-Mode...", spinner="dots"):
                match await git_core.AsyncRepo.open(project_root):
                    case Err(err):
                        raise RuntimeError(err.message)
                    case Ok(repo):
                        match await repo._run_git("pull"):
                            case Err(err):
                                raise RuntimeError(err.message)
                            case Ok(_):
                                pass

                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    ".",
                    cwd=str(project_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    message = stderr.decode("utf-8", errors="replace") or stdout.decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(message or "pip install failed")
        except Exception as exc:
            console.print(f"[red]Update failed: {exc}[/red]")
            raise typer.Exit(code=1)

        console.print("[green]Update complete.[/green]")

    asyncio.run(_run_update())


async def _run_ssh(target: str) -> int:
    """Run ssh while preserving TTY control using asyncio."""
    proc = await asyncio.create_subprocess_exec("ssh", target)
    return await proc.wait()


def panic(
    ctx: typer.Context,
    hard: bool = typer.Option(False, "--hard", help="Also reset git to HEAD"),
) -> None:
    """Emergency kill switch for runaway agent processes.

    Terminates all codex and MCP processes system-wide.
    Use --hard to also reset git workspace to HEAD.
    """
    killed_count = 0
    errors: list[str] = []

    console.print("[bold red]🚨 PANIC PROTOCOL ENGAGED[/bold red]")

    # Find and kill codex processes
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            proc_name = proc.info.get("name", "") or ""
            proc_cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(proc_cmdline) if proc_cmdline else ""

            # Kill codex processes
            if proc_name.lower() == "codex":
                console.print(f"[yellow]Terminating codex process: PID {proc.pid}[/yellow]")
                proc.send_signal(signal.SIGTERM)
                killed_count += 1

            # Kill MCP processes (Model Context Protocol servers)
            elif "mcp" in cmdline_str.lower():
                console.print(f"[yellow]Terminating MCP process: PID {proc.pid}[/yellow]")
                proc.send_signal(signal.SIGTERM)
                killed_count += 1

        except psutil.NoSuchProcess:
            # Process already terminated
            pass
        except psutil.AccessDenied:
            errors.append(f"Access denied for PID {proc.pid}")
        except Exception as exc:
            errors.append(f"Error terminating PID {proc.pid}: {exc}")

    console.print(f"[green]Terminated {killed_count} agent processes.[/green]")

    if errors:
        for err in errors:
            console.print(f"[dim]{err}[/dim]")

    # Hard reset if requested
    if hard:
        console.print("[yellow]Executing git reset --hard HEAD...[/yellow]")
        try:
            result = asyncio.run(_run_git_reset_hard())
            if result == 0:
                console.print("[green]Workspace sanitized.[/green]")
            else:
                console.print("[red]git reset failed.[/red]")
        except Exception as exc:
            console.print(f"[red]git reset failed: {exc}[/red]")

    console.print("[bold green]🚨 PANIC PROTOCOL COMPLETE.[/bold green]")


async def _run_git_reset_hard() -> int:
    """Execute git reset --hard HEAD."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "reset",
        "--hard",
        "HEAD",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode or 0
