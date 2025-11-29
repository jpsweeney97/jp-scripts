from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Deque
from collections import deque
from typing import TYPE_CHECKING

import typer
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.console import console, get_logger
from jpscripts.core.config import AppConfig
from jpscripts.core.memory import save_memory
from jpscripts.core.result import JPScriptsError
from jpscripts.core.security import validate_path, validate_workspace_root

app = typer.Typer(help="Watch the workspace and trigger JIT maintenance.")

logger = get_logger(__name__)
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".exe", ".zip", ".tar", ".gz"}

if TYPE_CHECKING:
    from jpscripts.main import AppState


@dataclass
class WatchEvent:
    path: Path
    event_type: str
    status: str = "pending"
    message: str = ""


class _AsyncDispatchHandler(FileSystemEventHandler):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        event_queue: "queue.Queue[Path]",
        root: Path,
        ignore_dirs: set[str],
    ) -> None:
        super().__init__()
        self._loop = loop
        self._queue = event_queue
        self._root = root
        self._ignore_dirs = ignore_dirs

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        try:
            safe_path = validate_path(path, self._root)
        except Exception:
            return
        if self._should_ignore(safe_path):
            return
        try:
            self._queue.put_nowait(safe_path)
        except queue.Full:
            logger.debug("Watch queue is full; dropping event for %s", safe_path)

    def _should_ignore(self, path: Path) -> bool:
        try:
            rel = path.relative_to(self._root)
        except ValueError:
            return True
        return any(part in self._ignore_dirs for part in rel.parts)


async def _run_ruff_syntax(path: Path, root: Path) -> WatchEvent:
    command = f"ruff check --select E9,F821 {path}"
    verdict, reason = validate_command(command, root)
    if verdict != CommandVerdict.ALLOWED:
        return WatchEvent(path=path, event_type="ruff", status="blocked", message=reason)

    proc = await asyncio.create_subprocess_exec(
        "ruff",
        "check",
        "--select",
        "E9,F821",
        str(path),
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        return WatchEvent(path=path, event_type="ruff", status="ok", message=stdout.decode().strip())
    return WatchEvent(
        path=path,
        event_type="ruff",
        status="error",
        message=stderr.decode(errors="replace") or stdout.decode(errors="replace"),
    )


async def _update_memory_for_file(path: Path, config: AppConfig) -> WatchEvent:
    try:
        content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return WatchEvent(path=path, event_type="memory", status="skipped", message=str(exc))

    try:
        await asyncio.to_thread(save_memory, content, ["watcher"], config=config, source_path=str(path))
    except JPScriptsError as exc:
        return WatchEvent(path=path, event_type="memory", status="error", message=str(exc))
    return WatchEvent(path=path, event_type="memory", status="ok", message="Embeddings refreshed")


def _render_dashboard(events: Deque[WatchEvent]) -> Panel:
    table = Table(title="Recent Events", expand=True)
    table.add_column("Path", overflow="fold")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Message", overflow="fold")

    for event in list(events)[-10:]:
        table.add_row(str(event.path), event.event_type, event.status, event.message)

    return Panel(
        Group(
            Panel("God-Mode Active • Watching workspace for changes", style="bold cyan"),
            table,
        ),
        title="jp watch",
        border_style="magenta",
    )


async def _watch_loop(state: "AppState", debounce_seconds: float = 5.0) -> None:
    try:
        root = await asyncio.to_thread(validate_workspace_root, state.config.workspace_root or state.config.notes_dir)
    except Exception as exc:
        console.print(f"[red]Workspace validation failed:[/red] {exc}")
        return
    loop = asyncio.get_running_loop()
    event_queue: "queue.Queue[Path]" = queue.Queue(maxsize=512)
    events: Deque[WatchEvent] = deque(maxlen=50)
    pending_memory: dict[Path, asyncio.Task[WatchEvent]] = {}

    handler = _AsyncDispatchHandler(loop, event_queue, root, set(state.config.ignore_dirs))
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer_thread = threading.Thread(target=observer.start, daemon=True)
    observer_thread.start()

    try:
        with Live(_render_dashboard(events), console=console, refresh_per_second=4) as live:
            def _append_event(task: asyncio.Task[WatchEvent]) -> None:
                try:
                    events.append(task.result())
                except asyncio.CancelledError:
                    return
                except Exception as exc:  # pragma: no cover - defensive
                    events.append(
                        WatchEvent(path=root, event_type="internal", status="error", message=str(exc))
                    )
                live.update(_render_dashboard(events))

            while True:
                try:
                    path = event_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.25)
                    continue

                suffix = path.suffix.lower()

                if suffix == ".py":
                    task = asyncio.create_task(_run_ruff_syntax(path, root))
                    task.add_done_callback(_append_event)

                if suffix not in BINARY_SUFFIXES:
                    existing = pending_memory.get(path)
                    if existing and not existing.done():
                        existing.cancel()

                    async def _debounced_memory(p: Path) -> WatchEvent:
                        await asyncio.sleep(debounce_seconds)
                        return await _update_memory_for_file(p, state.config)

                    mem_task = asyncio.create_task(_debounced_memory(path))
                    mem_task.add_done_callback(_append_event)
                    pending_memory[path] = mem_task

                # Clean up finished memory tasks to prevent growth
                for mem_path, mem_task in list(pending_memory.items()):
                    if mem_task.done():
                        pending_memory.pop(mem_path, None)

                live.update(_render_dashboard(events))
    except KeyboardInterrupt:
        console.print("[yellow]Stopping watcher...[/yellow]")
    finally:
        observer.stop()
        observer.join(timeout=2)


@app.command("watch")
def watch(ctx: typer.Context) -> None:
    """Run a God-Mode file watcher that triggers syntax checks and memory updates."""
    state = ctx.obj
    asyncio.run(_watch_loop(state))
