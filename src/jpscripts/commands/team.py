from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from jpscripts.core.console import console
from jpscripts.core.team import Persona, UpdateKind, get_default_swarm, swarm_chat

if TYPE_CHECKING:
    from jpscripts.main import AppState

app = typer.Typer(help="Coordinate a swarm of specialized Codex agents.")


def _agent_panel(role: Persona, log_lines: list[str], status: str) -> Panel:
    body = Text()
    for line in log_lines[-20:]:
        body.append(line + "\n")
    if not body.plain:
        body.append("[dim]Waiting for output...[/dim]")

    title = f"{role.label} • {status}"
    return Panel(body, title=title, border_style=role.color or "white", padding=(1, 2))


def _render_layout(
    objective: str,
    roles: Iterable[Persona],
    logs: dict[Persona, list[str]],
    statuses: dict[Persona, str],
    safe_mode: bool,
) -> Layout:
    layout = Layout()
    header = Panel(
        f"[bold]Objective[/bold]: {objective}",
        subtitle="Safe Mode config inherited" if safe_mode else None,
        border_style="blue",
        padding=(1, 2),
    )
    layout.split_column(Layout(header, name="header", size=5), Layout(name="agents"))

    agent_layouts = [
        Layout(_agent_panel(role, logs[role], statuses.get(role, "…")), name=role.name.lower())
        for role in roles
    ]
    layout["agents"].split_row(*agent_layouts)
    return layout


async def _run_swarm(
    objective: str, roles: list[Persona], state: AppState, safe_mode: bool
) -> None:
    logs: dict[Persona, list[str]] = {role: [] for role in roles}
    statuses: dict[Persona, str] = dict.fromkeys(roles, "queued")

    with Live(
        _render_layout(objective, roles, logs, statuses, safe_mode),
        console=console,
        refresh_per_second=30,
    ) as live:
        async for update in swarm_chat(
            objective=objective,
            roles=roles,
            config=state.config,
            repo_root=None,
            model=state.config.default_model,
            safe_mode=safe_mode,
        ):
            if update.kind in {UpdateKind.STDOUT, UpdateKind.STDERR}:
                logs[update.role].append(update.content)
                if len(logs[update.role]) > 200:
                    logs[update.role] = logs[update.role][-200:]
            elif update.kind == UpdateKind.STATUS:
                statuses[update.role] = update.content
                logs[update.role].append(f"[status] {update.content}")
            elif update.kind == UpdateKind.EXIT:
                statuses[update.role] = update.content
                logs[update.role].append(f"[exit] {update.content}")

            live.update(_render_layout(objective, roles, logs, statuses, safe_mode))

    console.print("[green]Swarm complete.[/green]")


@app.command("swarm")
def swarm(
    ctx: typer.Context,
    objective: str = typer.Argument(..., help="Goal for the multi-agent swarm."),
) -> None:
    """
    Launch architect, engineer, and QA Codex agents in parallel.
    """
    state = ctx.obj
    roles = get_default_swarm()
    safe_mode_active = bool(getattr(state, "config_meta", None) and state.config_meta.error)

    # Pre-flight check for MCP configuration in Codex
    codex_config = Path.home() / ".codex" / "config.toml"
    if codex_config.exists():
        try:
            content = codex_config.read_text(encoding="utf-8")
            if "jpscripts" not in content.lower():
                console.print(
                    "[yellow]Warning: `jpscripts` MCP server may not be configured in Codex. Agents might lack tools. Run `codex mcp add jpscripts ...` to fix.[/yellow]"
                )
        except OSError as exc:
            console.print(f"[dim]Could not read codex config: {exc}[/dim]")
    else:
        console.print(
            "[yellow]Warning: `jpscripts` MCP server may not be configured in Codex. Agents might lack tools. Run `codex mcp add jpscripts ...` to fix.[/yellow]"
        )

    try:
        asyncio.run(_run_swarm(objective, roles, state, safe_mode_active))
    except KeyboardInterrupt:
        console.print("[yellow]Swarm cancelled by user.[/yellow]")
        raise typer.Exit(code=1)
    except RuntimeError:
        raise typer.Exit(code=1)
