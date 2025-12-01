"""UI rendering utilities for agent commands.

This module contains the visual/console rendering logic for agent operations,
separated from the CLI orchestration in commands/agent.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rich import box
from rich.panel import Panel

from jpscripts.agent import EventKind
from jpscripts.core.console import console

if TYPE_CHECKING:
    from jpscripts.agent import RepairLoopOrchestrator


async def render_repair_loop_events(orchestrator: RepairLoopOrchestrator) -> bool:
    """Consume repair loop events and render UI.

    Args:
        orchestrator: The configured repair loop orchestrator.

    Returns:
        True if the repair succeeded, False otherwise.
    """
    success = False

    async for event in orchestrator.run():
        match event.kind:
            case EventKind.ATTEMPT_START:
                console.print(
                    f"[cyan]Attempt {event.data['attempt']}/{event.data['max']} "
                    f"({event.data['strategy']}): running `{event.data['command']}`[/cyan]"
                )
            case EventKind.COMMAND_SUCCESS:
                phase = event.data.get("phase", "")
                if phase == "initial":
                    console.print("[green]Command succeeded. Exiting repair loop.[/green]")
                elif event.data.get("after_fixes"):
                    console.print("[green]Command succeeded after applying fixes.[/green]")
                elif phase == "final_verification":
                    console.print("[green]Command succeeded after final verification.[/green]")
                else:
                    console.print(f"[green]{event.message}[/green]")
            case EventKind.COMMAND_FAILED:
                phase = event.data.get("phase", "")
                error = event.data.get("error", "")
                if phase == "initial":
                    console.print(f"[yellow]{event.message}:[/yellow] {error}")
                elif phase == "verification":
                    console.print(f"[yellow]Verification failed:[/yellow] {error}")
                elif phase == "final_verification_start":
                    console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
                elif phase == "final":
                    console.print(f"[red]Command still failing:[/red] {error}")
                else:
                    console.print(f"[yellow]{event.message}:[/yellow] {error}")
            case EventKind.TOOL_CALL:
                console.print(
                    Panel(
                        f"Agent invoking {event.data['tool_name']} with args {event.data['arguments']}",
                        title="Tool Call",
                        box=box.SIMPLE,
                    )
                )
            case EventKind.TOOL_OUTPUT:
                console.print(
                    Panel(event.data["output"], title="Tool Output", box=box.SIMPLE, style="cyan")
                )
            case EventKind.PATCH_PROPOSED:
                console.print("[green]Agent proposed a fix.[/green]")
            case EventKind.PATCH_APPLIED:
                pass  # Implied by PATCH_PROPOSED
            case EventKind.SYNTAX_ERROR:
                console.print(
                    f"[red]Syntax Check Failed (Self-Correction):[/red] {event.data['error']}"
                )
            case EventKind.DUPLICATE_PATCH:
                console.print("[yellow]Duplicate patch detected - skipping.[/yellow]")
            case EventKind.LOOP_DETECTED:
                console.print(
                    "[yellow]Repeated failure detected; applying strategy override "
                    "and higher reasoning effort.[/yellow]"
                )
            case EventKind.VALIDATION_ERROR:
                console.print(f"[red]{event.message}[/red]")
            case EventKind.NO_PATCH:
                console.print(f"[yellow]{event.data.get('message', event.message)}[/yellow]")
            case EventKind.REVERTING:
                console.print("[yellow]Reverting changes from failed attempts.[/yellow]")
            case EventKind.COMPLETE:
                success = event.data.get("success", False)

    return success


def display_agent_response(agent_response: Any) -> None:
    """Display the parsed agent response."""
    console.print(Panel(agent_response.thought_process, title="Thought process", box=box.SIMPLE))
    if agent_response.tool_call:
        console.print(
            Panel(
                json.dumps(agent_response.tool_call, indent=2),
                title="Tool call",
                box=box.SIMPLE,
            )
        )
    if agent_response.file_patch:
        console.print(Panel(agent_response.file_patch, title="Proposed patch", box=box.SIMPLE))
    if agent_response.final_message:
        console.print(Panel(agent_response.final_message, title="Final message", box=box.SIMPLE))
