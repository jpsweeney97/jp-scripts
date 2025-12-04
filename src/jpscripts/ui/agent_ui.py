"""UI rendering utilities for agent commands.

This module contains the visual/console rendering logic for agent operations,
separated from the CLI orchestration in commands/agent.py.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from rich import box
from rich.panel import Panel

from jpscripts.agent import EventKind, PreparedPrompt
from jpscripts.core.console import console

if TYPE_CHECKING:
    from jpscripts.agent import RepairLoopOrchestrator, SingleShotResult
    from jpscripts.providers import CompletionOptions, LLMProvider


# --- Event Handler Functions ---


def _handle_attempt_start(data: dict[str, Any], message: str) -> None:
    console.print(
        f"[cyan]Attempt {data['attempt']}/{data['max']} "
        f"({data['strategy']}): running `{data['command']}`[/cyan]"
    )


def _handle_command_success(data: dict[str, Any], message: str) -> None:
    phase = data.get("phase", "")
    if phase == "initial":
        console.print("[green]Command succeeded. Exiting repair loop.[/green]")
    elif data.get("after_fixes"):
        console.print("[green]Command succeeded after applying fixes.[/green]")
    elif phase == "final_verification":
        console.print("[green]Command succeeded after final verification.[/green]")
    else:
        console.print(f"[green]{message}[/green]")


def _handle_command_failed(data: dict[str, Any], message: str) -> None:
    phase = data.get("phase", "")
    error = data.get("error", "")
    if phase == "initial":
        console.print(f"[yellow]{message}:[/yellow] {error}")
    elif phase == "verification":
        console.print(f"[yellow]Verification failed:[/yellow] {error}")
    elif phase == "final_verification_start":
        console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
    elif phase == "final":
        console.print(f"[red]Command still failing:[/red] {error}")
    else:
        console.print(f"[yellow]{message}:[/yellow] {error}")


def _handle_tool_call(data: dict[str, Any], message: str) -> None:
    console.print(
        Panel(
            f"Agent invoking {data['tool_name']} with args {data['arguments']}",
            title="Tool Call",
            box=box.SIMPLE,
        )
    )


def _handle_tool_output(data: dict[str, Any], message: str) -> None:
    console.print(Panel(data["output"], title="Tool Output", box=box.SIMPLE, style="cyan"))


def _handle_syntax_error(data: dict[str, Any], message: str) -> None:
    console.print(f"[red]Syntax Check Failed (Self-Correction):[/red] {data['error']}")


def _handle_no_patch(data: dict[str, Any], message: str) -> None:
    console.print(f"[yellow]{data.get('message', message)}[/yellow]")


# --- Event Dispatch Table ---

_EVENT_HANDLERS: dict[EventKind, Callable[[dict[str, Any], str], None]] = {
    EventKind.ATTEMPT_START: _handle_attempt_start,
    EventKind.COMMAND_SUCCESS: _handle_command_success,
    EventKind.COMMAND_FAILED: _handle_command_failed,
    EventKind.TOOL_CALL: _handle_tool_call,
    EventKind.TOOL_OUTPUT: _handle_tool_output,
    EventKind.PATCH_PROPOSED: lambda d, m: console.print("[green]Agent proposed a fix.[/green]"),
    EventKind.PATCH_APPLIED: lambda d, m: None,  # Implied by PATCH_PROPOSED
    EventKind.SYNTAX_ERROR: _handle_syntax_error,
    EventKind.DUPLICATE_PATCH: lambda d, m: console.print(
        "[yellow]Duplicate patch detected - skipping.[/yellow]"
    ),
    EventKind.LOOP_DETECTED: lambda d, m: console.print(
        "[yellow]Repeated failure detected; applying strategy override "
        "and higher reasoning effort.[/yellow]"
    ),
    EventKind.VALIDATION_ERROR: lambda d, m: console.print(f"[red]{m}[/red]"),
    EventKind.NO_PATCH: _handle_no_patch,
    EventKind.REVERTING: lambda d, m: console.print(
        "[yellow]Reverting changes from failed attempts.[/yellow]"
    ),
}


async def render_repair_loop_events(orchestrator: RepairLoopOrchestrator) -> bool:
    """Consume repair loop events and render UI.

    Args:
        orchestrator: The configured repair loop orchestrator.

    Returns:
        True if the repair succeeded, False otherwise.
    """
    success = False

    async for event in orchestrator.run():
        if event.kind == EventKind.COMPLETE:
            success = event.data.get("success", False)
            continue

        if handler := _EVENT_HANDLERS.get(event.kind):
            handler(event.data, event.message)

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


async def stream_provider_response(
    provider: LLMProvider,
    prepared: PreparedPrompt,
    model: str,
    options: CompletionOptions,
) -> str:
    """Stream a response from the provider with live UI feedback.

    Args:
        provider: The LLM provider to use.
        prepared: The prepared prompt.
        model: Model ID to use.
        options: Completion options.

    Returns:
        The complete response text.
    """
    from jpscripts.providers import Message

    messages = [Message(role="user", content=prepared.prompt)]

    if provider.supports_streaming():
        parts: list[str] = []
        status = console.status("Thinking...", spinner="dots")
        status.start()
        try:
            async for chunk in provider.stream(messages, model=model, options=options):
                if chunk.content:
                    parts.append(chunk.content)
                    preview = "".join(parts)[-50:].replace("\n", " ")
                    status.update(f"[cyan]Receiving:[/cyan] ...{preview}")
        finally:
            status.stop()
        return "".join(parts)
    else:
        with console.status("Consulting LLM...", spinner="dots"):
            response = await provider.complete(messages, model=model, options=options)
        return response.content


def create_response_fetcher(
    provider: LLMProvider,
    model: str,
    temperature: float | None = None,
    max_tokens: int = 8192,
) -> Callable[[PreparedPrompt], Awaitable[str]]:
    """Create a ResponseFetcher from a provider for use with agent orchestrators.

    Args:
        provider: The LLM provider instance.
        model: Model ID to use.
        temperature: Optional temperature override.
        max_tokens: Maximum tokens in response.

    Returns:
        A callable that fetches responses from the provider.
    """
    from jpscripts.providers import CompletionOptions

    async def fetcher(prepared: PreparedPrompt) -> str:
        options = CompletionOptions(
            temperature=prepared.temperature if temperature is None else temperature,
            reasoning_effort=prepared.reasoning_effort,
            max_tokens=max_tokens,
        )
        return await stream_provider_response(provider, prepared, model, options)

    return fetcher


def display_single_shot_result(result: SingleShotResult) -> None:
    """Display the result of a single-shot agent run.

    Args:
        result: The single-shot execution result.
    """
    if result.prepared.attached_files:
        console.print(
            f"[green]Attached files:[/green] {', '.join(p.name for p in result.prepared.attached_files)}"
        )

    if result.error:
        console.print(Panel(result.error, title="Error", box=box.SIMPLE, style="red"))
        if result.raw_response:
            console.print(Panel(result.raw_response, title="Raw agent response", box=box.SIMPLE))
        return

    if result.agent_response:
        display_agent_response(result.agent_response)
