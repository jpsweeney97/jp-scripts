"""
Agent command for delegating tasks to LLM providers.

This module provides the CLI interface for the jp agent functionality,
supporting multiple LLM providers (Anthropic, OpenAI, Codex CLI).

Usage:
    jp agent "Fix the failing test" --run "pytest tests/"
    jp agent "Refactor this function" --model claude-opus-4-5 --provider anthropic
    jp fix "Debug this error" --run "python main.py"  # alias for agent
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any

import typer
from pydantic import ValidationError
from rich import box
from rich.panel import Panel

from jpscripts.core.agent import (
    PreparedPrompt,
    RepairLoopConfig,
    RepairLoopOrchestrator,
    parse_agent_response,
    prepare_agent_prompt,
)
from jpscripts.core.console import console
from jpscripts.providers import (
    CompletionOptions,
    LLMProvider,
    Message,
    ProviderError,
    ProviderType,
)
from jpscripts.providers.factory import ProviderConfig, get_provider, parse_provider_type

# ---------------------------------------------------------------------------
# Provider-based response fetching
# ---------------------------------------------------------------------------


async def _fetch_response_from_provider(
    prepared: PreparedPrompt,
    provider: LLMProvider,
    model: str,
    *,
    stream: bool = True,
) -> str:
    """Fetch a response from an LLM provider.

    Args:
        prepared: The prepared prompt with context
        provider: The LLM provider to use
        model: Model ID to use
        stream: Whether to stream the response (better UX)

    Returns:
        The complete response text
    """
    messages = [Message(role="user", content=prepared.prompt)]

    options = CompletionOptions(
        temperature=prepared.temperature,
        reasoning_effort=prepared.reasoning_effort,
        max_tokens=8192,
    )

    if stream and provider.supports_streaming():
        # Stream response for better UX
        parts: list[str] = []
        status = console.status("Thinking...", spinner="dots")
        status.start()

        try:
            async for chunk in provider.stream(messages, model=model, options=options):
                if chunk.content:
                    parts.append(chunk.content)
                    # Update status to show progress
                    preview = "".join(parts)[-50:].replace("\n", " ")
                    status.update(f"[cyan]Receiving:[/cyan] ...{preview}")
        finally:
            status.stop()

        return "".join(parts)
    else:
        # Non-streaming fallback
        with console.status("Consulting LLM...", spinner="dots"):
            response = await provider.complete(messages, model=model, options=options)
        return response.content


async def _fetch_agent_response(
    prepared: PreparedPrompt,
    config: Any,
    model: str,
    provider_type: str | None,
    *,
    full_auto: bool = False,
    web: bool = False,
) -> str:
    """Fetch agent response using the appropriate provider.

    This function selects the provider based on model ID and user preference,
    then fetches the response.

    Args:
        prepared: The prepared prompt
        config: Application configuration
        model: Model ID to use
        provider_type: Explicit provider type ("anthropic", "openai", "codex", or None for auto)
        full_auto: For Codex: run without confirmation
        web: For Codex: enable web search

    Returns:
        The response text from the LLM
    """
    # Convert string to ProviderType if provided
    ptype: ProviderType | None = None
    if provider_type:
        try:
            ptype = parse_provider_type(provider_type)
        except ValueError:
            console.print(f"[red]Unknown provider: {provider_type}[/red]")
            raise typer.Exit(code=1)

    # Create provider config - prefer_codex when no explicit provider given
    pconfig = ProviderConfig(
        prefer_codex=(provider_type is None),
        codex_full_auto=full_auto,
        codex_web_enabled=web,
    )

    try:
        provider = get_provider(
            config,
            model_id=model,
            provider_type=ptype,
            provider_config=pconfig,
        )
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        raise typer.Exit(code=1)

    # Show which provider we're using
    provider_name = provider.provider_type.name.lower()
    console.print(
        Panel(
            f"Using [bold magenta]{provider_name}[/bold magenta] provider with model [cyan]{model}[/cyan]",
            box=box.SIMPLE,
        )
    )

    try:
        return await _fetch_response_from_provider(prepared, provider, model)
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _display_agent_response(agent_response: Any) -> None:
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


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def codex_exec(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Instruction for the agent."),
    attach_recent: bool = typer.Option(
        False, "--recent", "-r", help="Attach top 5 recently modified files to context."
    ),
    diff: bool = typer.Option(
        True, "--diff/--no-diff", help="Include git diff (staged and unstaged) in context."
    ),
    run_command: str | None = typer.Option(
        None,
        "--run",
        "-x",
        help="Run this shell command first and attach referenced files from output (RAG).",
    ),
    full_auto: bool = typer.Option(
        False, "--full-auto", "-y", help="Run without asking for confirmation (dangerous)."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model to use. Defaults to config."
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: 'anthropic', 'openai', or 'codex'. Auto-detected from model if not specified.",
    ),
    loop: bool | None = typer.Option(
        None,
        "--loop/--no-loop",
        help="Run an autonomous repair loop. Defaults to on when --run is provided.",
    ),
    max_retries: int = typer.Option(
        3, "--max-retries", help="Maximum repair attempts when looping."
    ),
    keep_failed: bool = typer.Option(
        False, "--keep-failed", help="Keep changes even if the loop fails."
    ),
    archive: bool = typer.Option(
        True,
        "--archive/--no-archive",
        help="Save a summary of successful fixes to memory.",
    ),
    web: bool = typer.Option(
        False, "--web/--no-web", help="Enable web search tool for the agent (Codex only)."
    ),
) -> None:
    """Delegate a task to an LLM agent.

    Supports multiple providers:
    - Anthropic Claude (claude-opus-4-5, claude-sonnet-4-5, etc.)
    - OpenAI GPT/o1 (gpt-4o, o1, etc.)
    - Codex CLI (default for backward compatibility)

    Examples:
        jp agent "Fix the failing test" --run "pytest tests/"
        jp agent "Explain this code" --model claude-opus-4-5 --provider anthropic
        jp fix "Debug the error" --run "python main.py" --loop
    """
    state = ctx.obj
    target_model = model or state.config.default_model

    loop_enabled = bool(run_command) if loop is None else loop
    if loop_enabled and run_command is None:
        console.print("[red]--loop requires --run to know which command to verify.[/red]")
        raise typer.Exit(code=1)

    effective_retries = max(1, max_retries)

    # Repair loop mode
    if loop_enabled and run_command:

        def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
            return _fetch_agent_response(
                prepared,
                state.config,
                target_model,
                provider,
                full_auto=full_auto,
                web=web,
            )

        orchestrator = RepairLoopOrchestrator(
            base_prompt=prompt,
            command=run_command,
            model=target_model,
            fetch_response=fetcher,
            config=RepairLoopConfig(
                attach_recent=attach_recent,
                include_diff=diff,
                auto_archive=archive,
                max_retries=effective_retries,
                keep_failed=keep_failed,
                web_access=web,
            ),
        )
        success = asyncio.run(orchestrator.run())
        if not success:
            console.print("[red]Repair loop exhausted without a clean run.[/red]")
        return

    # Single-shot mode
    status_msg = None
    if run_command:
        status_msg = f"Diagnosing with `{run_command}`..."
    elif attach_recent:
        status_msg = "Scanning for recent context..."

    async def _prepare() -> PreparedPrompt:
        return await prepare_agent_prompt(
            base_prompt=prompt,
            model=target_model,
            run_command=run_command,
            attach_recent=attach_recent,
            include_diff=diff,
            web_access=web,
        )

    if status_msg:
        with console.status(status_msg, spinner="dots"):
            prepared: PreparedPrompt = asyncio.run(_prepare())
    else:
        prepared = asyncio.run(_prepare())

    if prepared.attached_files:
        console.print(
            f"[green]Attached files:[/green] {', '.join(p.name for p in prepared.attached_files)}"
        )
    elif run_command:
        console.print(
            "[yellow]No files detected in command output. Proceeding without file context.[/yellow]"
        )

    # Fetch response via unified provider path
    raw_response = asyncio.run(
        _fetch_agent_response(
            prepared,
            state.config,
            target_model,
            provider,
            full_auto=full_auto,
            web=web,
        )
    )

    if not raw_response:
        console.print("[yellow]No response received from agent.[/yellow]")
        return

    # Parse and display response
    try:
        agent_response = parse_agent_response(raw_response)
    except ValidationError as exc:
        console.print(
            Panel(
                f"[red]Agent response validation failed:[/red]\n{exc}",
                title="Parse error",
                box=box.SIMPLE,
            )
        )
        console.print(Panel(raw_response, title="Raw agent response", box=box.SIMPLE))
        return

    _display_agent_response(agent_response)
