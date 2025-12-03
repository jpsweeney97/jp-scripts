"""Agent command - thin CLI dispatcher for LLM agent tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import typer
from rich import box
from rich.panel import Panel

from jpscripts.agent import (
    PreparedPrompt,
    RepairLoopConfig,
    RepairLoopOrchestrator,
    SingleShotConfig,
    SingleShotRunner,
)
from jpscripts.core.console import console
from jpscripts.providers import LLMProvider, ProviderError
from jpscripts.providers.factory import ProviderConfig, get_provider, parse_provider_type
from jpscripts.ui.agent_ui import (
    create_response_fetcher,
    display_single_shot_result,
    render_repair_loop_events,
)


def _get_provider_and_fetcher(
    state: Any,
    model: str,
    provider_type: str | None,
    web: bool,
) -> tuple[LLMProvider, Callable[[PreparedPrompt], Awaitable[str]]]:
    """Get provider and create response fetcher. Returns (provider, fetcher) or raises Exit."""
    ptype = None
    if provider_type:
        try:
            ptype = parse_provider_type(provider_type)
        except ValueError as exc:
            console.print(f"[red]Provider error: {exc}[/red]")
            raise typer.Exit(code=1)

    try:
        provider = get_provider(
            state.config,
            model_id=model,
            provider_type=ptype,
            provider_config=ProviderConfig(web_enabled=web),
        )
    except ProviderError as exc:
        console.print(f"[red]Provider error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Using [bold magenta]{provider.provider_type.name.lower()}[/bold magenta] "
            f"provider with model [cyan]{model}[/cyan]",
            box=box.SIMPLE,
        )
    )
    return provider, create_response_fetcher(provider, model)


def codex_exec(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Instruction for the agent."),
    attach_recent: bool = typer.Option(False, "--recent", "-r", help="Attach recent files."),
    diff: bool = typer.Option(True, "--diff/--no-diff", help="Include git diff."),
    run_command: str | None = typer.Option(None, "--run", "-x", help="Run command for context."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use."),
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="Provider: anthropic/openai."
    ),
    loop: bool | None = typer.Option(None, "--loop/--no-loop", help="Run repair loop."),
    max_retries: int = typer.Option(3, "--max-retries", help="Max repair attempts."),
    keep_failed: bool = typer.Option(False, "--keep-failed", help="Keep changes on failure."),
    archive: bool = typer.Option(True, "--archive/--no-archive", help="Archive fixes to memory."),
    web: bool = typer.Option(False, "--web/--no-web", help="Enable web search."),
) -> None:
    """Delegate a task to an LLM agent."""
    state = ctx.obj
    target_model = model or state.config.ai.default_model
    loop_enabled = bool(run_command) if loop is None else loop

    if loop_enabled and run_command is None:
        console.print("[red]--loop requires --run to know which command to verify.[/red]")
        raise typer.Exit(code=1)

    _, fetcher = _get_provider_and_fetcher(state, target_model, provider, web)

    if loop_enabled and run_command:
        orchestrator = RepairLoopOrchestrator(
            base_prompt=prompt,
            command=run_command,
            model=target_model,
            fetch_response=fetcher,
            config=RepairLoopConfig(
                attach_recent=attach_recent,
                include_diff=diff,
                auto_archive=archive,
                max_retries=max(1, max_retries),
                keep_failed=keep_failed,
                web_access=web,
            ),
            app_config=state.config,
            workspace_root=state.runtime_ctx.workspace_root,
        )
        success = asyncio.run(render_repair_loop_events(orchestrator))
        if not success:
            console.print("[red]Repair loop exhausted without a clean run.[/red]")
        return

    runner = SingleShotRunner(
        prompt=prompt,
        model=target_model,
        fetch_response=fetcher,
        config=SingleShotConfig(attach_recent=attach_recent, include_diff=diff, web_access=web),
        run_command=run_command,
    )
    result = asyncio.run(runner.run())
    display_single_shot_result(result)
