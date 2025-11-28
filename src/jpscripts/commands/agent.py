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
import shutil
from typing import Any, Awaitable

import typer
from pydantic import ValidationError
from rich import box
from rich.panel import Panel

from jpscripts.core.agent import (
    PreparedPrompt,
    parse_agent_response,
    prepare_agent_prompt,
    run_repair_loop,
)
from jpscripts.core.console import console
from jpscripts.providers import (
    CompletionOptions,
    LLMProvider,
    Message,
    ProviderError,
    ProviderType,
    infer_provider_type,
)
from jpscripts.providers.codex import is_codex_available
from jpscripts.providers.factory import ProviderConfig, get_provider


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
    # Determine provider type
    ptype: ProviderType | None = None
    if provider_type:
        ptype_map = {
            "anthropic": ProviderType.ANTHROPIC,
            "openai": ProviderType.OPENAI,
            "codex": ProviderType.CODEX,
        }
        ptype = ptype_map.get(provider_type.lower())
        if ptype is None:
            console.print(f"[red]Unknown provider: {provider_type}[/red]")
            raise typer.Exit(code=1)

    # Create provider config
    pconfig = ProviderConfig(
        prefer_codex=(provider_type == "codex" or provider_type is None),
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
# Legacy Codex CLI support (for backward compatibility)
# ---------------------------------------------------------------------------


def _ensure_codex() -> str:
    """Ensure Codex CLI is available."""
    binary = shutil.which("codex")
    if not binary:
        console.print(
            "[red]Codex CLI not found. Install it via `npm install -g @openai/codex` or `brew install codex`.[/red]"
        )
        raise typer.Exit(code=1)
    return binary


def _build_codex_command(
    codex_bin: str,
    model: str,
    prompt: str,
    full_auto: bool,
    web: bool,
    *,
    temperature: float | None = None,
    reasoning_effort: str | None = "high",
) -> list[str]:
    """Build command line for Codex CLI."""
    cmd = [codex_bin, "exec", "--json", "--model", model]
    if reasoning_effort:
        cmd.extend(["-c", f"reasoning.effort={reasoning_effort}"])
    if temperature is not None:
        cmd.extend(["-c", f"temperature={temperature}"])
    if web:
        cmd.append("--search")
    if full_auto:
        cmd.append("--full-auto")
    cmd.append(prompt)
    return cmd


async def _execute_codex_prompt(
    cmd: list[str], *, status_label: str
) -> tuple[list[str], str | None]:
    """Execute Codex CLI and stream output."""
    assistant_parts: list[str] = []
    status = console.status(status_label, spinner="dots")
    status.start()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except KeyboardInterrupt:
        status.stop()
        console.print("[yellow]Codex session cancelled.[/yellow]")
        raise typer.Exit(code=1)
    except Exception as exc:
        status.stop()
        console.print(f"[red]Failed to start Codex:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        if proc.stdout is None or proc.stderr is None:
            return [], "Codex did not provide output."

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                console.print(f"[red]Failed to parse Codex output:[/red] {line}")
                continue

            data: dict[str, Any] = event.get("data") or {}
            event_type = event.get("event") or event.get("type")

            if event_type == "item.started":
                action = (
                    data.get("action")
                    or data.get("command")
                    or data.get("name")
                    or "working"
                )
                status.update(f"[cyan]Running[/cyan] {action}")
            elif event_type == "turn.failed":
                error_msg = (
                    data.get("error") or event.get("error") or "Codex execution failed."
                )
                console.print(
                    Panel(f"[red]{error_msg}[/red]", title="Codex error", box=box.SIMPLE)
                )
            elif event_type == "item.completed":
                action = (
                    data.get("action")
                    or data.get("command")
                    or data.get("name")
                    or "task"
                )
                status.update(f"[green]Completed[/green] {action}")

            message = (
                data.get("assistant_message")
                or event.get("assistant_message")
                or data.get("message")
            )
            if isinstance(message, str) and message.strip():
                assistant_parts.append(message.strip())

        await proc.wait()
        stderr_text = (await proc.stderr.read()).decode(errors="replace").strip()
        return assistant_parts, stderr_text or None
    finally:
        status.stop()


async def _fetch_agent_response_from_codex(
    prepared: PreparedPrompt,
    codex_bin: str,
    model: str,
    full_auto: bool,
    web: bool,
) -> str:
    """Fetch response using Codex CLI directly (legacy path)."""
    cmd = _build_codex_command(
        codex_bin,
        model,
        prepared.prompt,
        full_auto,
        web,
        temperature=prepared.temperature,
        reasoning_effort=prepared.reasoning_effort or "high",
    )
    assistant_parts, stderr_text = await _execute_codex_prompt(
        cmd, status_label="Consulting Codex..."
    )
    if stderr_text:
        console.print(
            Panel(f"[red]{stderr_text}[/red]", title="Codex stderr", box=box.SIMPLE)
        )
    return "\n\n".join(assistant_parts)


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
    use_legacy_codex: bool = typer.Option(
        False,
        "--legacy-codex",
        help="Force use of legacy Codex CLI path (skip provider abstraction).",
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
    root = state.config.workspace_root or state.config.notes_dir
    target_model = model or state.config.default_model

    loop_enabled = bool(run_command) if loop is None else loop
    if loop_enabled and run_command is None:
        console.print("[red]--loop requires --run to know which command to verify.[/red]")
        raise typer.Exit(code=1)

    effective_retries = max(1, max_retries)

    # Determine if we should use legacy Codex path
    use_legacy = use_legacy_codex
    if not use_legacy and provider is None:
        # Auto-detect: use legacy Codex if model looks like OpenAI and Codex is available
        try:
            inferred = infer_provider_type(target_model)
            if inferred == ProviderType.OPENAI and is_codex_available():
                use_legacy = True
        except Exception:
            pass

    # Repair loop mode
    if loop_enabled and run_command:
        if use_legacy:
            codex_bin = _ensure_codex()

            def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
                return _fetch_agent_response_from_codex(
                    prepared, codex_bin, target_model, full_auto, web
                )
        else:

            def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
                return _fetch_agent_response(
                    prepared,
                    state.config,
                    target_model,
                    provider,
                    full_auto=full_auto,
                    web=web,
                )

        success = asyncio.run(
            run_repair_loop(
                base_prompt=prompt,
                command=run_command,
                config=state.config,
                model=target_model,
                attach_recent=attach_recent,
                include_diff=diff,
                fetch_response=fetcher,
                auto_archive=archive,
                max_retries=effective_retries,
                keep_failed=keep_failed,
                web_access=web,
            )
        )
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
            root=root,
            config=state.config,
            model=target_model,
            run_command=run_command,
            attach_recent=attach_recent,
            include_diff=diff,
            ignore_dirs=state.config.ignore_dirs,
            max_file_context_chars=state.config.max_file_context_chars,
            max_command_output_chars=state.config.max_command_output_chars,
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

    # Fetch response
    if use_legacy:
        codex_bin = _ensure_codex()
        console.print(
            Panel("Handing off to [bold magenta]Codex[/bold magenta]...", box=box.SIMPLE)
        )
        cmd = _build_codex_command(
            codex_bin,
            target_model,
            prepared.prompt,
            full_auto,
            web,
            temperature=prepared.temperature,
            reasoning_effort=prepared.reasoning_effort or "high",
        )
        assistant_parts, stderr_text = asyncio.run(
            _execute_codex_prompt(cmd, status_label="Connecting to Codex...")
        )

        if stderr_text:
            console.print(
                Panel(f"[red]{stderr_text}[/red]", title="Codex stderr", box=box.SIMPLE)
            )

        raw_response = "\n\n".join(assistant_parts)
    else:
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

    console.print(
        Panel(agent_response.thought_process, title="Thought process", box=box.SIMPLE)
    )
    if agent_response.tool_call:
        console.print(
            Panel(
                json.dumps(agent_response.tool_call, indent=2),
                title="Tool call",
                box=box.SIMPLE,
            )
        )
    if agent_response.file_patch:
        console.print(
            Panel(agent_response.file_patch, title="Proposed patch", box=box.SIMPLE)
        )
    if agent_response.final_message:
        console.print(
            Panel(agent_response.final_message, title="Final message", box=box.SIMPLE)
        )
