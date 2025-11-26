from __future__ import annotations

import asyncio
import json
import shutil
from typing import Any

import typer
from rich import box
from rich.panel import Panel

from jpscripts.core.agent import PreparedPrompt, prepare_agent_prompt, run_repair_loop
from jpscripts.core.console import console


def _ensure_codex() -> str:
    binary = shutil.which("codex")
    if not binary:
        console.print("[red]Codex CLI not found. Install it via `npm install -g @openai/codex` or `brew install codex`.[/red]")
        raise typer.Exit(code=1)
    return binary


def _build_codex_command(codex_bin: str, model: str, prompt: str, full_auto: bool) -> list[str]:
    cmd = [codex_bin, "exec", "--json", "--model", model, "-c", "reasoning.effort=high"]
    if full_auto:
        cmd.append("--full-auto")
    cmd.append(prompt)
    return cmd


async def _execute_codex_prompt(cmd: list[str], *, status_label: str) -> tuple[list[str], str | None]:
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
                action = data.get("action") or data.get("command") or data.get("name") or "working"
                status.update(f"[cyan]Running[/cyan] {action}")
            elif event_type == "turn.failed":
                error_msg = data.get("error") or event.get("error") or "Codex execution failed."
                console.print(Panel(f"[red]{error_msg}[/red]", title="Codex error", box=box.SIMPLE))
            elif event_type == "item.completed":
                action = data.get("action") or data.get("command") or data.get("name") or "task"
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


async def _fetch_patch_from_codex(prepared: PreparedPrompt, codex_bin: str, model: str, full_auto: bool) -> str:
    cmd = _build_codex_command(codex_bin, model, prepared.prompt, full_auto)
    assistant_parts, stderr_text = await _execute_codex_prompt(cmd, status_label="Consulting Codex...")
    if stderr_text:
        console.print(Panel(f"[red]{stderr_text}[/red]", title="Codex stderr", box=box.SIMPLE))
    return "\n\n".join(assistant_parts)


def codex_exec(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Instruction for Codex."),
    attach_recent: bool = typer.Option(False, "--recent", "-r", help="Attach top 5 recently modified files to context."),
    diff: bool = typer.Option(True, "--diff/--no-diff", help="Include git diff (staged and unstaged) in context."),
    run_command: str | None = typer.Option(None, "--run", "-x", help="Run this shell command first and attach referenced files from output (RAG)."),
    full_auto: bool = typer.Option(False, "--full-auto", "-y", help="Run without asking for confirmation (dangerous)."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use. Defaults to config."),
    loop: bool | None = typer.Option(
        None,
        "--loop/--no-loop",
        help="Run an autonomous repair loop. Defaults to on when --run is provided.",
    ),
    max_retries: int = typer.Option(3, "--max-retries", help="Maximum repair attempts when looping."),
    keep_failed: bool = typer.Option(False, "--keep-failed", help="Keep changes even if the loop fails."),
) -> None:
    """Delegate a task to the Codex agent."""
    state = ctx.obj
    root = state.config.workspace_root or state.config.notes_dir
    target_model = model or state.config.default_model
    codex_bin = _ensure_codex()

    loop_enabled = bool(run_command) if loop is None else loop
    if loop_enabled and run_command is None:
        console.print("[red]--loop requires --run to know which command to verify.[/red]")
        raise typer.Exit(code=1)

    effective_retries = max(1, max_retries)

    if loop_enabled and run_command:
        fetcher = lambda prepared: _fetch_patch_from_codex(prepared, codex_bin, target_model, full_auto)
        success = asyncio.run(
            run_repair_loop(
                base_prompt=prompt,
                command=run_command,
                config=state.config,
                attach_recent=attach_recent,
                include_diff=diff,
                fetch_patch=fetcher,
                max_retries=effective_retries,
                keep_failed=keep_failed,
            )
        )
        if not success:
            console.print("[red]Repair loop exhausted without a clean run.[/red]")
        return

    status_msg = None
    if run_command:
        status_msg = f"Diagnosing with `{run_command}`..."
    elif attach_recent:
        status_msg = "Scanning for recent context..."

    prepare_kwargs = dict(
        base_prompt=prompt,
        root=root,
        run_command=run_command,
        attach_recent=attach_recent,
        include_diff=diff,
        ignore_dirs=state.config.ignore_dirs,
        max_file_context_chars=state.config.max_file_context_chars,
        max_command_output_chars=state.config.max_command_output_chars,
    )

    if status_msg:
        with console.status(status_msg, spinner="dots"):
            prepared: PreparedPrompt = asyncio.run(prepare_agent_prompt(**prepare_kwargs))
    else:
        prepared = asyncio.run(prepare_agent_prompt(**prepare_kwargs))

    if prepared.attached_files:
        console.print(f"[green]Attached files:[/green] {', '.join(p.name for p in prepared.attached_files)}")
    elif run_command:
        console.print("[yellow]No files detected in command output. Proceeding without file context.[/yellow]")

    console.print(Panel("Handing off to [bold magenta]Codex[/bold magenta]...", box=box.SIMPLE))

    cmd = _build_codex_command(codex_bin, target_model, prepared.prompt, full_auto)
    assistant_parts, stderr_text = asyncio.run(_execute_codex_prompt(cmd, status_label="Connecting to Codex..."))

    if stderr_text:
        console.print(Panel(f"[red]{stderr_text}[/red]", title="Codex stderr", box=box.SIMPLE))

    if assistant_parts:
        final_message = "\n\n".join(assistant_parts)
        console.print(Panel(final_message, title="Codex response", box=box.SIMPLE))
