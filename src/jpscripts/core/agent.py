from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, Field, ValidationError

from jpscripts.core import git as git_core
from jpscripts.core import git_ops
from jpscripts.core import security
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console, get_logger
from jpscripts.core.context import gather_context, read_file_context, smart_read_context
from jpscripts.core.nav import scan_recent
from jpscripts.core.structure import generate_map, get_import_dependencies

logger = get_logger(__name__)

AGENT_TEMPLATE_NAME = "agent_system.json.j2"


@dataclass
class PreparedPrompt:
    prompt: str
    attached_files: list[Path]


class AgentResponse(BaseModel):
    """Structured response contract for agent outputs."""

    thought_process: str = Field(..., description="Deep analysis of the problem")
    shell_command: str | None = Field(None, description="Command to run (optional)")
    file_patch: str | None = Field(None, description="Unified diff to apply (optional)")
    final_message: str | None = Field(None, description="Response to user if no action needed")


def parse_agent_response(payload: str) -> AgentResponse:
    """Parse and validate a JSON agent response."""
    return AgentResponse.model_validate_json(payload)


@dataclass
class AttemptContext:
    iteration: int
    last_error: str
    files_changed: list[Path]


PatchFetcher = Callable[[PreparedPrompt], Awaitable[str]]
ResponseFetcher = Callable[[PreparedPrompt], Awaitable[str]]


def _resolve_template_root() -> Path:
    package_root = Path(__file__).resolve().parent.parent
    return security.validate_path(package_root / "templates", package_root)


@lru_cache(maxsize=1)
def _get_template_environment(template_root: Path) -> Environment:
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    env.filters["cdata"] = _safe_cdata
    return env


def _render_prompt_from_template(context: dict[str, object], template_root: Path) -> str:
    try:
        template = _get_template_environment(template_root).get_template(AGENT_TEMPLATE_NAME)
    except TemplateNotFound as exc:
        logger.error("Agent template %s missing in %s", AGENT_TEMPLATE_NAME, template_root)
        raise FileNotFoundError(f"Template {AGENT_TEMPLATE_NAME} not found in {template_root}") from exc
    return template.render(**context)


async def prepare_agent_prompt(
    base_prompt: str,
    *,
    root: Path,
    run_command: str | None,
    attach_recent: bool,
    include_diff: bool = False,
    ignore_dirs: Sequence[str],
    max_file_context_chars: int,
    max_command_output_chars: int,
) -> PreparedPrompt:
    """
    Builds a structured, JSON-oriented prompt for Codex.
    """
    branch, commit, is_dirty = await _collect_git_context(root)

    repository_map = await asyncio.to_thread(generate_map, root, 3)
    constitution_text = await _load_constitution(root)

    attached: list[Path] = []

    diagnostic_section = ""
    file_context_section = ""
    dependency_section = ""

    # 2. Command Output (Diagnostic)
    if run_command:
        output, detected_files = await gather_context(run_command, root)
        trimmed = output[-max_command_output_chars:]
        diagnostic_section = (
            f"Command: {run_command}\n"
            f"Output (last {max_command_output_chars} chars):\n"
            f"{trimmed}\n"
        )

        # Prioritize files detected in the stack trace
        detected_paths = list(sorted(detected_files))[:5]
        file_context_section, attached = await _build_file_context_section(detected_paths, max_file_context_chars)
        dependency_section = await _build_dependency_section(detected_paths, root, max_file_context_chars)

    # 3. Recent Context (Ambient)
    elif attach_recent:
        recents = await scan_recent(root, 3, False, set(ignore_dirs))
        recent_paths = [entry.path for entry in recents[:5]]
        file_context_section, attached = await _build_file_context_section(recent_paths, max_file_context_chars)
        dependency_section = await _build_dependency_section(recent_paths[:1], root, max_file_context_chars)

    # 4. Git Diff (Work in Progress)
    git_diff_section = ""
    if include_diff:
        diff_text = await _collect_git_diff(root, 10_000)
        if diff_text:
            git_diff_section = diff_text
        else:
            git_diff_section = "NO CHANGES"

    template_root = _resolve_template_root()
    response_schema = AgentResponse.model_json_schema()
    context = {
        "workspace_root": str(root),
        "branch": branch,
        "head": commit,
        "dirty": is_dirty,
        "repository_map": repository_map,
        "constitution": constitution_text,
        "diagnostic_section": diagnostic_section,
        "file_context_section": file_context_section,
        "dependency_section": dependency_section,
        "git_diff_section": git_diff_section,
        "instruction": base_prompt.strip(),
        "response_schema": response_schema,
    }

    prompt = await asyncio.to_thread(_render_prompt_from_template, context, template_root)

    return PreparedPrompt(prompt=prompt, attached_files=attached)


def _safe_cdata(content: str) -> str:
    """Escape CDATA terminators inside arbitrary content."""
    return content.replace("]]>", "]]]]><![CDATA[>")


async def _load_constitution(root: Path) -> str:
    """Read AGENTS.md content under the workspace root.

    Args:
        root: Workspace root for validation and lookup.

    Returns:
        Constitution text or a fallback message when unavailable.
    """
    try:
        candidate = security.validate_path(root / "AGENTS.md", root)
    except Exception as exc:
        logger.debug("Unable to resolve AGENTS.md under %s: %s", root, exc)
        return "AGENTS.md not accessible."

    if not candidate.exists():
        return "AGENTS.md not found."

    content = await asyncio.to_thread(read_file_context, candidate, 5000)
    if not content:
        return "AGENTS.md is empty or unreadable."

    return content


async def _collect_git_context(root: Path) -> tuple[str, str, bool]:
    if not root.exists() or not (root / ".git").exists():
        return "(no repo)", "(no repo)", False

    try:
        repo = await git_core.AsyncRepo.open(root)
    except git_core.GitOperationError as exc:
        logger.error("Failed to open git repo at %s: %s", root, exc)
        return "(error)", "(error)", False
    except Exception as exc:
        logger.error("Failed to open git repo at %s: %s", root, exc)
        return "(error)", "(error)", False

    branch = "(unknown)"
    commit = "(unknown)"
    is_dirty = False

    try:
        status = await repo.status()
        branch = status.branch
        is_dirty = status.dirty
        _ = git_ops.format_status(status)
    except Exception as exc:
        logger.error("Failed to describe git status for %s: %s", root, exc)
        return "(error)", "(error)", False

    try:
        commit = await repo.head(short=True)
    except Exception as exc:
        logger.error("Failed to resolve git head for %s: %s", root, exc)
        commit = "(error)"

    return branch, commit, is_dirty


async def _build_file_context_section(paths: Sequence[Path], max_file_context_chars: int) -> tuple[str, list[Path]]:
    sections: list[str] = []
    attached: list[Path] = []
    for path in paths:
        snippet = await asyncio.to_thread(smart_read_context, path, max_file_context_chars)
        if not snippet:
            continue
        sections.append(f"Path: {path}\n---\n{snippet}\n")
        attached.append(path)
    if not sections:
        return "", attached
    return "\n".join(sections), attached


async def _build_dependency_section(paths: Sequence[Path], root: Path, max_file_context_chars: int) -> str:
    dependencies: set[Path] = set()
    for path in paths:
        deps = await asyncio.to_thread(get_import_dependencies, path, root)
        dependencies.update(deps)

    if not dependencies:
        return ""

    sections: list[str] = []
    for dep in sorted(dependencies):
        snippet = await asyncio.to_thread(smart_read_context, dep, max_file_context_chars)
        if not snippet:
            continue
        sections.append(f"Dependency: {dep}\n---\n{snippet}\n")

    return "\n".join(sections)


async def _collect_git_diff(root: Path, max_chars: int) -> str | None:
    if not root.exists() or not (root / ".git").exists():
        return None

    try:
        proc = await asyncio.create_subprocess_shell(
            "git diff HEAD",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return None

    if proc.returncode != 0:
        return None

    diff = stdout.decode(errors="replace")
    if not diff.strip():
        return None

    if len(diff) > max_chars:
        return f"{diff[:max_chars]}... [truncated]"

    return diff


def _summarize_output(stdout: str, stderr: str, limit: int) -> str:
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    if not combined:
        return "Command failed without output."
    if len(combined) > limit:
        return combined[-limit:] + "... [truncated]"
    return combined


def _build_history_summary(history: Sequence[AttemptContext], root: Path) -> str:
    if not history:
        return "None yet."

    lines: list[str] = []
    for attempt in history:
        relative_files = []
        for path in attempt.files_changed:
            try:
                relative_files.append(str(path.relative_to(root)))
            except ValueError:
                relative_files.append(str(path))
        file_part = f" | files: {', '.join(relative_files)}" if relative_files else ""
        lines.append(f"Attempt {attempt.iteration}: {attempt.last_error}{file_part}")

    return "\n".join(lines)


def _build_repair_instruction(
    base_prompt: str,
    current_error: str,
    history: Sequence[AttemptContext],
    root: Path,
) -> str:
    history_block = _build_history_summary(history, root)
    return (
        f"{base_prompt.strip()}\n\n"
        "Autonomous repair loop in progress. Use the failure details to craft a minimal fix.\n"
        f"Current error:\n{current_error.strip()}\n\n"
        f"Previous attempts:\n{history_block}\n\n"
        "Respond with a single JSON object that matches the AgentResponse schema. "
        "Place the unified diff in `file_patch`. Do not return Markdown or prose."
    )


async def _run_shell_command(command: str, cwd: Path) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        return 1, "", str(exc)

    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _extract_patch_paths(patch_text: str, root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for raw_line in patch_text.splitlines():
        if not raw_line.startswith(("+++ ", "--- ")):
            continue
        try:
            _, path_str = raw_line.split(" ", 1)
        except ValueError:
            continue
        path_str = path_str.strip()
        if path_str in {"/dev/null", "dev/null", "a/dev/null", "b/dev/null"}:
            continue
        if path_str.startswith(("a/", "b/")):
            path_str = path_str[2:]
        try:
            candidates.add(security.validate_path(root / path_str, root))
        except PermissionError as exc:
            logger.debug("Skipped unsafe patch path %s: %s", path_str, exc)
        except Exception as exc:
            logger.debug("Failed to normalize patch path %s: %s", path_str, exc)
    return sorted(candidates)


def _write_failed_patch(patch_text: str, root: Path) -> None:
    try:
        destination = security.validate_path(root / "agent_failed_patch.diff", root)
        destination.write_text(patch_text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to persist failed patch for inspection: %s", exc)


async def _apply_patch_text(patch_text: str, root: Path) -> list[Path]:
    if not patch_text.strip():
        return []

    target_paths = _extract_patch_paths(patch_text, root)

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "apply",
            "--whitespace=nowarn",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        proc = None

    if proc:
        stdout, stderr = await proc.communicate(patch_text.encode())
        if proc.returncode == 0:
            return target_paths
        logger.debug("git apply failed: %s", stderr.decode(errors="replace"))

    try:
        fallback = await asyncio.create_subprocess_exec(
            "patch",
            "-p1",
            cwd=root,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        _write_failed_patch(patch_text, root)
        return []

    out, err = await fallback.communicate(patch_text.encode())
    if fallback.returncode == 0:
        return target_paths

    logger.error(
        "Patch application failed: %s",
        err.decode(errors="replace") or out.decode(errors="replace"),
    )
    _write_failed_patch(patch_text, root)
    return []


async def _revert_changed_files(paths: Sequence[Path], root: Path) -> None:
    if not paths:
        return

    safe_paths: list[Path] = []
    for path in paths:
        try:
            safe_paths.append(security.validate_path(path, root))
        except PermissionError as exc:
            logger.debug("Skipping revert for unsafe path %s: %s", path, exc)

    if not safe_paths:
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "checkout",
            "--",
            *[str(path) for path in safe_paths],
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return

    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug("Failed to revert files after unsuccessful loop: %s", stderr.decode(errors="replace"))


async def run_repair_loop(
    *,
    base_prompt: str,
    command: str,
    config: AppConfig,
    attach_recent: bool,
    include_diff: bool,
    fetch_response: ResponseFetcher,
    max_retries: int = 3,
    keep_failed: bool = False,
) -> bool:
    """
    Execute an autonomous repair loop that retries the provided command.
    """
    root = security.validate_workspace_root(config.workspace_root or config.notes_dir)
    attempt_cap = max(1, max_retries)
    history: list[AttemptContext] = []
    changed_files: set[Path] = set()

    for attempt in range(attempt_cap):
        console.print(f"[cyan]Attempt {attempt + 1}/{attempt_cap}: running `{command}`[/cyan]")
        exit_code, stdout, stderr = await _run_shell_command(command, root)
        if exit_code == 0:
            console.print("[green]Command succeeded. Exiting repair loop.[/green]")
            return True

        current_error = _summarize_output(stdout, stderr, config.max_command_output_chars)
        console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {current_error}")

        iteration_prompt = _build_repair_instruction(base_prompt, current_error, history, root)
        prepared = await prepare_agent_prompt(
            iteration_prompt,
            root=root,
            run_command=None,
            attach_recent=attach_recent,
            include_diff=include_diff,
            ignore_dirs=config.ignore_dirs,
            max_file_context_chars=config.max_file_context_chars,
            max_command_output_chars=config.max_command_output_chars,
        )

        if prepared.attached_files:
            console.print(
                f"[green]Attached files for attempt {attempt + 1}:[/green] "
                f"{', '.join(path.name for path in prepared.attached_files)}"
            )

        try:
            raw_response = await fetch_response(prepared)
        except Exception as exc:
            console.print(f"[red]Failed to retrieve agent response:[/red] {exc}")
            break

        if not raw_response.strip():
            console.print("[red]Agent returned empty response. Aborting loop.[/red]")
            break

        try:
            agent_response = parse_agent_response(raw_response)
        except ValidationError as exc:
            validation_error = f"Agent response validation failed: {exc}"
            console.print(f"[red]{validation_error}[/red]")
            current_error = validation_error
            history.append(AttemptContext(iteration=attempt + 1, last_error=validation_error, files_changed=[]))
            continue

        if agent_response.shell_command:
            console.print(f"[yellow]Agent suggested shell command (not executed automatically):[/yellow] {agent_response.shell_command}")

        patch_text = (agent_response.file_patch or "").strip()
        if not patch_text:
            message = agent_response.final_message or "Agent returned no patch content."
            console.print(f"[red]{message}[/red]")
            break

        applied_paths = await _apply_patch_text(patch_text, root)
        history.append(AttemptContext(iteration=attempt + 1, last_error=current_error, files_changed=applied_paths))
        changed_files.update(applied_paths)

    console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
    exit_code, stdout, stderr = await _run_shell_command(command, root)
    if exit_code == 0:
        console.print("[green]Command succeeded after final verification.[/green]")
        return True

    console.print(f"[red]Command still failing:[/red] {_summarize_output(stdout, stderr, config.max_command_output_chars)}")
    if changed_files and not keep_failed:
        console.print("[yellow]Reverting changes from failed attempts.[/yellow]")
        await _revert_changed_files(list(changed_files), root)

    return False
