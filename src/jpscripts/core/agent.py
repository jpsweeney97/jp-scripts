from __future__ import annotations

import asyncio
import json
import sys
import re
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
from jpscripts.core.context import gather_context, get_file_skeleton, read_file_context, smart_read_context
from jpscripts.core.memory import query_memory, save_memory
from jpscripts.core.nav import scan_recent
from jpscripts.core.structure import generate_map, get_import_dependencies

logger = get_logger(__name__)

AGENT_TEMPLATE_NAME = "agent_system.json.j2"
STRATEGY_OVERRIDE_TEXT = (
    "You are stuck in a loop. Stop editing code. Analyze the error trace and the file content again. "
    "List three possible root causes before proposing a new patch."
)


@dataclass
class PreparedPrompt:
    prompt: str
    attached_files: list[Path]
    temperature: float | None = None
    reasoning_effort: str | None = None


class AgentResponse(BaseModel):
    """Structured response contract for agent outputs."""

    thought_process: str = Field(..., description="Deep analysis of the problem")
    shell_command: str | None = Field(None, description="Command to run (optional)")
    file_patch: str | None = Field(None, description="Unified diff to apply (optional)")
    final_message: str | None = Field(None, description="Response to user if no action needed")


def _clean_json_payload(text: str) -> str:
    """Extract JSON content from raw agent output, tolerating code fences and stray prose."""
    stripped = text.strip()
    if not stripped:
        return stripped

    fence = re.search(r"```json\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1].strip()
        if candidate:
            return candidate

    return stripped


def parse_agent_response(payload: str) -> AgentResponse:
    """Parse and validate a JSON agent response."""
    cleaned = _clean_json_payload(payload)
    return AgentResponse.model_validate_json(cleaned)


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
    env.filters["tojson"] = json.dumps
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
    config: AppConfig,
    model: str | None = None,
    run_command: str | None,
    attach_recent: bool,
    include_diff: bool = False,
    ignore_dirs: Sequence[str],
    max_file_context_chars: int,
    max_command_output_chars: int,
    web_access: bool = False,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> PreparedPrompt:
    """
    Builds a structured, JSON-oriented prompt for Codex.
    """
    active_model = model or config.default_model
    model_limit = config.model_context_limits.get(active_model, config.model_context_limits.get("default", max_file_context_chars))
    context_limit = min(max_file_context_chars, model_limit)

    branch, commit, is_dirty = await _collect_git_context(root)

    repository_map = await asyncio.to_thread(generate_map, root, 3)
    constitution_text = await _load_constitution(root)

    attached: list[Path] = []

    diagnostic_section = ""
    file_context_section = ""
    dependency_section = ""
    relevant_memories: list[str] = []
    boosted_tags: list[str] = []

    # 2. Command Output (Diagnostic)
    if run_command:
        output, detected_files = await gather_context(run_command, root)
        trimmed = output if len(output) <= max_command_output_chars else _summarize_stack_trace(output, max_command_output_chars)
        diagnostic_section = (
            f"Command: {run_command}\n"
            f"Output (summary up to {max_command_output_chars} chars):\n"
            f"{trimmed}\n"
        )
        diag_lines = diagnostic_section.splitlines()
        query = "\n".join(diag_lines[-3:]).strip()
        if query:
            try:
                relevant_memories = await asyncio.to_thread(query_memory, query, 3, config)
            except Exception as exc:
                logger.debug("Memory query failed: %s", exc)

        # Prioritize files detected in the stack trace
        detected_paths = list(sorted(detected_files))[:5]
        file_context_section, attached = await _build_file_context_section(detected_paths, context_limit)
        dependency_section = await _build_dependency_section(detected_paths, root, context_limit)

    # 3. Recent Context (Ambient)
    elif attach_recent:
        recents = await scan_recent(root, 3, False, set(ignore_dirs))
        recent_paths = [entry.path for entry in recents[:5]]
        file_context_section, attached = await _build_file_context_section(recent_paths, context_limit)
        dependency_section = await _build_dependency_section(recent_paths[:1], root, context_limit)

    if not relevant_memories:
        base_query = base_prompt.strip()
        lowered_prompt = base_query.lower()
        for tag in ("architecture", "security"):
            if tag in lowered_prompt:
                boosted_tags.append(tag)
        boosted_query = f"{base_query}\nTags: {' '.join(boosted_tags)}" if boosted_tags else base_query
        if boosted_query:
            try:
                relevant_memories = await asyncio.to_thread(query_memory, boosted_query, 3, config)
            except Exception as exc:
                logger.debug("Memory query from base prompt failed: %s", exc)

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
        "relevant_memories": relevant_memories,
        "web_tool": "Web search and page retrieval is available via fetch_page_content(url) returning markdown." if web_access else "",
    }

    prompt = await asyncio.to_thread(_render_prompt_from_template, context, template_root)

    return PreparedPrompt(prompt=prompt, attached_files=attached, temperature=temperature, reasoning_effort=reasoning_effort)


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

    def _count_lines(target: Path) -> int:
        try:
            with target.open("r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except (OSError, UnicodeDecodeError):
            return 0

    for path in paths:
        line_count = await asyncio.to_thread(_count_lines, path)
        use_skeleton = path.suffix.lower() == ".py" and line_count > 200

        if use_skeleton:
            snippet = await asyncio.to_thread(get_file_skeleton, path)
        else:
            snippet = await asyncio.to_thread(smart_read_context, path, max_file_context_chars)

        if not snippet:
            continue
        label = " (Skeleton - Request full content if needed)" if use_skeleton else ""
        sections.append(f"Path: {path}{label}\n---\n{snippet}\n")
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
    if len(combined) <= limit:
        return combined
    return _summarize_stack_trace(combined, limit)


def _summarize_stack_trace(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    lines = text.splitlines()
    if len(text) <= limit:
        return text
    if len(lines) < 4:
        return text[:limit] + "... [truncated]"

    head_keep = max(3, min(12, len(lines) // 3))
    tail_keep = max(6, min(20, len(lines) // 2))
    head_lines = lines[:head_keep]
    tail_lines = lines[-tail_keep:]
    middle_lines = lines[head_keep:-tail_keep] if tail_keep < len(lines) - head_keep else []

    middle_summary = ""
    if middle_lines:
        mid_idx = len(middle_lines) // 2
        window = middle_lines[max(0, mid_idx - 3) : min(len(middle_lines), mid_idx + 4)]
        middle_summary = "\n[... middle truncated ...]\n" + "\n".join(window) + "\n[... resumes ...]\n"

    assembled = "\n".join(head_lines) + middle_summary + "\n".join(tail_lines)
    if len(assembled) > limit:
        head_budget = max(limit // 3, 1)
        tail_budget = max(limit - head_budget - 40, 1)
        trimmed_head = "\n".join(lines)[:head_budget]
        trimmed_tail = "\n".join(lines)[-tail_budget:]
        return f"{trimmed_head}\n[... truncated for length ...]\n{trimmed_tail}"

    return assembled


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


def _detect_repeated_failure(history: Sequence[AttemptContext], current_error: str) -> bool:
    normalized_current = current_error.strip()
    if not normalized_current:
        return False
    occurrences = sum(1 for attempt in history if attempt.last_error.strip() == normalized_current)
    return occurrences + 1 >= 2


def _build_repair_instruction(
    base_prompt: str,
    current_error: str,
    history: Sequence[AttemptContext],
    root: Path,
    *,
    strategy_override: str | None = None,
    reasoning_hint: str | None = None,
) -> str:
    history_block = _build_history_summary(history, root)
    override_block = f"\n\nStrategy Override:\n{strategy_override}" if strategy_override else ""
    reasoning_block = f"\n\nHigh reasoning effort requested: {reasoning_hint}" if reasoning_hint else ""
    return (
        f"{base_prompt.strip()}\n\n"
        "Autonomous repair loop in progress. Use the failure details to craft a minimal fix.\n"
        f"Current error:\n{current_error.strip()}\n\n"
        f"Previous attempts:\n{history_block}{override_block}{reasoning_block}\n\n"
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


async def _verify_syntax(files: list[Path]) -> str | None:
    """Verify Python syntax for changed files using py_compile."""
    py_files = [path for path in files if path.suffix == ".py"]
    if not py_files:
        return None

    for path in py_files:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "py_compile",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return "Python interpreter not found for syntax check."
        except Exception as exc:  # pragma: no cover - defensive
            return f"Syntax check failed for {path}: {exc}"

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            message = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
            return f"Syntax error in {path}: {message or 'py_compile failed'}"

    return None


async def _archive_session_summary(
    fetch_response: ResponseFetcher,
    *,
    base_prompt: str,
    command: str,
    last_error: str | None,
    config: AppConfig,
    model: str | None,
    web_access: bool = False,
) -> None:
    summary_prompt = (
        "Summarize the error fixed and the solution applied in one sentence for a knowledge base.\n"
        f"Command: {command}\n"
        f"Task: {base_prompt}\n"
        f"Last error before success: {last_error or 'N/A'}"
    )
    prepared = PreparedPrompt(prompt=summary_prompt, attached_files=[])
    try:
        raw_summary = await fetch_response(prepared)
    except Exception as exc:
        logger.debug("Summary fetch failed: %s", exc)
        return

    if not raw_summary.strip():
        return

    summary_text = raw_summary.strip()
    try:
        parsed = parse_agent_response(summary_text)
        summary_text = parsed.final_message or parsed.thought_process or summary_text
    except ValidationError:
        pass

    try:
        archive_config = config.model_copy(update={"use_semantic_search": False}) if hasattr(config, "model_copy") else config
        await asyncio.to_thread(save_memory, summary_text, ["auto-fix", "agent"], config=archive_config)
    except Exception as exc:
        logger.debug("Failed to archive repair summary: %s", exc)


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
    model: str | None,
    attach_recent: bool,
    include_diff: bool,
    fetch_response: ResponseFetcher,
    auto_archive: bool = True,
    max_retries: int = 3,
    keep_failed: bool = False,
    web_access: bool = False,
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
            if auto_archive:
                await _archive_session_summary(
                    fetch_response,
                    base_prompt=base_prompt,
                    command=command,
                    last_error=history[-1].last_error if history else None,
                    config=config,
                    model=model,
                    web_access=web_access,
                )
            return True

        current_error = _summarize_output(stdout, stderr, config.max_command_output_chars)
        console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {current_error}")

        loop_detected = _detect_repeated_failure(history, current_error)
        strategy_override = STRATEGY_OVERRIDE_TEXT if loop_detected else None
        reasoning_hint = "Increase temperature or reasoning effort to escape repetition." if loop_detected else None
        temperature_override = 0.7 if loop_detected else None
        if loop_detected:
            console.print("[yellow]Repeated failure detected; applying strategy override and higher reasoning effort.[/yellow]")

        iteration_prompt = _build_repair_instruction(
            base_prompt,
            current_error,
            history,
            root,
            strategy_override=strategy_override,
            reasoning_hint=reasoning_hint,
        )
        prepared = await prepare_agent_prompt(
            iteration_prompt,
            root=root,
            config=config,
            model=model,
            run_command=None,
            attach_recent=attach_recent,
            include_diff=include_diff,
            ignore_dirs=config.ignore_dirs,
            max_file_context_chars=config.max_file_context_chars,
            max_command_output_chars=config.max_command_output_chars,
            temperature=temperature_override,
            reasoning_effort="high" if loop_detected else None,
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
        syntax_error = await _verify_syntax(applied_paths)
        if syntax_error:
            console.print(f"[red]Syntax Check Failed (Self-Correction):[/red] {syntax_error}")
            history.append(AttemptContext(iteration=attempt + 1, last_error=syntax_error, files_changed=applied_paths))
            changed_files.update(applied_paths)
            current_error = syntax_error
            continue

        history.append(AttemptContext(iteration=attempt + 1, last_error=current_error, files_changed=applied_paths))
        changed_files.update(applied_paths)

    console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
    exit_code, stdout, stderr = await _run_shell_command(command, root)
    if exit_code == 0:
        console.print("[green]Command succeeded after final verification.[/green]")
        if auto_archive:
            await _archive_session_summary(
                fetch_response,
                base_prompt=base_prompt,
                command=command,
                last_error=None if not history else history[-1].last_error,
                config=config,
                model=model,
                web_access=web_access,
            )
        return True

    console.print(f"[red]Command still failing:[/red] {_summarize_output(stdout, stderr, config.max_command_output_chars)}")
    if changed_files and not keep_failed:
        console.print("[yellow]Reverting changes from failed attempts.[/yellow]")
        await _revert_changed_files(list(changed_files), root)

    return False
