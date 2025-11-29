from __future__ import annotations

import asyncio
import json
import shlex
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable, Literal, Sequence

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import ValidationError
from rich import box
from rich.panel import Panel

from jpscripts.core import git as git_core
from jpscripts.core import git_ops
from jpscripts.core import security
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console, get_logger
from jpscripts.core.context_gatherer import gather_context, get_file_skeleton, resolve_files_from_output, smart_read_context
from jpscripts.core.tokens import TokenBudgetManager
from jpscripts.core.engine import (
    AgentEngine,
    AgentResponse,
    Message,
    PreparedPrompt,
    ToolCall,
    parse_agent_response,
)
from jpscripts.core.memory import query_memory, save_memory, fetch_relevant_patterns, format_patterns_for_prompt
from jpscripts.core.nav import scan_recent
from jpscripts.core.result import Err, Ok
from jpscripts.core.structure import generate_map, get_import_dependencies

logger = get_logger(__name__)

AGENT_TEMPLATE_NAME = "agent_system.json.j2"
GOVERNANCE_ANTI_PATTERNS: list[str] = [
    "Using subprocess.run or os.system (Strictly forbidden: use asyncio)",
    "Using shell=True (Strictly forbidden: use tokenized lists)",
    "Bare except: clauses (Strictly forbidden: catch specific exceptions)",
]
STRATEGY_OVERRIDE_TEXT = (
    "You are stuck in a loop. Stop editing code. Analyze the error trace and the file content again. "
    "List three possible root causes before proposing a new patch."
)
_ACTIVE_ROOT: Path | None = None


class SecurityError(RuntimeError):
    """Raised when a tool invocation is considered unsafe."""


@dataclass
class AttemptContext:
    iteration: int
    last_error: str
    files_changed: list[Path]
    strategy: Literal["fast", "deep", "step_back"]


RepairStrategy = Literal["fast", "deep", "step_back"]


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
    tool_history: str | None = None,
    extra_paths: Sequence[Path] | None = None,
) -> PreparedPrompt:
    """
    Builds a structured, JSON-oriented prompt for Codex.

    Uses priority-based token budget allocation:
    - Priority 1: Diagnostic output (command failures, stack traces)
    - Priority 2: Git diff (current work in progress)
    - Priority 3: File context and dependencies (supporting information)
    """
    active_model = model or config.default_model
    model_limit = config.model_context_limits.get(
        active_model,
        config.model_context_limits.get("default", max_file_context_chars),
    )

    # Reserve ~10% for template overhead (prompt structure, instructions, etc.)
    template_overhead = min(50_000, int(model_limit * 0.1))
    budget = TokenBudgetManager(
        total_budget=model_limit,
        reserved_budget=template_overhead,
        model_context_limit=model_limit,
        model=active_model,
        truncator=smart_read_context,
    )

    branch, commit, is_dirty = await _collect_git_context(root)

    repository_map = await asyncio.to_thread(generate_map, root, 3)
    constitution_text = await _load_constitution(root)

    attached: list[Path] = []
    detected_paths: list[Path] = []
    extra_detected = list(extra_paths) if extra_paths else []

    diagnostic_section = ""
    file_context_section = ""
    dependency_section = ""
    git_diff_section = ""
    relevant_memories: list[str] = []
    boosted_tags: list[str] = []

    # === Priority 1: Diagnostic Section (highest priority) ===
    if run_command:
        gathered_context = await gather_context(run_command, root)
        output = gathered_context.output
        detected_files = gathered_context.files
        trimmed = (
            output
            if len(output) <= max_command_output_chars
            else _summarize_stack_trace(output, max_command_output_chars)
        )
        raw_diagnostic = (
            f"Command: {run_command}\n"
            f"Output (summary up to {max_command_output_chars} chars):\n"
            f"{trimmed}\n"
        )
        diagnostic_section = budget.allocate(1, raw_diagnostic)

        diag_lines = diagnostic_section.splitlines()
        query = "\n".join(diag_lines[-3:]).strip()
        if query:
            try:
                relevant_memories = await asyncio.to_thread(
                    lambda: query_memory(query, 3, config=config)
                )
            except Exception as exc:
                logger.debug("Memory query failed: %s", exc)

        detected_paths = list(sorted(detected_files))[:5]

    elif attach_recent:
        match await scan_recent(root, 3, False, set(ignore_dirs)):
            case Err(err):
                logger.debug("Recent scan failed for %s: %s", root, err)
            case Ok(recents):
                detected_paths = [entry.path for entry in recents[:5]]

    # === Priority 2: Git Diff Section (medium priority) ===
    if include_diff:
        diff_text = await _collect_git_diff(root, 10_000)
        if diff_text:
            git_diff_section = budget.allocate(2, diff_text)
        else:
            git_diff_section = "NO CHANGES"

    # === Priority 3: File Context + Dependencies (Sequential Greedy) ===
    # Files get what they need first, dependencies get whatever remains
    combined_paths: list[Path] = detected_paths + extra_detected
    if budget.remaining() > 0 and combined_paths:
        file_context_section, attached = await _build_file_context_section(
            combined_paths, budget
        )

        # Dependencies only get leftover budget after files
        if budget.remaining() > 0:
            dependency_section = await _build_dependency_section(
                combined_paths[:1], root, budget
            )

    # Memory query fallback
    if not relevant_memories:
        base_query = base_prompt.strip()
        lowered_prompt = base_query.lower()
        for tag in ("architecture", "security"):
            if tag in lowered_prompt:
                boosted_tags.append(tag)
        boosted_query = (
            f"{base_query}\nTags: {' '.join(boosted_tags)}" if boosted_tags else base_query
        )
        if boosted_query:
            try:
                relevant_memories = await asyncio.to_thread(
                    lambda: query_memory(boosted_query, 3, config=config)
                )
            except Exception as exc:
                logger.debug("Memory query from base prompt failed: %s", exc)

    # Fetch relevant patterns for prompt injection
    patterns_section = ""
    try:
        patterns = await fetch_relevant_patterns(
            base_prompt.strip() or diagnostic_section[:500],
            config,
            limit=2,
            min_confidence=0.75,
        )
        if patterns:
            patterns_section = format_patterns_for_prompt(patterns)
            logger.debug("Injecting %d patterns into prompt", len(patterns))
    except Exception as exc:
        logger.debug("Pattern fetch failed: %s", exc)

    logger.debug(
        "Token budget allocation: %s, remaining: %d",
        budget.summary(),
        budget.remaining(),
    )

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
        "patterns_section": patterns_section,
        "anti_patterns": GOVERNANCE_ANTI_PATTERNS,
        "instruction": base_prompt.strip(),
        "tool_history": tool_history or "",
        "response_schema": response_schema,
        "relevant_memories": relevant_memories,
        "web_tool": (
            "Web search and page retrieval is available via fetch_page_content(url) returning markdown."
            if web_access
            else ""
        ),
    }

    prompt = await asyncio.to_thread(_render_prompt_from_template, context, template_root)

    return PreparedPrompt(
        prompt=prompt,
        attached_files=attached,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def _safe_cdata(content: str) -> str:
    """Escape CDATA terminators inside arbitrary content."""
    return content.replace("]]>", "]]]]><![CDATA[>")


async def _load_constitution(root: Path) -> dict[str, object]:
    """Load and validate the constitutional JSON from AGENTS.md."""
    try:
        candidate = security.validate_path(root / "AGENTS.md", root)
    except Exception as exc:
        logger.debug("Unable to resolve AGENTS.md under %s: %s", root, exc)
        return {"status": "unavailable", "message": "AGENTS.md not accessible", "error": str(exc)}

    exists = await asyncio.to_thread(candidate.exists)
    if not exists:
        return {"status": "missing", "message": "AGENTS.md not found"}

    try:
        content = await asyncio.to_thread(candidate.read_text, encoding="utf-8")
    except OSError as exc:
        logger.debug("Failed to read AGENTS.md: %s", exc)
        return {"status": "unreadable", "message": "AGENTS.md unreadable", "error": str(exc)}

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse AGENTS.md as JSON: %s", exc)
        return {"status": "parse_error", "message": str(exc)}

    if not isinstance(parsed, dict):
        return {"status": "invalid", "message": "AGENTS.md root must be an object"}

    constitution = parsed.get("constitution")
    if not isinstance(constitution, dict):
        return {"status": "invalid", "message": "Missing or invalid 'constitution' object"}

    return constitution


async def _collect_git_context(root: Path) -> tuple[str, str, bool]:
    if not root.exists() or not (root / ".git").exists():
        return "(no repo)", "(no repo)", False

    match await git_core.AsyncRepo.open(root):
        case Err(err):
            logger.error("Failed to open git repo at %s: %s", root, err)
            return "(error)", "(error)", False
        case Ok(repo):
            pass

    branch = "(unknown)"
    commit = "(unknown)"
    is_dirty = False

    match await repo.status():
        case Err(err):
            logger.error("Failed to describe git status for %s: %s", root, err)
            return "(error)", "(error)", False
        case Ok(status):
            branch = status.branch
            is_dirty = status.dirty
            _ = git_ops.format_status(status)

    match await repo.head(short=True):
        case Err(err):
            logger.error("Failed to resolve git head for %s: %s", root, err)
            commit = "(error)"
        case Ok(head_ref):
            commit = head_ref

    return branch, commit, is_dirty


async def _build_file_context_section(
    paths: Sequence[Path],
    budget: TokenBudgetManager,
) -> tuple[str, list[Path]]:
    """Build file context section using sequential greedy allocation.

    Each file is read only if budget remains, and allocated individually
    to preserve syntax boundaries per file.
    """
    sections: list[str] = []
    attached: list[Path] = []

    def _count_lines(target: Path) -> int:
        try:
            with target.open("r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except (OSError, UnicodeDecodeError):
            return 0

    for path in paths:
        # Check budget before reading each file
        remaining_tokens = budget.remaining()
        if remaining_tokens <= 0:
            break
        char_budget = budget.tokens_to_characters(remaining_tokens)

        line_count = await asyncio.to_thread(_count_lines, path)
        use_skeleton = path.suffix.lower() == ".py" and line_count > 200

        if use_skeleton:
            snippet = await asyncio.to_thread(get_file_skeleton, path)
        else:
            # Read file with syntax-aware truncation up to remaining budget
            snippet = await asyncio.to_thread(
                smart_read_context,
                path,
                char_budget,
                remaining_tokens,
                limit=budget.tokens_to_characters(budget.model_context_limit),
            )

        if not snippet:
            continue

        label = " (Skeleton - Request full content if needed)" if use_skeleton else ""
        file_entry = f"Path: {path}{label}\n---\n{snippet}\n"

        # Allocate this file's content - may be truncated if over budget
        allocated = budget.allocate(3, file_entry, source_path=path)
        if allocated:
            sections.append(allocated)
            attached.append(path)

    if not sections:
        return "", attached
    return "\n".join(sections), attached


async def _build_dependency_section(
    paths: Sequence[Path],
    root: Path,
    budget: TokenBudgetManager,
) -> str:
    """Build dependency section using sequential greedy allocation.

    Dependencies are read only if budget remains after file context.
    """
    dependencies: set[Path] = set()
    for path in paths:
        deps = await asyncio.to_thread(get_import_dependencies, path, root)
        dependencies.update(deps)

    if not dependencies:
        return ""

    sections: list[str] = []
    for dep in sorted(dependencies):
        # Check budget before reading each dependency
        remaining_tokens = budget.remaining()
        if remaining_tokens <= 0:
            break
        char_budget = budget.tokens_to_characters(remaining_tokens)

        snippet = await asyncio.to_thread(
            smart_read_context,
            dep,
            char_budget,
            remaining_tokens,
            limit=budget.tokens_to_characters(budget.model_context_limit),
        )
        if not snippet:
            continue

        dep_entry = f"Dependency: {dep}\n---\n{snippet}\n"

        # Allocate this dependency's content
        allocated = budget.allocate(3, dep_entry, source_path=dep)
        if allocated:
            sections.append(allocated)

    return "\n".join(sections)


async def _collect_git_diff(root: Path, max_chars: int) -> str | None:
    if not root.exists() or not (root / ".git").exists():
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            "HEAD",
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


def _append_history(history: list[Message], entry: Message, keep: int = 3) -> None:
    history.append(entry)
    if len(history) > keep:
        del history[:-keep]


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
    strategy: Literal["fast", "deep", "step_back"] = "fast",
) -> str:
    history_block = _build_history_summary(history, root)
    override_block = f"\n\nStrategy Override:\n{strategy_override}" if strategy_override else ""
    reasoning_block = f"\n\nHigh reasoning effort requested: {reasoning_hint}" if reasoning_hint else ""
    strategy_block = ""
    if strategy == "deep":
        strategy_block = (
            "\n\nSystem Notice: Attempt 1 failed. Context has been expanded to include imported "
            "dependencies and referenced modules. Analyze interactions across modules."
        )
    elif strategy == "step_back":
        strategy_block = (
            "\n\nSystem Notice: Attempt 2 failed. Tool use is disabled for this turn. "
            "Perform Root Cause Analysis and propose a brief plan before patching."
        )
    return (
        f"{base_prompt.strip()}\n\n"
        "Autonomous repair loop in progress. Use the failure details to craft a minimal fix.\n"
        f"Current error:\n{current_error.strip()}\n\n"
        f"Previous attempts:\n{history_block}{override_block}{reasoning_block}{strategy_block}\n\n"
        "Respond with a single JSON object that matches the AgentResponse schema. "
        "Place the unified diff in `file_patch`. Do not return Markdown or prose."
    )


async def _run_shell_command(command: str, cwd: Path) -> tuple[int, str, str]:
    """Executes command without shell interpolation."""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        logger.warning("Failed to parse shell command: %s", exc)
        return 1, "", f"Unable to parse command; simplify quoting. ({exc})"

    if not tokens:
        return 1, "", "Invalid command."

    try:
        logger.debug("Running safe command: %s", tokens)
        proc = await asyncio.create_subprocess_exec(
            *tokens,
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


async def _expand_context_paths(
    error_output: str,
    root: Path,
    changed_files: set[Path],
    ignore_dirs: Sequence[str],
) -> set[Path]:
    """Derive additional context paths from the latest failure."""
    discovered: set[Path] = set()
    discovered.update(changed_files)
    discovered.update(resolve_files_from_output(error_output, root))

    dependencies: set[Path] = set()
    for path in discovered:
        try:
            deps = await asyncio.to_thread(get_import_dependencies, path, root)
            dependencies.update(deps)
        except Exception as exc:
            logger.debug("Dependency discovery failed for %s: %s", path, exc)

    if not discovered and not dependencies:
        match await scan_recent(root, 3, False, set(ignore_dirs)):
            case Err(err):
                logger.debug("Recent scan fallback failed: %s", err)
            case Ok(recents):
                discovered.update(entry.path for entry in recents[:3])

    return {security.validate_path(path, root) for path in (discovered | dependencies) if path.exists()}


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
    Execute an autonomous repair loop using AgentEngine for orchestration.
    """
    global _ACTIVE_ROOT
    root = security.validate_workspace_root(config.workspace_root or config.notes_dir)
    attempt_cap = max(1, max_retries)
    changed_files: set[Path] = set()
    attempt_history: list[AttemptContext] = []
    history: list[Message] = []
    previous_active_root = _ACTIVE_ROOT
    _ACTIVE_ROOT = root

    async def _prompt_builder(
        history_messages: Sequence[Message],
        iteration_prompt: str,
        loop_detected_flag: bool,
        temp_override: float | None,
        strategy: RepairStrategy,
        extra_paths: Sequence[Path],
    ) -> PreparedPrompt:
        history_text = "\n".join(msg.content for msg in history_messages)
        instruction = iteration_prompt
        if history_text:
            instruction = f"{instruction}\n\nPrevious tool interactions:\n{history_text}"
        reasoning = "high" if loop_detected_flag or strategy == "step_back" else None
        run_cmd = command if strategy in {"fast", "deep"} else None
        return await prepare_agent_prompt(
            instruction,
            root=root,
            config=config,
            model=model,
            run_command=run_cmd,
            attach_recent=attach_recent,
            include_diff=include_diff,
            ignore_dirs=config.ignore_dirs,
            max_file_context_chars=config.max_file_context_chars,
            max_command_output_chars=config.max_command_output_chars,
            reasoning_effort=reasoning,
            temperature=temp_override,
            tool_history=history_text,
            extra_paths=extra_paths,
            web_access=web_access,
        )

    async def _fetch(prepared: PreparedPrompt) -> str:
        return await fetch_response(prepared)

    try:
        for attempt in range(attempt_cap):
            strategy: RepairStrategy = "fast" if attempt == 0 else "deep" if attempt == 1 else "step_back"
            console.print(
                f"[cyan]Attempt {attempt + 1}/{attempt_cap} ({strategy} strategy): running `{command}`[/cyan]"
            )
            exit_code, stdout, stderr = await _run_shell_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded. Exiting repair loop.[/green]")
                if auto_archive:
                    await _archive_session_summary(
                        fetch_response,
                        base_prompt=base_prompt,
                        command=command,
                        last_error=attempt_history[-1].last_error if attempt_history else None,
                        config=config,
                        model=model,
                        web_access=web_access,
                    )
                return True

            current_error = _summarize_output(stdout, stderr, config.max_command_output_chars)
            console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {current_error}")

            loop_detected = _detect_repeated_failure(attempt_history, current_error)
            strategy_override = STRATEGY_OVERRIDE_TEXT if loop_detected else None
            reasoning_hint = "Increase temperature or reasoning effort to escape repetition." if loop_detected else None
            temperature_override = 0.7 if loop_detected else None
            if loop_detected:
                console.print("[yellow]Repeated failure detected; applying strategy override and higher reasoning effort.[/yellow]")

            dynamic_paths: set[Path] = set(changed_files)
            if strategy == "deep":
                dynamic_paths = await _expand_context_paths(current_error, root, changed_files, config.ignore_dirs)
            elif strategy == "step_back":
                dynamic_paths = set(changed_files)

            applied_paths: list[Path] = []
            for turn in range(5):
                iteration_prompt = _build_repair_instruction(
                    base_prompt,
                    current_error,
                    attempt_history,
                    root,
                    strategy_override=strategy_override,
                    reasoning_hint=reasoning_hint,
                    strategy=strategy,
                )

                # Lambda with default args from captured scope; mypy cannot infer types
                # tools=None uses unified registry from get_tool_registry()
                # workspace_root enables governance checks for constitutional compliance
                engine = AgentEngine[AgentResponse](
                    persona="Engineer",
                    model=model or config.default_model,
                    prompt_builder=lambda msgs, ip=iteration_prompt, ld=loop_detected, temp=temperature_override, strat=strategy, paths=list(dynamic_paths): _prompt_builder(  # type: ignore[misc]
                        msgs, ip, ld, temp, strat, paths
                    ),
                    fetch_response=_fetch,
                    parser=parse_agent_response,
                    tools={} if strategy == "step_back" else None,
                    template_root=root,
                    workspace_root=root,
                    governance_enabled=True,
                )

                try:
                    agent_response = await engine.step(history)
                except ValidationError as exc:
                    validation_error = f"Agent response validation failed: {exc}"
                    console.print(f"[red]{validation_error}[/red]")
                    _append_history(
                        history,
                        Message(
                            role="system",
                            content=(
                                "<Turn>\nAgent thought: (invalid)\nTool output: "
                                f"{validation_error}\n</Turn>"
                            ),
                        ),
                    )
                    current_error = validation_error
                    continue

                tool_call: ToolCall | None = agent_response.tool_call
                patch_text = (agent_response.file_patch or "").strip()
                thought = agent_response.thought_process

                if tool_call:
                    tool_name = tool_call.tool
                    tool_args = tool_call.arguments
                    console.print(Panel(f"🕵️ Agent invoking {tool_name} with args {tool_args}", title="Tool Call", box=box.SIMPLE))
                    try:
                        output = await engine.execute_tool(tool_call)
                    except Exception as exc:
                        output = f"Tool execution failed: {exc}"
                    console.print(Panel(output, title="Tool Output", box=box.SIMPLE, style="cyan"))
                    history_entry = (
                        "<Turn>\n"
                        f"Agent thought: {thought}\n"
                        f"Tool call: {tool_name}({tool_args})\n"
                        f"Tool output: {output}\n"
                        "</Turn>"
                    )
                    _append_history(history, Message(role="system", content=history_entry))
                    continue

                if patch_text:
                    console.print("[green]⚡ Agent proposed a fix.[/green]")
                    applied_paths = await _apply_patch_text(patch_text, root)
                    syntax_error = await _verify_syntax(applied_paths)
                    if syntax_error:
                        console.print(f"[red]Syntax Check Failed (Self-Correction):[/red] {syntax_error}")
                        _append_history(
                            history,
                            Message(
                                role="system",
                                content=(
                                    "<Turn>\n"
                                    f"Agent thought: {thought}\n"
                                    "Tool call: none\n"
                                    f"Tool output: Syntax check failed: {syntax_error}\n"
                                    "</Turn>"
                                ),
                            ),
                        )
                        changed_files.update(applied_paths)
                        current_error = syntax_error
                        continue

                    changed_files.update(applied_paths)
                    break

                message = agent_response.final_message or "Agent returned no patch content."
                console.print(f"[yellow]{message}[/yellow]")
                _append_history(
                    history,
                    Message(
                        role="system",
                        content=(
                            "<Turn>\n"
                            f"Agent thought: {thought}\n"
                            "Tool call: none\n"
                            f"Tool output: {message}\n"
                            "</Turn>"
                        ),
                    ),
                )
                break

            exit_code, stdout, stderr = await _run_shell_command(command, root)
            if exit_code == 0:
                console.print("[green]Command succeeded after applying fixes.[/green]")
                if auto_archive:
                    await _archive_session_summary(
                        fetch_response,
                        base_prompt=base_prompt,
                        command=command,
                        last_error=current_error,
                        config=config,
                        model=model,
                        web_access=web_access,
                    )
                return True

            failure_msg = _summarize_output(stdout, stderr, config.max_command_output_chars)
            console.print(f"[yellow]Verification failed:[/yellow] {failure_msg}")
            attempt_history.append(
                AttemptContext(
                    iteration=attempt + 1,
                    last_error=failure_msg,
                    files_changed=list(changed_files),
                    strategy=strategy,
                )
            )
            _append_history(history, Message(role="system", content=f"Verification failure (attempt {attempt + 1}): {failure_msg}"))

        console.print("[yellow]Max retries reached. Verifying one last time...[/yellow]")
        exit_code, stdout, stderr = await _run_shell_command(command, root)
        if exit_code == 0:
            console.print("[green]Command succeeded after final verification.[/green]")
            if auto_archive:
                await _archive_session_summary(
                    fetch_response,
                    base_prompt=base_prompt,
                    command=command,
                    last_error=None if not attempt_history else attempt_history[-1].last_error,
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
    finally:
        _ACTIVE_ROOT = previous_active_root


# Re-export engine types for backwards compatibility
__all__ = [
    # Re-exported from engine
    "PreparedPrompt",
    "parse_agent_response",
    # Locally defined
    "prepare_agent_prompt",
    "run_repair_loop",
]
