"""Prompt building and template rendering for agent interactions.

This module provides the core prompt preparation logic, including
template rendering, context assembly, and token budget management.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from jpscripts.core import security
from jpscripts.core.agent.context import (
    build_dependency_section,
    build_file_context_section,
    collect_git_context,
    collect_git_diff,
    load_constitution,
)
from jpscripts.core.console import get_logger
from jpscripts.core.context_gatherer import gather_context, smart_read_context
from jpscripts.core.engine import AgentResponse, PreparedPrompt
from jpscripts.core.memory import fetch_relevant_patterns, format_patterns_for_prompt, query_memory
from jpscripts.core.nav import scan_recent
from jpscripts.core.result import Err, Ok
from jpscripts.core.runtime import get_runtime
from jpscripts.core.structure import generate_map
from jpscripts.core.tokens import TokenBudgetManager

logger = get_logger(__name__)

AGENT_TEMPLATE_NAME = "agent_system.json.j2"
GOVERNANCE_ANTI_PATTERNS: list[str] = [
    "Using subprocess.run or os.system (Strictly forbidden: use asyncio)",
    "Using shell=True (Strictly forbidden: use tokenized lists)",
    "Bare except: clauses (Strictly forbidden: catch specific exceptions)",
]


def _resolve_template_root() -> Path:
    package_root = Path(__file__).resolve().parent.parent.parent
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
        raise FileNotFoundError(
            f"Template {AGENT_TEMPLATE_NAME} not found in {template_root}"
        ) from exc
    return template.render(**context)


def _safe_cdata(content: str) -> str:
    """Escape CDATA terminators inside arbitrary content."""
    return content.replace("]]>", "]]]]><![CDATA[>")


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
        middle_summary = (
            "\n[... middle truncated ...]\n" + "\n".join(window) + "\n[... resumes ...]\n"
        )

    assembled = "\n".join(head_lines) + middle_summary + "\n".join(tail_lines)
    if len(assembled) > limit:
        head_budget = max(limit // 3, 1)
        tail_budget = max(limit - head_budget - 40, 1)
        trimmed_head = "\n".join(lines)[:head_budget]
        trimmed_tail = "\n".join(lines)[-tail_budget:]
        return f"{trimmed_head}\n[... truncated for length ...]\n{trimmed_tail}"

    return assembled


async def prepare_agent_prompt(
    base_prompt: str,
    *,
    model: str | None = None,
    run_command: str | None,
    attach_recent: bool,
    include_diff: bool = False,
    ignore_dirs: Sequence[str] | None = None,
    max_file_context_chars: int | None = None,
    max_command_output_chars: int | None = None,
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
    runtime = get_runtime()
    config = runtime.config
    root = runtime.workspace_root
    effective_ignore_dirs = (
        list(ignore_dirs) if ignore_dirs is not None else list(config.ignore_dirs)
    )
    file_context_limit = (
        max_file_context_chars
        if max_file_context_chars is not None
        else config.max_file_context_chars
    )
    command_output_limit = (
        max_command_output_chars
        if max_command_output_chars is not None
        else config.max_command_output_chars
    )
    active_model = model or config.default_model
    model_limit = config.model_context_limits.get(
        active_model,
        config.model_context_limits.get("default", file_context_limit),
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

    branch, commit, is_dirty = await collect_git_context(root)

    repository_map = await asyncio.to_thread(generate_map, root, 3)
    constitution_text = await load_constitution(root)

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
        ordered_detected = list(gathered_context.ordered_files)
        trimmed = (
            output
            if len(output) <= command_output_limit
            else _summarize_stack_trace(output, command_output_limit)
        )
        raw_diagnostic = (
            f"Command: {run_command}\n"
            f"Output (summary up to {command_output_limit} chars):\n"
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

        ordered_sources = ordered_detected if ordered_detected else sorted(detected_files)
        detected_paths = list(dict.fromkeys(ordered_sources))[:5]

    elif attach_recent:
        match await scan_recent(root, 3, False, set(effective_ignore_dirs)):
            case Err(err):
                logger.debug("Recent scan failed for %s: %s", root, err)
            case Ok(recents):
                detected_paths = [entry.path for entry in recents[:5]]

    # === Priority 2: Git Diff Section (medium priority) ===
    # === Priority 2 & 3: File Context + Dependencies (Sequential Greedy) ===
    # Files get what they need first (direct source prioritized), dependencies get leftovers
    combined_paths: list[Path] = detected_paths + extra_detected
    if budget.remaining() > 0 and combined_paths:
        file_context_section, attached = await build_file_context_section(combined_paths, budget)

        # Dependencies only get leftover budget after files
        if budget.remaining() > 0:
            dependency_section = await build_dependency_section(combined_paths[:1], root, budget)

    # Git diff is lowest priority after files and dependencies
    if include_diff and budget.remaining() > 0:
        diff_text = await collect_git_diff(root, 10_000)
        git_diff_section = budget.allocate(3, diff_text) if diff_text else "NO CHANGES"

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


__all__ = [
    "AGENT_TEMPLATE_NAME",
    "GOVERNANCE_ANTI_PATTERNS",
    "prepare_agent_prompt",
]
