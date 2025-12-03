"""Session archiving for agent repair loops.

This module provides functionality to archive successful repair sessions
to the memory store for future reference and learning.
"""

from __future__ import annotations

import asyncio

from pydantic import ValidationError

from jpscripts.agent.models import PreparedPrompt, ResponseFetcher
from jpscripts.agent.parsing import parse_agent_response
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.memory import save_memory

logger = get_logger(__name__)


async def archive_session_summary(
    fetch_response: ResponseFetcher,
    config: AppConfig,
    *,
    base_prompt: str,
    command: str,
    last_error: str | None,
    model: str | None,
    web_access: bool = False,
) -> None:
    """Archive a successful repair session summary to memory.

    Generates a one-sentence summary of the error and solution,
    then saves it to the memory store with relevant tags.

    Args:
        fetch_response: Function to fetch LLM responses.
        config: Application configuration.
        base_prompt: The original repair instruction.
        command: Shell command that was used to verify fixes.
        last_error: Last error message before success.
        model: LLM model ID used.
        web_access: Whether web access was enabled.
    """
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
        archive_config = (
            config.model_copy(update={"use_semantic_search": False})
            if hasattr(config, "model_copy")
            else config
        )
        await asyncio.to_thread(
            save_memory, summary_text, ["auto-fix", "agent"], config=archive_config
        )
    except Exception as exc:
        logger.debug("Failed to archive repair summary: %s", exc)


# Backward-compatible alias
_archive_session_summary = archive_session_summary

__all__ = [
    "_archive_session_summary",
    "archive_session_summary",
]
