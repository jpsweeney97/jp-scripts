"""MCP memory tools for vector store operations.

Provides tools for memory store access:
    - remember: Store facts and lessons
    - recall: Query memories by semantic similarity
    - recall_by_tag: Query memories by tag
"""

from __future__ import annotations

import asyncio

from jpscripts import memory as memory_core
from jpscripts.core.config import AppConfig
from jpscripts.core.result import JPScriptsError
from jpscripts.core.runtime import get_runtime
from jpscripts.mcp import tool, tool_error_handler


@tool()
@tool_error_handler
async def remember(fact: str, tags: str | None = None) -> str:
    """
    Save a fact or lesson to the persistent memory store.
    Tags can be provided as a comma-separated list.
    """
    ctx = get_runtime()
    tag_list = [t.strip() for t in (tags.split(",") if tags else []) if t.strip()]
    return await asyncio.to_thread(_save_memory, fact, tag_list, ctx.config)


@tool()
@tool_error_handler
async def recall(query: str, limit: int = 5) -> str:
    """Retrieve the most relevant memories for a query."""
    ctx = get_runtime()
    return await asyncio.to_thread(_recall_memories, query, limit, ctx.config)


def _save_memory(fact: str, tag_list: list[str], cfg: AppConfig) -> str:
    try:
        entry = memory_core.save_memory(fact, tag_list, config=cfg)
        return f"Saved memory at {entry.ts}"
    except JPScriptsError as exc:
        return f"Error saving memory: {exc}"


def _recall_memories(query: str, limit: int, cfg: AppConfig) -> str:
    try:
        results = memory_core.query_memory(query, limit=limit, config=cfg)
        return "\n".join(results) if results else "No matching memories."
    except JPScriptsError as exc:
        return f"Error recalling memories: {exc}"
