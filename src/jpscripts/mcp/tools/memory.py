from __future__ import annotations

import asyncio

from jpscripts.core import memory as memory_core
from jpscripts.core.config import AppConfig
from jpscripts.mcp import get_config, tool


@tool()
async def remember(fact: str, tags: str | None = None) -> str:
    """
    Save a fact or lesson to the persistent memory store.
    Tags can be provided as a comma-separated list.
    """
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    tag_list = [t.strip() for t in (tags.split(",") if tags else []) if t.strip()]
    return await asyncio.to_thread(_save_memory, fact, tag_list, cfg)


@tool()
async def recall(query: str, limit: int = 5) -> str:
    """Retrieve the most relevant memories for a query."""
    cfg = get_config()
    if cfg is None:
        return "Config not loaded."

    return await asyncio.to_thread(_recall_memories, query, limit, cfg)


def _save_memory(fact: str, tag_list: list[str], cfg: AppConfig) -> str:
    entry = memory_core.save_memory(fact, tag_list, config=cfg)
    return f"Saved memory at {entry.ts}"


def _recall_memories(query: str, limit: int, cfg: AppConfig) -> str:
    results = memory_core.query_memory(query, limit=limit, config=cfg)
    return "\n".join(results) if results else "No matching memories."
