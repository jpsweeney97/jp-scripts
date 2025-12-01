"""CLI command modules for the jp toolkit.

This package contains all user-facing CLI commands organized by domain:
    - agent: AI agent operations and code repair
    - git_extra: Enhanced git operations (branches, commits, conflicts)
    - git_ops: Core git utilities
    - memory: Memory store management
    - nav: Directory navigation
    - notes: Note-taking and clipboard history
    - search: Codebase search
    - system: System utilities (process/port management)
    - team: Team collaboration features
    - watch: File watching and auto-reload
    - web: Web scraping and fetching
"""

from __future__ import annotations

from . import (
    agent,
    git_extra,
    git_ops,
    init,
    memory,
    nav,
    notes,
    search,
    serialize,
    system,
    team,
    watch,
    web,
)

__all__ = [
    "agent",
    "git_extra",
    "git_ops",
    "init",
    "memory",
    "nav",
    "notes",
    "search",
    "serialize",
    "system",
    "team",
    "watch",
    "web",
]
