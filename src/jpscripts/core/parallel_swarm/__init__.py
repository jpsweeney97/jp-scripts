"""Parallel swarm controller with git worktree isolation.

This package provides the ParallelSwarmController for executing DAG-based
tasks in parallel using isolated git worktrees. Each parallel task runs
in its own worktree to prevent filesystem conflicts and git index.lock
contention.

Key classes:
- WorktreeManager: Manages lifecycle of git worktrees for task isolation
- TaskResult: Result of executing a single task
- ParallelSwarmController: Orchestrates parallel task execution

[invariant:typing] All types are explicit; mypy --strict compliant.
[invariant:async-io] All I/O operations use async patterns.
"""

from jpscripts.core.parallel_swarm.controller import ParallelSwarmController
from jpscripts.core.parallel_swarm.types import MergeResult, TaskResult
from jpscripts.core.parallel_swarm.worktree import WorktreeManager

__all__ = [
    "MergeResult",
    "ParallelSwarmController",
    "TaskResult",
    "WorktreeManager",
]
