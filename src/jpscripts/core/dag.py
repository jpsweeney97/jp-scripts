"""DAG-based task orchestration models for parallel swarm execution.

This module defines the data structures for representing tasks as a Directed
Acyclic Graph (DAG), enabling parallel execution of tasks with disjoint file
sets while respecting dependencies.

Key classes:
- DAGTask: Individual task node with objective, files touched, and dependencies
- DAGGraph: Complete graph with methods for scheduling and parallel execution
- WorktreeContext: Execution context for tasks running in isolated git worktrees
- TaskStatus: Enum for tracking task lifecycle

[invariant:typing] All types are explicit; mypy --strict compliant.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(Enum):
    """Lifecycle status for a task in the swarm."""

    PENDING = auto()  # Not yet started
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Finished successfully
    FAILED = auto()  # Finished with error
    BLOCKED = auto()  # Waiting on failed dependency


class DAGTask(BaseModel):
    """A single task node in the dependency graph.

    Each task represents a unit of work that can be assigned to an agent.
    Tasks declare their dependencies and the files they will touch, enabling
    parallel execution of tasks with disjoint file sets.

    Attributes:
        id: Unique task identifier (e.g., 'task-001')
        objective: What this task should accomplish
        files_touched: Files this task will read/write (relative to repo root)
        depends_on: Task IDs that must complete before this task
        persona: Which agent persona executes this task
        priority: Higher priority tasks execute first within a parallelizable batch
        estimated_complexity: Complexity hint for resource allocation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="Unique task identifier (e.g., 'task-001')")
    objective: str = Field(..., description="What this task should accomplish")
    files_touched: list[str] = Field(
        default_factory=list,
        description="Files this task will read/write (relative to repo root)",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task",
    )
    persona: Literal["engineer", "qa"] = Field(
        default="engineer",
        description="Which agent persona executes this task",
    )
    priority: int = Field(
        default=0,
        description="Higher priority tasks execute first within a parallelizable batch",
    )
    estimated_complexity: Literal["trivial", "simple", "moderate", "complex"] = Field(
        default="moderate",
        description="Complexity hint for resource allocation",
    )


class DAGGraph(BaseModel):
    """Complete dependency graph for parallel execution.

    Provides methods to:
    - Get tasks ready for execution (dependencies satisfied)
    - Detect disjoint subgraphs for parallel execution
    - Validate the graph is acyclic

    Attributes:
        tasks: List of all tasks in the graph
        metadata: Arbitrary metadata (e.g., architect reasoning)
    """

    model_config = ConfigDict(extra="forbid")

    tasks: list[DAGTask] = Field(default_factory=list)
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g., architect reasoning)",
    )

    def get_ready_tasks(self, completed: set[str]) -> list[DAGTask]:
        """Return tasks whose dependencies are all satisfied.

        Tasks are sorted by priority (descending) so higher priority
        tasks are executed first.

        Args:
            completed: Set of task IDs that have completed

        Returns:
            List of tasks ready for execution, sorted by priority
        """
        ready: list[DAGTask] = []
        for task in self.tasks:
            # Skip already completed tasks
            if task.id in completed:
                continue
            # Check if all dependencies are satisfied
            if all(dep in completed for dep in task.depends_on):
                ready.append(task)
        # Sort by priority descending (higher priority first)
        return sorted(ready, key=lambda t: -t.priority)

    def detect_disjoint_subgraphs(self) -> list[set[str]]:
        """Identify disjoint file sets that can run in parallel.

        Uses union-find algorithm to cluster tasks that share files.
        Tasks in different clusters can safely run in parallel.

        Returns:
            List of task ID sets where each set has non-overlapping files.
        """
        if not self.tasks:
            return []

        # Build file->task mapping
        file_to_tasks: dict[str, set[str]] = {}
        for task in self.tasks:
            for f in task.files_touched:
                file_to_tasks.setdefault(f, set()).add(task.id)

        # Union-find data structure
        parent: dict[str, str] = {t.id: t.id for t in self.tasks}

        def find(x: str) -> str:
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            """Union two sets."""
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        # Merge tasks that share files
        for tasks_sharing_file in file_to_tasks.values():
            task_list = list(tasks_sharing_file)
            for i in range(1, len(task_list)):
                union(task_list[0], task_list[i])

        # Group by root
        groups: dict[str, set[str]] = {}
        for task in self.tasks:
            root = find(task.id)
            groups.setdefault(root, set()).add(task.id)

        return list(groups.values())

    def validate_acyclic(self) -> bool:
        """Validate that the DAG has no cycles using Kahn's algorithm.

        Returns:
            True if the graph is acyclic, False if cycles detected
        """
        if not self.tasks:
            return True

        # Build adjacency list and in-degree counts
        task_ids = {t.id for t in self.tasks}
        in_degree: dict[str, int] = {t.id: 0 for t in self.tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in self.tasks}

        for task in self.tasks:
            for dep in task.depends_on:
                # Only count dependencies that exist in the graph
                if dep in task_ids:
                    adjacency[dep].append(task.id)
                    in_degree[task.id] += 1
                elif dep == task.id:
                    # Self-dependency is a cycle
                    return False

        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            current = queue.pop(0)
            visited += 1
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(self.tasks)


class WorktreeContext(BaseModel):
    """Context for a worktree-isolated task execution.

    Each parallel task runs in its own git worktree to prevent
    filesystem conflicts and git index.lock contention.

    Attributes:
        task_id: ID of the task being executed
        worktree_path: Filesystem path to the worktree
        branch_name: Git branch name for this worktree
        base_branch: Branch to merge back to (default: main)
        status: Current task status
        error_message: Error message if task failed
        commit_sha: Final commit SHA if task completed
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_id: str
    worktree_path: Path
    branch_name: str
    base_branch: str = Field(default="main")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    error_message: str | None = None
    commit_sha: str | None = None  # Final commit if successful


__all__ = [
    "TaskStatus",
    "DAGTask",
    "DAGGraph",
    "WorktreeContext",
]
