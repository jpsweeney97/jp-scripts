"""Data types for parallel swarm execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from jpscripts.structures.dag import TaskStatus


@dataclass
class TaskResult:
    """Result of executing a single DAG task.

    Attributes:
        task_id: ID of the completed task
        status: Final task status
        branch_name: Git branch created for this task
        commit_sha: Final commit SHA if successful
        error_message: Error details if failed
        artifacts: Any artifacts produced by the task
    """

    task_id: str
    status: TaskStatus
    branch_name: str
    commit_sha: str | None = None
    error_message: str | None = None
    artifacts: list[str] = field(default_factory=list)


class MergeResult(BaseModel):
    """Result of merging parallel branches.

    Attributes:
        success: Whether all merges succeeded
        merged_branches: List of successfully merged branches
        conflict_branches: Branches with conflicts requiring resolution
        final_commit: Final merge commit SHA if successful
    """

    model_config = ConfigDict(extra="forbid")

    success: bool
    merged_branches: list[str] = Field(default_factory=list)
    conflict_branches: list[str] = Field(default_factory=list)
    final_commit: str | None = None


__all__ = [
    "MergeResult",
    "TaskResult",
]
