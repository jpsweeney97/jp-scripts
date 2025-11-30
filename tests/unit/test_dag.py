"""Tests for DAG-based task orchestration models."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jpscripts.core.dag import (
    DAGGraph,
    DAGTask,
    TaskStatus,
    WorktreeContext,
)


class TestDAGTask:
    """Test DAGTask model."""

    def test_create_minimal_task(self) -> None:
        """Create a task with minimal required fields."""
        task = DAGTask(
            id="task-001",
            objective="Implement feature X",
        )
        assert task.id == "task-001"
        assert task.objective == "Implement feature X"
        assert task.files_touched == []
        assert task.depends_on == []
        assert task.persona == "engineer"

    def test_create_full_task(self) -> None:
        """Create a task with all fields specified."""
        task = DAGTask(
            id="task-002",
            objective="Add tests for feature X",
            files_touched=["src/foo.py", "tests/test_foo.py"],
            depends_on=["task-001"],
            persona="qa",
            priority=10,
            estimated_complexity="complex",
        )
        assert task.id == "task-002"
        assert task.files_touched == ["src/foo.py", "tests/test_foo.py"]
        assert task.depends_on == ["task-001"]
        assert task.persona == "qa"
        assert task.priority == 10
        assert task.estimated_complexity == "complex"


class TestDAGGraph:
    """Test DAGGraph model and methods."""

    def test_empty_graph(self) -> None:
        """Empty graph should be valid."""
        graph = DAGGraph(tasks=[])
        assert graph.tasks == []
        assert graph.validate_acyclic() is True

    def test_get_ready_tasks_no_dependencies(self) -> None:
        """Tasks with no dependencies should all be ready."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A"),
                DAGTask(id="task-002", objective="B"),
                DAGTask(id="task-003", objective="C"),
            ]
        )
        ready = graph.get_ready_tasks(completed=set())
        assert len(ready) == 3
        assert {t.id for t in ready} == {"task-001", "task-002", "task-003"}

    def test_get_ready_tasks_with_dependencies(self) -> None:
        """Only tasks with satisfied dependencies should be ready."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A"),
                DAGTask(id="task-002", objective="B", depends_on=["task-001"]),
                DAGTask(id="task-003", objective="C", depends_on=["task-002"]),
            ]
        )

        # Initially only task-001 is ready
        ready = graph.get_ready_tasks(completed=set())
        assert len(ready) == 1
        assert ready[0].id == "task-001"

        # After task-001 completes, task-002 is ready
        ready = graph.get_ready_tasks(completed={"task-001"})
        assert len(ready) == 1
        assert ready[0].id == "task-002"

        # After both complete, task-003 is ready
        ready = graph.get_ready_tasks(completed={"task-001", "task-002"})
        assert len(ready) == 1
        assert ready[0].id == "task-003"

    def test_get_ready_tasks_respects_priority(self) -> None:
        """Ready tasks should be sorted by priority (descending)."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="Low", priority=1),
                DAGTask(id="task-002", objective="High", priority=10),
                DAGTask(id="task-003", objective="Medium", priority=5),
            ]
        )
        ready = graph.get_ready_tasks(completed=set())
        assert [t.id for t in ready] == ["task-002", "task-003", "task-001"]

    def test_validate_acyclic_valid(self) -> None:
        """Valid DAG should pass validation."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A"),
                DAGTask(id="task-002", objective="B", depends_on=["task-001"]),
                DAGTask(id="task-003", objective="C", depends_on=["task-001"]),
                DAGTask(id="task-004", objective="D", depends_on=["task-002", "task-003"]),
            ]
        )
        assert graph.validate_acyclic() is True

    def test_validate_acyclic_detects_cycle(self) -> None:
        """DAG with cycle should fail validation."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", depends_on=["task-003"]),
                DAGTask(id="task-002", objective="B", depends_on=["task-001"]),
                DAGTask(id="task-003", objective="C", depends_on=["task-002"]),
            ]
        )
        assert graph.validate_acyclic() is False

    def test_validate_acyclic_self_cycle(self) -> None:
        """Task depending on itself should fail validation."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", depends_on=["task-001"]),
            ]
        )
        assert graph.validate_acyclic() is False

    def test_detect_disjoint_subgraphs_single_group(self) -> None:
        """Tasks with overlapping files should be in the same group."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=["src/foo.py"]),
                DAGTask(id="task-002", objective="B", files_touched=["src/foo.py", "src/bar.py"]),
                DAGTask(id="task-003", objective="C", files_touched=["src/bar.py"]),
            ]
        )
        groups = graph.detect_disjoint_subgraphs()
        assert len(groups) == 1
        assert groups[0] == {"task-001", "task-002", "task-003"}

    def test_detect_disjoint_subgraphs_multiple_groups(self) -> None:
        """Tasks with non-overlapping files should be in separate groups."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=["src/foo.py"]),
                DAGTask(id="task-002", objective="B", files_touched=["src/bar.py"]),
                DAGTask(id="task-003", objective="C", files_touched=["src/baz.py"]),
            ]
        )
        groups = graph.detect_disjoint_subgraphs()
        assert len(groups) == 3
        # Each task in its own group
        group_sets = list(groups)
        assert {"task-001"} in group_sets
        assert {"task-002"} in group_sets
        assert {"task-003"} in group_sets

    def test_detect_disjoint_subgraphs_empty_files(self) -> None:
        """Tasks with no files_touched should be isolated."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=[]),
                DAGTask(id="task-002", objective="B", files_touched=[]),
            ]
        )
        groups = graph.detect_disjoint_subgraphs()
        assert len(groups) == 2

    def test_detect_disjoint_mixed(self) -> None:
        """Mixed scenario with some overlapping and some disjoint."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=["src/foo.py"]),
                DAGTask(id="task-002", objective="B", files_touched=["src/foo.py"]),
                DAGTask(id="task-003", objective="C", files_touched=["src/bar.py"]),
                DAGTask(id="task-004", objective="D", files_touched=["src/bar.py"]),
            ]
        )
        groups = graph.detect_disjoint_subgraphs()
        assert len(groups) == 2
        # task-001 and task-002 should be together
        # task-003 and task-004 should be together
        group_ids = [frozenset(g) for g in groups]
        assert frozenset({"task-001", "task-002"}) in group_ids
        assert frozenset({"task-003", "task-004"}) in group_ids


class TestWorktreeContext:
    """Test WorktreeContext model."""

    def test_create_worktree_context(self) -> None:
        """Create a worktree context with all fields."""
        ctx = WorktreeContext(
            task_id="task-001",
            worktree_path=Path("/tmp/worktree-001"),
            branch_name="swarm/task-001",
        )
        assert ctx.task_id == "task-001"
        assert ctx.worktree_path == Path("/tmp/worktree-001")
        assert ctx.branch_name == "swarm/task-001"
        assert ctx.base_branch == "main"
        assert ctx.status == TaskStatus.PENDING
        assert ctx.error_message is None
        assert ctx.commit_sha is None

    def test_worktree_context_status_transitions(self) -> None:
        """Worktree context status can be updated."""
        ctx = WorktreeContext(
            task_id="task-001",
            worktree_path=Path("/tmp/worktree-001"),
            branch_name="swarm/task-001",
            status=TaskStatus.RUNNING,
        )
        assert ctx.status == TaskStatus.RUNNING

        # Create new context with completed status
        completed_ctx = ctx.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "commit_sha": "abc123",
            }
        )
        assert completed_ctx.status == TaskStatus.COMPLETED
        assert completed_ctx.commit_sha == "abc123"


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_all_status_values(self) -> None:
        """Verify all expected status values exist."""
        assert TaskStatus.PENDING
        assert TaskStatus.RUNNING
        assert TaskStatus.COMPLETED
        assert TaskStatus.FAILED
        assert TaskStatus.BLOCKED
