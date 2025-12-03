"""Unit tests for ParallelSwarmController."""

from __future__ import annotations

from pathlib import Path

import pytest

from jpscripts.core.config import AIConfig, AppConfig, InfraConfig, UserConfig
from jpscripts.core.result import Err
from jpscripts.structures.dag import DAGGraph, DAGTask
from jpscripts.swarm.controller import ParallelSwarmController


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Create minimal AppConfig for controller tests."""
    return AppConfig(
        ai=AIConfig(default_model="gpt-4o"),
        infra=InfraConfig(worktree_root=tmp_path / "worktrees"),
        user=UserConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path / "notes",
        ),
    )


class TestParallelSwarmControllerInit:
    """Tests for controller initialization."""

    def test_raises_value_error_without_executor_or_fetch_response(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """Controller requires either task_executor or fetch_response."""
        with pytest.raises(ValueError, match="requires either task_executor or fetch_response"):
            ParallelSwarmController(
                objective="Test objective",
                config=test_config,
                repo_root=tmp_path,
                # Neither task_executor nor fetch_response provided
            )


class TestParallelSwarmControllerRun:
    """Tests for controller run method."""

    @pytest.mark.asyncio
    async def test_run_returns_error_when_no_dag_set(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """run() returns WorkspaceError when no DAG has been set."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )
        # Don't call set_dag()

        result = await controller.run()

        assert isinstance(result, Err)
        assert "No DAG set" in str(result.error)


class TestParallelSwarmControllerSetDag:
    """Tests for DAG validation."""

    def test_set_dag_rejects_cyclic_graph(self, test_config: AppConfig, tmp_path: Path) -> None:
        """set_dag() returns ValidationError for cyclic DAGs."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # Create DAG with cycle: A -> B -> A
        cyclic_dag = DAGGraph(
            tasks=[
                DAGTask(
                    id="task-a",
                    objective="Task A",
                    files_touched=["a.txt"],
                    depends_on=["task-b"],
                    persona="engineer",
                ),
                DAGTask(
                    id="task-b",
                    objective="Task B",
                    files_touched=["b.txt"],
                    depends_on=["task-a"],
                    persona="engineer",
                ),
            ],
            metadata={},
        )

        result = controller.set_dag(cyclic_dag)

        assert isinstance(result, Err)
        assert "cycles" in str(result.error).lower()


class TestSwarmStateRecovery:
    """Tests for state persistence and recovery."""

    def test_recover_previous_session_returns_none_when_no_state(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """recover_previous_session() returns None when no state file exists."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        result = controller.recover_previous_session()

        assert result is None

    def test_load_state_returns_none_for_invalid_json(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """_load_state() returns None when state file contains invalid JSON."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # Create invalid state file
        state_path = tmp_path / ".jpscripts" / "swarm_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("not valid json {{{")

        result = controller._load_state()

        assert result is None

    def test_load_state_returns_valid_state(self, test_config: AppConfig, tmp_path: Path) -> None:
        """_load_state() returns SwarmState when valid state file exists."""
        from jpscripts.swarm.controller import SwarmState

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # Create valid state file
        valid_state = SwarmState(
            swarm_id="abc123",
            tasks={"task-1": "COMPLETED"},
            worktree_paths={"task-1": "swarm/task-1"},
            started_at="2025-01-01T00:00:00+00:00",
            objective="Test",
        )
        state_path = tmp_path / ".jpscripts" / "swarm_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(valid_state.model_dump_json())

        result = controller._load_state()

        assert result is not None
        assert result.swarm_id == "abc123"
        assert result.tasks == {"task-1": "COMPLETED"}

    def test_recover_previous_session_returns_state_when_exists(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """recover_previous_session() returns SwarmState when state file exists."""
        from jpscripts.swarm.controller import SwarmState

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # Create valid state file
        valid_state = SwarmState(
            swarm_id="xyz789",
            tasks={},
            worktree_paths={},
            started_at="2025-01-01T00:00:00+00:00",
            objective="Previous session",
        )
        state_path = tmp_path / ".jpscripts" / "swarm_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(valid_state.model_dump_json())

        result = controller.recover_previous_session()

        assert result is not None
        assert result.swarm_id == "xyz789"

    def test_clear_state_removes_file(self, test_config: AppConfig, tmp_path: Path) -> None:
        """_clear_state() removes the state file when it exists."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # Create state file
        state_path = tmp_path / ".jpscripts" / "swarm_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}")

        assert state_path.exists()
        controller._clear_state()
        assert not state_path.exists()

    def test_clear_state_noop_when_no_file(self, test_config: AppConfig, tmp_path: Path) -> None:
        """_clear_state() does nothing when no state file exists."""

        async def mock_fetch(_: object) -> str:
            return "{}"

        controller = ParallelSwarmController(
            objective="Test objective",
            config=test_config,
            repo_root=tmp_path,
            fetch_response=mock_fetch,
        )

        # No state file exists
        state_path = tmp_path / ".jpscripts" / "swarm_state.json"
        assert not state_path.exists()

        # Should not raise
        controller._clear_state()
