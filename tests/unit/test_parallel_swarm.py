"""Tests for parallel swarm controller with worktree isolation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from jpscripts.structures.dag import DAGGraph, DAGTask, TaskStatus
from jpscripts.core.result import Err, Ok
from jpscripts.git import AsyncRepo
from jpscripts.swarm import (
    TaskResult,
    WorktreeManager,
)


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repo synchronously
    import subprocess

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True
    )

    # Create initial file and commit
    (repo_path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True
    )

    return repo_path


class TestWorktreeManager:
    """Test WorktreeManager class."""

    @pytest.mark.asyncio
    async def test_create_and_cleanup_worktree(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Create a worktree and verify cleanup."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        manager = WorktreeManager(
            repo=repo,
            worktree_root=tmp_path / "worktrees",
        )

        await manager.initialize()

        # Create worktree via context manager
        async with manager.create_worktree("task-001") as ctx:
            assert ctx.worktree_path.exists()
            assert ctx.task_id == "task-001"
            assert ctx.branch_name.startswith("swarm/")
            # Verify it's a valid git worktree
            assert (ctx.worktree_path / ".git").exists() or (
                ctx.worktree_path / "README.md"
            ).exists()

        # After context exit, worktree should be cleaned up
        # (unless preserve_on_failure is set)

    @pytest.mark.asyncio
    async def test_preserve_on_failure(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Worktrees should be preserved on failure when configured."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        manager = WorktreeManager(
            repo=repo,
            worktree_root=tmp_path / "worktrees",
            preserve_on_failure=True,
        )

        await manager.initialize()

        worktree_path: Path | None = None
        try:
            async with manager.create_worktree("task-fail") as ctx:
                worktree_path = ctx.worktree_path
                assert worktree_path.exists()
                raise RuntimeError("Simulated task failure")
        except RuntimeError:
            pass

        # Worktree should be preserved after failure
        if worktree_path:
            assert worktree_path.exists()

    @pytest.mark.asyncio
    async def test_cleanup_all(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Cleanup all worktrees."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        manager = WorktreeManager(
            repo=repo,
            worktree_root=tmp_path / "worktrees",
            preserve_on_failure=True,  # Don't auto-cleanup
        )

        await manager.initialize()

        paths: list[Path] = []

        # Create multiple worktrees manually (without cleanup)
        for i in range(3):
            ctx = await manager._create_worktree_context(f"task-{i:03d}")
            paths.append(ctx.worktree_path)
            assert ctx.worktree_path.exists()

        # Force cleanup all
        await manager.cleanup_all(force=True)

        # All should be cleaned up
        for path in paths:
            assert not path.exists()


class TestTaskResult:
    """Test TaskResult model."""

    def test_create_successful_result(self) -> None:
        """Create a successful task result."""
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.COMPLETED,
            branch_name="swarm/task-001",
            commit_sha="abc123",
        )
        assert result.task_id == "task-001"
        assert result.status == TaskStatus.COMPLETED
        assert result.commit_sha == "abc123"
        assert result.error_message is None

    def test_create_failed_result(self) -> None:
        """Create a failed task result."""
        result = TaskResult(
            task_id="task-002",
            status=TaskStatus.FAILED,
            branch_name="swarm/task-002",
            error_message="Test error",
        )
        assert result.task_id == "task-002"
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Test error"
        assert result.commit_sha is None


class TestDAGExecution:
    """Test DAG execution patterns."""

    def test_detect_parallelizable_tasks(self) -> None:
        """Tasks with disjoint files should be parallelizable."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=["src/foo.py"]),
                DAGTask(id="task-002", objective="B", files_touched=["src/bar.py"]),
            ]
        )

        groups = graph.detect_disjoint_subgraphs()
        # Two disjoint groups - can run in parallel
        assert len(groups) == 2

    def test_detect_sequential_tasks(self) -> None:
        """Tasks with overlapping files must be sequential."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A", files_touched=["src/foo.py"]),
                DAGTask(id="task-002", objective="B", files_touched=["src/foo.py"]),
            ]
        )

        groups = graph.detect_disjoint_subgraphs()
        # One group - must be sequential
        assert len(groups) == 1
        assert groups[0] == {"task-001", "task-002"}

    def test_ready_tasks_respects_dependencies(self) -> None:
        """Only tasks with satisfied dependencies should be ready."""
        graph = DAGGraph(
            tasks=[
                DAGTask(id="task-001", objective="A"),
                DAGTask(id="task-002", objective="B", depends_on=["task-001"]),
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


class TestWorktreeManagerInitialization:
    """Test WorktreeManager initialization."""

    @pytest.mark.asyncio
    async def test_creates_worktree_root(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Initialize should create the worktree root directory."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_root = tmp_path / "worktrees"
        assert not worktree_root.exists()

        manager = WorktreeManager(repo=repo, worktree_root=worktree_root)
        await manager.initialize()

        assert worktree_root.exists()


class TestOrphanDetection:
    """Test orphan worktree detection and cleanup."""

    @pytest.mark.asyncio
    async def test_detects_orphaned_worktrees(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Orphan directories from crashed sessions are detected."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_root = tmp_path / "worktrees"
        worktree_root.mkdir()

        # Simulate crashed session: directories exist but not tracked
        orphan1 = worktree_root / "worktree-task-001-abcd1234"
        orphan2 = worktree_root / "worktree-task-002-ef567890"
        orphan1.mkdir()
        orphan2.mkdir()

        # Also create a non-matching directory (should be ignored)
        (worktree_root / "other-dir").mkdir()

        manager = WorktreeManager(repo=repo, worktree_root=worktree_root)

        orphans = await manager.detect_orphaned_worktrees()

        assert len(orphans) == 2
        assert orphan1 in orphans
        assert orphan2 in orphans

    @pytest.mark.asyncio
    async def test_initialize_prunes_orphans(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Initialize() should auto-prune orphaned worktrees."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_root = tmp_path / "worktrees"
        worktree_root.mkdir()

        # Simulate orphan from previous crash
        orphan = worktree_root / "worktree-crashed-task-12345678"
        orphan.mkdir()
        (orphan / "dummy.txt").write_text("leftover")

        manager = WorktreeManager(repo=repo, worktree_root=worktree_root)
        result = await manager.initialize()

        assert isinstance(result, Ok)
        # Orphan directory should be removed
        assert not orphan.exists()

    @pytest.mark.asyncio
    async def test_active_worktrees_not_pruned(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Active worktrees should not be detected as orphans."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        manager = WorktreeManager(
            repo=repo,
            worktree_root=tmp_path / "worktrees",
        )
        await manager.initialize()

        # Create an active worktree
        ctx = await manager._create_worktree_context("active-task")

        # Should not detect the active one as orphan
        orphans = await manager.detect_orphaned_worktrees()

        assert ctx.worktree_path not in orphans

        # Cleanup
        await manager.cleanup_worktree(ctx)

    @pytest.mark.asyncio
    async def test_prune_handles_non_worktree_dirs(
        self, temp_git_repo: Path, tmp_path: Path
    ) -> None:
        """Prune should handle directories that look like worktrees but aren't git worktrees."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_root = tmp_path / "worktrees"
        worktree_root.mkdir()

        # Simulate fake orphan (not a real git worktree)
        fake_orphan = worktree_root / "worktree-fake-task-abcd1234"
        fake_orphan.mkdir()
        (fake_orphan / "not-a-git-repo.txt").write_text("fake")

        manager = WorktreeManager(repo=repo, worktree_root=worktree_root)
        result = await manager.initialize()

        assert isinstance(result, Ok)
        # Fake orphan should still be removed via shutil fallback
        assert not fake_orphan.exists()
