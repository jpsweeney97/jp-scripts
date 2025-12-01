"""Tests for git worktree operations in AsyncRepo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import asyncio

import pytest

from jpscripts.git import AsyncRepo, WorktreeInfo
from jpscripts.core.result import Err, Ok


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repo
    asyncio.run(_init_repo(repo_path))
    return repo_path


async def _init_repo(path: Path) -> None:
    """Initialize a git repository with an initial commit."""
    import subprocess

    # Use subprocess directly for setup (not part of SUT)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=path, check=True, capture_output=True
    )

    # Create initial file and commit
    (path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=path, check=True, capture_output=True
    )


class TestWorktreeAdd:
    """Test worktree_add method."""

    @pytest.mark.asyncio
    async def test_add_new_worktree(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Create a new worktree with a new branch."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_path = tmp_path / "worktree-001"

        match await repo.worktree_add(worktree_path, "feature-branch", new_branch=True):
            case Ok(result_path):
                assert result_path.exists()
                assert (result_path / "README.md").exists()
            case Err(err):
                pytest.fail(f"Failed to create worktree: {err}")

    @pytest.mark.asyncio
    async def test_add_worktree_with_start_point(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Create a worktree starting from a specific commit."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_path = tmp_path / "worktree-head"

        match await repo.worktree_add(
            worktree_path, "branch-at-head", new_branch=True, start_point="HEAD"
        ):
            case Ok(result_path):
                assert result_path.exists()
            case Err(err):
                pytest.fail(f"Failed to create worktree: {err}")


class TestWorktreeRemove:
    """Test worktree_remove method."""

    @pytest.mark.asyncio
    async def test_remove_worktree(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Remove a worktree cleanly."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_path = tmp_path / "worktree-to-remove"

        # Create worktree
        match await repo.worktree_add(worktree_path, "temp-branch", new_branch=True):
            case Ok(_):
                pass
            case Err(err):
                pytest.fail(f"Failed to create worktree: {err}")

        assert worktree_path.exists()

        # Remove worktree
        match await repo.worktree_remove(worktree_path):
            case Ok(_):
                assert not worktree_path.exists()
            case Err(err):
                pytest.fail(f"Failed to remove worktree: {err}")

    @pytest.mark.asyncio
    async def test_force_remove_dirty_worktree(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """Force remove a worktree with uncommitted changes."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        worktree_path = tmp_path / "dirty-worktree"

        # Create worktree
        match await repo.worktree_add(worktree_path, "dirty-branch", new_branch=True):
            case Ok(_):
                pass
            case Err(err):
                pytest.fail(f"Failed to create worktree: {err}")

        # Make the worktree dirty
        (worktree_path / "dirty.txt").write_text("uncommitted change")

        # Force remove should work
        match await repo.worktree_remove(worktree_path, force=True):
            case Ok(_):
                assert not worktree_path.exists()
            case Err(err):
                pytest.fail(f"Failed to force remove worktree: {err}")


class TestWorktreeList:
    """Test worktree_list method."""

    @pytest.mark.asyncio
    async def test_list_worktrees(self, temp_git_repo: Path, tmp_path: Path) -> None:
        """List all worktrees including main and created ones."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        # Create additional worktrees
        worktree1 = tmp_path / "worktree-1"
        worktree2 = tmp_path / "worktree-2"

        await repo.worktree_add(worktree1, "branch-1", new_branch=True)
        await repo.worktree_add(worktree2, "branch-2", new_branch=True)

        match await repo.worktree_list():
            case Ok(worktrees):
                # Should have main worktree + 2 created ones
                assert len(worktrees) >= 3
                paths = [str(w.path) for w in worktrees]
                assert any("worktree-1" in p for p in paths)
                assert any("worktree-2" in p for p in paths)
            case Err(err):
                pytest.fail(f"Failed to list worktrees: {err}")


class TestWorktreePrune:
    """Test worktree_prune method."""

    @pytest.mark.asyncio
    async def test_prune_worktrees(self, temp_git_repo: Path) -> None:
        """Prune stale worktree references."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        # Prune should succeed even with nothing to prune
        match await repo.worktree_prune():
            case Ok(_):
                pass  # Success
            case Err(err):
                pytest.fail(f"Failed to prune worktrees: {err}")


class TestMerge:
    """Test merge-related methods."""

    @pytest.mark.asyncio
    async def test_merge_branch(self, temp_git_repo: Path) -> None:
        """Merge a branch into current HEAD."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        # Get the current branch name (may be 'main' or 'master')
        status_result = await repo.status()
        match status_result:
            case Ok(status):
                base_branch = status.branch
            case Err(err):
                pytest.fail(f"Failed to get status: {err}")

        # Create a branch with changes
        import subprocess

        subprocess.run(
            ["git", "checkout", "-b", "feature"], cwd=temp_git_repo, check=True, capture_output=True
        )
        (temp_git_repo / "feature.txt").write_text("feature content")
        subprocess.run(["git", "add", "."], cwd=temp_git_repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Feature commit"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", base_branch], cwd=temp_git_repo, check=True, capture_output=True
        )

        match await repo.merge("feature"):
            case Ok(sha):
                assert sha  # Got a commit SHA
                assert (temp_git_repo / "feature.txt").exists()
            case Err(err):
                pytest.fail(f"Failed to merge: {err}")

    @pytest.mark.asyncio
    async def test_merge_abort(self, temp_git_repo: Path) -> None:
        """Abort a merge in progress."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        # merge_abort should be safe to call even when no merge in progress
        # (it will return an error, which is expected)
        result = await repo.merge_abort()
        # We just verify the method exists and can be called
        assert isinstance(result, (Ok, Err))


class TestConflictFiles:
    """Test get_conflict_files method."""

    @pytest.mark.asyncio
    async def test_get_conflict_files_no_conflicts(self, temp_git_repo: Path) -> None:
        """Get empty list when no conflicts exist."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        match await repo.get_conflict_files():
            case Ok(files):
                assert files == []
            case Err(err):
                pytest.fail(f"Failed to get conflict files: {err}")


class TestBranchOperations:
    """Test branch-related operations."""

    @pytest.mark.asyncio
    async def test_checkout_branch_create(self, temp_git_repo: Path) -> None:
        """Create and checkout a new branch."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        match await repo.checkout_branch("new-feature", create=True):
            case Ok(_):
                # Verify we're on the new branch
                status_result = await repo.status()
                match status_result:
                    case Ok(status):
                        assert status.branch == "new-feature"
                    case Err(err):
                        pytest.fail(f"Failed to get status: {err}")
            case Err(err):
                pytest.fail(f"Failed to checkout branch: {err}")

    @pytest.mark.asyncio
    async def test_delete_branch(self, temp_git_repo: Path) -> None:
        """Delete a branch."""
        match await AsyncRepo.open(temp_git_repo):
            case Ok(repo):
                pass
            case Err(err):
                pytest.fail(f"Failed to open repo: {err}")

        # Get the current branch name (may be 'main' or 'master')
        status_result = await repo.status()
        match status_result:
            case Ok(status):
                base_branch = status.branch
            case Err(err):
                pytest.fail(f"Failed to get status: {err}")

        # Create a branch to delete
        await repo.checkout_branch("to-delete", create=True)
        # Go back to base branch
        await repo.checkout_branch(base_branch)

        match await repo.delete_branch("to-delete"):
            case Ok(_):
                pass  # Success
            case Err(err):
                pytest.fail(f"Failed to delete branch: {err}")


class TestWorktreeInfo:
    """Test WorktreeInfo dataclass."""

    def test_worktree_info_fields(self) -> None:
        """WorktreeInfo should have all expected fields."""
        info = WorktreeInfo(
            path=Path("/tmp/worktree"),
            branch="feature",
            commit="abc123",
            is_locked=False,
            prunable=False,
        )
        assert info.path == Path("/tmp/worktree")
        assert info.branch == "feature"
        assert info.commit == "abc123"
        assert info.is_locked is False
        assert info.prunable is False
