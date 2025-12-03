"""Integration tests for parallel swarm orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from pathlib import Path

import pytest

from jpscripts.agent import PreparedPrompt
from jpscripts.core.config import AIConfig, AppConfig, InfraConfig, UserConfig
from jpscripts.core.result import Ok
from jpscripts.core.runtime import RuntimeContext, reset_runtime_context, set_runtime_context
from jpscripts.git.client import AsyncRepo
from jpscripts.structures.dag import DAGGraph, DAGTask
from jpscripts.swarm import ParallelSwarmController


@pytest.fixture
async def async_git_repo(tmp_path: Path) -> AsyncGenerator[AsyncRepo, None]:
    """Create a temporary git repository using AsyncRepo.

    Uses subprocess for git init (no async init method available),
    then uses AsyncRepo for all subsequent operations.
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repo using subprocess (one-time operation)
    proc = await asyncio.create_subprocess_exec(
        "git", "init",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    assert proc.returncode == 0, "git init failed"

    # Open repo with AsyncRepo
    repo_result = await AsyncRepo.open(repo_path)
    assert isinstance(repo_result, Ok), f"Failed to open repo: {repo_result}"
    repo = repo_result.value

    # Configure user
    await repo.run_git("config", "user.email", "test@test.com")
    await repo.run_git("config", "user.name", "Test User")

    # Create initial commit
    (repo_path / "README.md").write_text("# Test Repo\n")
    await repo.add(all=True)
    await repo.commit("Initial commit")

    yield repo


@pytest.fixture
def swarm_config(tmp_path: Path, async_git_repo: AsyncRepo) -> AppConfig:
    """Create AppConfig for swarm testing."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(exist_ok=True)
    return AppConfig(
        ai=AIConfig(
            default_model="gpt-4o",
            max_file_context_chars=5000,
            max_command_output_chars=5000,
        ),
        infra=InfraConfig(
            worktree_root=tmp_path / "worktrees",
        ),
        user=UserConfig(
            workspace_root=async_git_repo.path,
            notes_dir=notes_dir,
            ignore_dirs=[".git"],
            use_semantic_search=False,
        ),
    )


@pytest.fixture
def runtime_ctx(
    swarm_config: AppConfig, async_git_repo: AsyncRepo
) -> Generator[RuntimeContext, None, None]:
    """Set up runtime context for the test."""
    ctx = RuntimeContext(
        config=swarm_config,
        workspace_root=async_git_repo.path,
        trace_id="test-swarm",
        dry_run=False,
    )
    token = set_runtime_context(ctx)
    yield ctx
    reset_runtime_context(token)


def create_mock_agent(
    file_patches: dict[str, str],
) -> Callable[[PreparedPrompt], Awaitable[str]]:
    """Create a mock agent that returns file-specific patches.

    Args:
        file_patches: Maps file path to the content to write
    """

    async def mock_fetch_response(prepared: PreparedPrompt) -> str:
        # Parse the task from the prompt to determine which patch to return
        prompt_text = prepared.prompt

        for file_path, content in file_patches.items():
            if file_path in prompt_text:
                return json.dumps(
                    {
                        "thought_process": f"Creating {file_path} with requested content",
                        "criticism": None,
                        "file_patch": f"""--- /dev/null
+++ b/{file_path}
@@ -0,0 +1 @@
+{content}
""",
                        "final_message": f"Created {file_path}",
                    }
                )

        # Fallback: no-op response
        return json.dumps(
            {
                "thought_process": "No specific file to modify",
                "criticism": None,
                "file_patch": None,
                "final_message": "No changes needed",
            }
        )

    return mock_fetch_response


@pytest.mark.local_only
@pytest.mark.asyncio
async def test_parallel_swarm_creates_files_and_merges(
    async_git_repo: AsyncRepo,
    swarm_config: AppConfig,
    runtime_ctx: RuntimeContext,
) -> None:
    """Verify parallel swarm executes tasks and merges results."""
    # Define expected file patches
    file_patches = {
        "file_a.txt": "A",
        "file_b.txt": "B",
    }

    # Create DAG with 2 parallel tasks (no dependencies)
    dag = DAGGraph(
        tasks=[
            DAGTask(
                id="task-a",
                objective="Create file_a.txt with content 'A'",
                files_touched=["file_a.txt"],
                depends_on=[],
                persona="engineer",
            ),
            DAGTask(
                id="task-b",
                objective="Create file_b.txt with content 'B'",
                files_touched=["file_b.txt"],
                depends_on=[],
                persona="engineer",
            ),
        ],
        metadata={"description": "Parallel file creation test"},
    )

    # Create controller with mock agent
    controller = ParallelSwarmController(
        objective="Create two files in parallel",
        config=swarm_config,
        repo_root=async_git_repo.path,
        fetch_response=create_mock_agent(file_patches),
        max_parallel=2,
    )

    controller.set_dag(dag)

    # Execute
    result = await controller.run()

    # Assertions
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    merge_result = result.value

    assert merge_result.success, "Merge should succeed"
    assert (async_git_repo.path / "file_a.txt").exists(), "file_a.txt should exist"
    assert (async_git_repo.path / "file_b.txt").exists(), "file_b.txt should exist"
    assert (async_git_repo.path / "file_a.txt").read_text().strip() == "A"
    assert (async_git_repo.path / "file_b.txt").read_text().strip() == "B"

    # Verify git log shows merge commits using AsyncRepo
    log_result = await async_git_repo.run_git("log", "--oneline", "-10")
    assert isinstance(log_result, Ok), f"Failed to get git log: {log_result}"
    # Should have merge commits from parallel branches
    assert len(log_result.value.strip().split("\n")) > 1
