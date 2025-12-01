"""Integration tests for parallel swarm orchestration."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from jpscripts.core.config import AIConfig, AppConfig, InfraConfig, UserConfig
from jpscripts.core.dag import DAGGraph, DAGTask
from jpscripts.core.result import Ok
from jpscripts.core.runtime import RuntimeContext, reset_runtime_context, set_runtime_context
from jpscripts.engine import PreparedPrompt
from jpscripts.swarm import ParallelSwarmController


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit
    (repo_path / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


@pytest.fixture
def swarm_config(tmp_path: Path, temp_git_repo: Path) -> AppConfig:
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
            workspace_root=temp_git_repo,
            notes_dir=notes_dir,
            ignore_dirs=[".git"],
            use_semantic_search=False,
        ),
    )


@pytest.fixture
def runtime_ctx(swarm_config: AppConfig) -> Generator[RuntimeContext, None, None]:
    """Set up runtime context for the test."""
    ctx = RuntimeContext(
        config=swarm_config,
        workspace_root=swarm_config.user.workspace_root,
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
    temp_git_repo: Path,
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
        repo_root=temp_git_repo,
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
    assert (temp_git_repo / "file_a.txt").exists(), "file_a.txt should exist"
    assert (temp_git_repo / "file_b.txt").exists(), "file_b.txt should exist"
    assert (temp_git_repo / "file_a.txt").read_text().strip() == "A"
    assert (temp_git_repo / "file_b.txt").read_text().strip() == "B"

    # Verify git log shows merge commits
    log_result = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
    )
    assert log_result.returncode == 0
    # Should have merge commits from parallel branches
    assert len(log_result.stdout.strip().split("\n")) > 1
