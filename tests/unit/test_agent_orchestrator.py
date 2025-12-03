"""State-based unit tests for RepairLoopOrchestrator.

Tests verify orchestrator logic by inspecting yielded AgentEvent objects,
without touching real shell commands or LLM providers.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

from jpscripts.agent import PreparedPrompt, execution, ops
from jpscripts.agent.execution import (
    AgentEvent,
    EventKind,
    RepairLoopConfig,
    RepairLoopOrchestrator,
)
from jpscripts.core.config import AppConfig
from jpscripts.core.runtime import RuntimeContext, runtime_context

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Create minimal AppConfig for testing."""
    return AppConfig(
        workspace_root=tmp_path,
        notes_dir=tmp_path / "notes",
        use_semantic_search=False,
    )


@pytest.fixture
def runtime_ctx(app_config: AppConfig, tmp_path: Path) -> Generator[RuntimeContext, None, None]:
    """Set up runtime context for tests that need it."""
    with runtime_context(app_config, workspace=tmp_path) as ctx:
        yield ctx


async def collect_events(orchestrator: RepairLoopOrchestrator) -> list[AgentEvent]:
    """Collect all events from orchestrator run."""
    events: list[AgentEvent] = []
    async for event in orchestrator.run():
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Test 1: Success Path (command passes immediately)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_immediate(
    tmp_path: Path,
    app_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestrator exits immediately when command succeeds."""

    async def mock_run_command(command: str, root: Path) -> tuple[int, str, str]:
        return (0, "ok", "")

    monkeypatch.setattr(ops, "run_agent_command", mock_run_command)

    # Disable archiving to avoid save_memory calls
    async def mock_fetch(prepared: PreparedPrompt) -> str:
        return json.dumps({"thought_process": "ok", "final_message": "done"})

    orchestrator = RepairLoopOrchestrator(
        base_prompt="Fix the bug",
        command="echo test",
        model="gpt-4o",
        fetch_response=mock_fetch,
        config=RepairLoopConfig(max_retries=3, auto_archive=False),
        app_config=app_config,
        workspace_root=tmp_path,
    )

    events = await collect_events(orchestrator)

    # Assert event sequence
    assert events[0].kind == EventKind.ATTEMPT_START
    assert events[0].data["attempt"] == 1

    assert events[1].kind == EventKind.COMMAND_SUCCESS
    assert events[1].data["phase"] == "initial"

    assert events[-1].kind == EventKind.COMPLETE
    assert events[-1].data["success"] is True

    # No repair events should have occurred
    event_kinds = {e.kind for e in events}
    assert EventKind.PATCH_PROPOSED not in event_kinds
    assert EventKind.TOOL_CALL not in event_kinds
    assert EventKind.COMMAND_FAILED not in event_kinds


# ---------------------------------------------------------------------------
# Test 2: Repair Loop (fail -> patch -> success)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_loop_success(
    tmp_path: Path,
    app_config: AppConfig,
    runtime_ctx: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestrator applies patch and succeeds after initial failure."""
    call_count = 0

    async def mock_run_command(command: str, root: Path) -> tuple[int, str, str]:
        nonlocal call_count
        call_count += 1
        # First call fails, subsequent calls succeed
        if call_count == 1:
            return (1, "", "Error: test failure")
        return (0, "ok", "")

    captured_prompts: list[str] = []

    async def mock_fetch(prepared: PreparedPrompt) -> str:
        captured_prompts.append(prepared.prompt)
        return json.dumps(
            {
                "thought_process": "Fixing the error by updating test.py",
                "criticism": None,
                "tool_call": None,
                "file_patch": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
                "final_message": None,
            }
        )

    async def mock_apply_patch(patch_text: str, root: Path) -> list[Path]:
        return [tmp_path / "test.py"]

    async def mock_verify_syntax(files: list[Path]) -> str | None:
        return None  # No syntax errors

    monkeypatch.setattr(ops, "run_agent_command", mock_run_command)
    monkeypatch.setattr(execution, "apply_patch_text", mock_apply_patch)
    monkeypatch.setattr(ops, "verify_syntax", mock_verify_syntax)

    orchestrator = RepairLoopOrchestrator(
        base_prompt="Fix the bug",
        command="pytest",
        model="gpt-4o",
        fetch_response=mock_fetch,
        config=RepairLoopConfig(max_retries=3, auto_archive=False),
        app_config=app_config,
        workspace_root=tmp_path,
    )

    events = await collect_events(orchestrator)

    # Assert repair events occurred
    event_kinds = [e.kind for e in events]

    assert EventKind.ATTEMPT_START in event_kinds
    assert EventKind.COMMAND_FAILED in event_kinds
    assert EventKind.PATCH_PROPOSED in event_kinds
    assert EventKind.PATCH_APPLIED in event_kinds
    assert EventKind.COMMAND_SUCCESS in event_kinds

    # Final event should be success
    assert events[-1].kind == EventKind.COMPLETE
    assert events[-1].data["success"] is True

    # Verify prompt was captured and contains error
    assert captured_prompts
    assert "test failure" in captured_prompts[0].lower() or "fix" in captured_prompts[0].lower()


# ---------------------------------------------------------------------------
# Test 3: Max Retries Exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_retries_exhausted(
    tmp_path: Path,
    app_config: AppConfig,
    runtime_ctx: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestrator fails after exhausting all retries."""

    async def mock_run_command(command: str, root: Path) -> tuple[int, str, str]:
        return (1, "", "persistent failure")

    async def mock_fetch(prepared: PreparedPrompt) -> str:
        return json.dumps(
            {
                "thought_process": "Cannot fix this issue",
                "criticism": None,
                "tool_call": None,
                "file_patch": None,
                "final_message": "Unable to resolve the error",
            }
        )

    monkeypatch.setattr(ops, "run_agent_command", mock_run_command)

    orchestrator = RepairLoopOrchestrator(
        base_prompt="Fix the bug",
        command="pytest",
        model="gpt-4o",
        fetch_response=mock_fetch,
        config=RepairLoopConfig(max_retries=2, keep_failed=True, auto_archive=False),
        app_config=app_config,
        workspace_root=tmp_path,
    )

    events = await collect_events(orchestrator)

    # Count attempt starts
    attempt_starts = [e for e in events if e.kind == EventKind.ATTEMPT_START]
    assert len(attempt_starts) == 2

    # Assert failure
    assert events[-1].kind == EventKind.COMPLETE
    assert events[-1].data["success"] is False

    # Verify NO_PATCH events occurred (agent returned no patch)
    event_kinds = [e.kind for e in events]
    assert EventKind.NO_PATCH in event_kinds


# ---------------------------------------------------------------------------
# Test 4: Duplicate Patch Detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_patch_detection(
    tmp_path: Path,
    app_config: AppConfig,
    runtime_ctx: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestrator detects and handles duplicate patches."""
    call_count = 0

    async def mock_run_command(command: str, root: Path) -> tuple[int, str, str]:
        nonlocal call_count
        call_count += 1
        # Always fail to force multiple patch attempts
        return (1, "", f"failure {call_count}")

    # Return the same patch content every time
    same_patch = "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new"

    async def mock_fetch(prepared: PreparedPrompt) -> str:
        return json.dumps(
            {
                "thought_process": "Trying to fix",
                "criticism": None,
                "tool_call": None,
                "file_patch": same_patch,
                "final_message": None,
            }
        )

    async def mock_apply_patch(patch_text: str, root: Path) -> list[Path]:
        return [tmp_path / "test.py"]

    async def mock_verify_syntax(files: list[Path]) -> str | None:
        return None

    monkeypatch.setattr(ops, "run_agent_command", mock_run_command)
    monkeypatch.setattr(execution, "apply_patch_text", mock_apply_patch)
    monkeypatch.setattr(ops, "verify_syntax", mock_verify_syntax)

    orchestrator = RepairLoopOrchestrator(
        base_prompt="Fix the bug",
        command="pytest",
        model="gpt-4o",
        fetch_response=mock_fetch,
        config=RepairLoopConfig(max_retries=2, keep_failed=True, auto_archive=False),
        app_config=app_config,
        workspace_root=tmp_path,
    )

    events = await collect_events(orchestrator)
    event_kinds = [e.kind for e in events]

    # Should have at least one DUPLICATE_PATCH event
    assert EventKind.DUPLICATE_PATCH in event_kinds


# ---------------------------------------------------------------------------
# Test 5: Syntax Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_syntax_error_handling(
    tmp_path: Path,
    app_config: AppConfig,
    runtime_ctx: RuntimeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify orchestrator handles syntax errors in patches."""
    call_count = 0

    async def mock_run_command(command: str, root: Path) -> tuple[int, str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (1, "", "initial failure")
        return (0, "ok", "")

    patch_count = 0

    async def mock_fetch(prepared: PreparedPrompt) -> str:
        nonlocal patch_count
        patch_count += 1
        # First patch has syntax error, second is valid
        if patch_count == 1:
            return json.dumps(
                {
                    "thought_process": "First attempt",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+invalid syntax (",
                    "final_message": None,
                }
            )
        return json.dumps(
            {
                "thought_process": "Fixed syntax",
                "criticism": None,
                "tool_call": None,
                "file_patch": "--- a/test.py\n+++ b/test.py\n@@ -1 +1 @@\n-old\n+new",
                "final_message": None,
            }
        )

    async def mock_apply_patch(patch_text: str, root: Path) -> list[Path]:
        return [tmp_path / "test.py"]

    syntax_check_count = 0

    async def mock_verify_syntax(files: list[Path]) -> str | None:
        nonlocal syntax_check_count
        syntax_check_count += 1
        if syntax_check_count == 1:
            return "SyntaxError: unexpected EOF"
        return None

    monkeypatch.setattr(ops, "run_agent_command", mock_run_command)
    monkeypatch.setattr(execution, "apply_patch_text", mock_apply_patch)
    monkeypatch.setattr(ops, "verify_syntax", mock_verify_syntax)

    orchestrator = RepairLoopOrchestrator(
        base_prompt="Fix the bug",
        command="pytest",
        model="gpt-4o",
        fetch_response=mock_fetch,
        config=RepairLoopConfig(max_retries=3, auto_archive=False),
        app_config=app_config,
        workspace_root=tmp_path,
    )

    events = await collect_events(orchestrator)
    event_kinds = [e.kind for e in events]

    # Should have SYNTAX_ERROR event
    assert EventKind.SYNTAX_ERROR in event_kinds

    # Find the syntax error event and verify it has error data
    syntax_events = [e for e in events if e.kind == EventKind.SYNTAX_ERROR]
    assert syntax_events
    assert "error" in syntax_events[0].data
    assert "SyntaxError" in syntax_events[0].data["error"]
