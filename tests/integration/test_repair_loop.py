"""Integration tests for the agent repair loop.

These tests verify the autonomous repair loop works end-to-end
using mock LLM responses instead of real API calls.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jpscripts.core.agent import PreparedPrompt, run_repair_loop
from jpscripts.core.agent import execution as agent_execution
from jpscripts.core.config import AppConfig
from jpscripts.core.runtime import runtime_context


@pytest.fixture
def bypass_security(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass command security policy for integration tests.

    The security policy blocks Python interpreters by default (FORBIDDEN_BINARIES),
    but these tests need to run Python commands to verify the repair loop.
    """

    async def fake_run_command(command: str, root: Path) -> tuple[int, str, str]:
        """Execute command directly without security validation."""
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    monkeypatch.setattr(agent_execution, "_run_command", fake_run_command)


@pytest.mark.local_only
class TestRepairLoopIntegration:
    """Integration tests for the autonomous repair loop.

    These tests require git apply to work correctly with patches, which can
    behave differently in CI environments. Run locally for full coverage.
    """

    def test_repairs_syntax_error(self, tmp_path: Path, bypass_security: None) -> None:
        """Full loop: syntax error -> mock returns patch -> code fixed."""
        # Setup: git repo with broken script (needs initial commit for git apply)
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        script = tmp_path / "broken.py"
        script.write_text("def foo(\n    print('missing paren')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        # Patch that fixes the syntax error - replace all lines to avoid context matching issues
        fix_patch = textwrap.dedent("""\
            diff --git a/broken.py b/broken.py
            index 1111111..2222222 100644
            --- a/broken.py
            +++ b/broken.py
            @@ -1,2 +1,2 @@
            -def foo(
            -    print('missing paren')
            +def foo():
            +    print('missing paren')
            """)

        # Mock returns the fix
        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return json.dumps(
                {
                    "thought_process": "Detected syntax error in foo() definition",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": fix_patch,
                    "final_message": None,
                }
            )

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Fix the syntax error",
                    command=f"{sys.executable} -m py_compile {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=2,
                    keep_failed=False,
                )
            )

        assert success, "Repair loop should succeed"
        # Verify the file compiles now
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True,
        )
        assert result.returncode == 0, f"Fixed file should compile: {result.stderr.decode()}"

    def test_repairs_runtime_error(self, tmp_path: Path, bypass_security: None) -> None:
        """Full loop: runtime error -> mock returns patch -> script runs."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "script.py"
        script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

        # Patch that fixes the exit code
        fix_patch = textwrap.dedent("""\
            diff --git a/script.py b/script.py
            index 1111111..2222222 100644
            --- a/script.py
            +++ b/script.py
            @@ -1,2 +1,1 @@
            -import sys
            -sys.exit(1)
            +print("ok")
            """)

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return json.dumps(
                {
                    "thought_process": "Script exits with error, replacing with print",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": fix_patch,
                    "final_message": None,
                }
            )

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Fix the script",
                    command=f"{sys.executable} {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=2,
                    keep_failed=False,
                )
            )

        assert success
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_loop_succeeds_when_command_passes(self, tmp_path: Path, bypass_security: None) -> None:
        """Loop should succeed immediately if command passes on first try."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "working.py"
        script.write_text("print('already works')\n", encoding="utf-8")

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return json.dumps(
                {
                    "thought_process": "No fix needed",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": None,
                    "final_message": "Code is already working",
                }
            )

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Check the script",
                    command=f"{sys.executable} {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=3,
                    keep_failed=False,
                )
            )

        assert success
        # When command passes on first try, summary fetch may or may not be called
        # depending on auto_archive setting - we just verify the loop succeeds

    def test_loop_stops_after_max_retries(self, tmp_path: Path) -> None:
        """Loop should fail after exhausting retries without success."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "unfixable.py"
        script.write_text("syntax error here!!!\n", encoding="utf-8")

        call_count = 0

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            nonlocal call_count
            call_count += 1
            # Return patches that don't actually fix the syntax error
            return json.dumps(
                {
                    "thought_process": f"Attempt {call_count}",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": None,  # No patch = no fix
                    "final_message": None,
                }
            )

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Fix the syntax error",
                    command=f"{sys.executable} -m py_compile {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=2,
                    keep_failed=False,
                )
            )

        assert not success, "Should fail after max retries"
        assert call_count == 2, "Should have tried exactly max_retries times"


class TestCircuitBreaker:
    """Tests for circuit breaker behavior with oversized responses."""

    def test_handles_huge_response(self, tmp_path: Path, bypass_security: None) -> None:
        """Loop should handle oversized responses gracefully."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "test.py"
        script.write_text("print('ok')\n", encoding="utf-8")

        # Mock returns a massive response
        huge_content = "x" * 200_000  # ~200KB of garbage

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return json.dumps(
                {
                    "thought_process": huge_content,
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": None,
                    "final_message": None,
                }
            )

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
            max_command_output_chars=10_000,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Do something",
                    command=f"{sys.executable} {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=1,
                    keep_failed=False,
                )
            )

        # Command passes (script works), so loop succeeds despite huge response
        assert success

    def test_empty_response_handled(self, tmp_path: Path) -> None:
        """Loop should handle empty/invalid JSON responses."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "test.py"
        script.write_text("syntax error!\n", encoding="utf-8")

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return ""  # Empty response

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Fix it",
                    command=f"{sys.executable} -m py_compile {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=1,
                    keep_failed=False,
                )
            )

        # Should fail but not crash
        assert not success

    def test_malformed_json_handled(self, tmp_path: Path) -> None:
        """Loop should handle malformed JSON responses."""
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        script = tmp_path / "test.py"
        script.write_text("syntax error!\n", encoding="utf-8")

        async def mock_fetch(prepared: PreparedPrompt) -> str:
            return "{not valid json"

        config = AppConfig(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            use_semantic_search=False,
        )

        with runtime_context(config, workspace=tmp_path):
            success = asyncio.run(
                run_repair_loop(
                    base_prompt="Fix it",
                    command=f"{sys.executable} -m py_compile {script}",
                    model="mock-model",
                    attach_recent=False,
                    include_diff=False,
                    fetch_response=mock_fetch,
                    max_retries=1,
                    keep_failed=False,
                )
            )

        # Should fail but not crash
        assert not success
