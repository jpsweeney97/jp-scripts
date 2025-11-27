from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer

from jpscripts.commands.agent import codex_exec
from jpscripts.core.agent import parse_agent_response

# Setup a test harness that mimics the main app's context injection
agent_app = typer.Typer()

@agent_app.callback()
def main_callback(ctx: typer.Context):
    # Inject a mock state object so ctx.obj.config works
    mock_state = MagicMock()
    mock_state.config.workspace_root = Path("/mock/workspace")
    mock_state.config.notes_dir = Path("/mock/notes")
    mock_state.config.ignore_dirs = [".git", "node_modules"]
    mock_state.config.max_file_context_chars = 50_000
    mock_state.config.max_command_output_chars = 20_000
    mock_state.config.default_model = "test-model"
    mock_state.config.model_context_limits = {"test-model": 10_000, "default": 50_000}
    ctx.obj = mock_state

agent_app.command(name="fix")(codex_exec)

def test_codex_exec_builds_command(runner):
    """Verify jp fix constructs the correct codex CLI call."""
    captured: list[list[str]] = []

    async def fake_execute(cmd, *, status_label):
        captured.append(cmd)
        return ["done"], None

    with patch("jpscripts.commands.agent._execute_codex_prompt", side_effect=fake_execute), \
         patch("jpscripts.commands.agent._ensure_codex", return_value="/usr/bin/codex"):

        result = runner.invoke(agent_app, ["fix", "Fix the bug", "--full-auto"])

        assert result.exit_code == 0
        assert captured

        cmd = captured[0]

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert any("Fix the bug" in part for part in cmd)
        assert "--full-auto" in cmd

def test_codex_exec_attaches_recent_files(runner):
    """Verify --recent flag scans and attaches files."""
    mock_entry = MagicMock()
    mock_entry.path = Path("fake_recent.py")
    mock_entry.path.write_text("hello world", encoding="utf-8")

    captured: list[list[str]] = []

    async def fake_scan_recent(*_args, **_kwargs):
        return [mock_entry]

    async def fake_execute(cmd, *, status_label):
        captured.append(cmd)
        return ["done"], None

    with patch("jpscripts.commands.agent._execute_codex_prompt", side_effect=fake_execute), \
         patch("jpscripts.commands.agent._ensure_codex", return_value="/usr/bin/codex"), \
         patch("jpscripts.core.agent.scan_recent", side_effect=fake_scan_recent):

        result = runner.invoke(agent_app, ["fix", "Refactor", "--recent"])

        assert result.exit_code == 0
        assert captured
        cmd = captured[0]

        # Prompt (last arg) should include the recent file snippet/path
        prompt_arg = cmd[-1]
        assert "fake_recent.py" in prompt_arg


def test_run_repair_loop_auto_archives(monkeypatch, tmp_path: Path) -> None:
    from jpscripts.core import agent as agent_core
    from jpscripts.core.config import AppConfig

    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path)

    async def fake_run_shell_command(command: str, cwd: Path):
        return 0, "ok", ""

    calls: list[str] = []

    async def fake_fetch(prepared):
        calls.append(prepared.prompt)
        return "Fixed summary."

    saved: list[tuple[str, list[str] | None]] = []

    def fake_save_memory(content: str, tags=None, *, config=None, store_path=None):
        saved.append((content, tags))
        return MagicMock()

    monkeypatch.setattr(agent_core, "_run_shell_command", fake_run_shell_command)
    monkeypatch.setattr(agent_core, "save_memory", fake_save_memory)

    success = asyncio.run(
        agent_core.run_repair_loop(
            base_prompt="Fix the thing",
            command="echo ok",
            config=config,
            model=config.default_model,
            attach_recent=False,
            include_diff=False,
            fetch_response=fake_fetch,
            auto_archive=True,
            max_retries=1,
            keep_failed=False,
        )
    )

    assert success
    assert calls  # Summary fetch invoked
    assert saved
    assert "auto-fix" in (saved[0][1] or [])


def test_parse_agent_response_handles_json_variants() -> None:
    base = {
        "thought_process": "Reasoned",
        "tool_call": None,
        "file_patch": None,
        "final_message": "All good",
    }

    raw_json = json.dumps(base)
    fenced_json = f"```json\n{raw_json}\n```"
    prose_json = f"Here you go:\n{raw_json}\nThanks!"

    for payload in (raw_json, fenced_json, prose_json):
        parsed = parse_agent_response(payload)
        assert parsed.final_message == "All good"
