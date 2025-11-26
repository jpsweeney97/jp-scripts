from __future__ import annotations
from unittest.mock import MagicMock, patch
import io
from pathlib import Path

import typer

from jpscripts.commands.agent import codex_exec

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
