from __future__ import annotations
from unittest.mock import patch, MagicMock
import typer
from pathlib import Path
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
    ctx.obj = mock_state

agent_app.command(name="fix")(codex_exec)

def test_codex_exec_builds_command(runner):
    """Verify jp fix constructs the correct codex CLI call."""
    # We patch shutil.which to ensure the command doesn't fail validation
    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"):

        result = runner.invoke(agent_app, ["fix", "Fix the bug", "--full-auto"])

        assert result.exit_code == 0

        args, _ = mock_run.call_args
        cmd = args[0]

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "Fix the bug" in cmd
        assert "--full-auto" in cmd

def test_codex_exec_attaches_recent_files(runner):
    """Verify --recent flag scans and attaches files."""
    mock_entry = MagicMock()
    mock_entry.path = Path("fake_recent.py")

    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"), \
         patch("jpscripts.commands.agent.scan_recent", return_value=[mock_entry]):

        result = runner.invoke(agent_app, ["fix", "Refactor", "--recent"])

        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]

        # Ensure the file flag was added
        assert "--file" in cmd
        assert "fake_recent.py" in cmd
