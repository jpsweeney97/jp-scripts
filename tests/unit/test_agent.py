from __future__ import annotations
from unittest.mock import patch, MagicMock
import typer
from pathlib import Path
from jpscripts.commands.agent import codex_exec

# Create a test harness app
test_app = typer.Typer()
test_app.command(name="fix")(codex_exec)

def test_codex_exec_builds_command(runner, isolate_config):
    """Verify jp fix constructs the correct codex CLI call."""
    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"):

        # Invoke the test_app, ensuring we pass the subcommand name "fix"
        result = runner.invoke(test_app, ["fix", "Fix the bug", "--full-auto"])

        assert result.exit_code == 0

        args, _ = mock_run.call_args
        cmd = args[0]

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "Fix the bug" in cmd
        assert "--full-auto" in cmd

def test_codex_exec_attaches_recent_files(runner, isolate_config, monkeypatch):
    """Verify --recent flag scans and attaches files."""
    # Mock scan_recent to return a fake file object
    mock_entry = MagicMock()
    mock_entry.path = Path("fake_recent.py")

    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"), \
         patch("jpscripts.commands.agent.scan_recent", return_value=[mock_entry]):

        result = runner.invoke(test_app, ["fix", "Refactor", "--recent"])

        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]

        # Ensure the file flag was added correctly
        assert "--file" in cmd
        assert "fake_recent.py" in cmd
