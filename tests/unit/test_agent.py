from __future__ import annotations
from unittest.mock import patch, MagicMock
from jpscripts.commands.agent import codex_exec
from pathlib import Path

def test_codex_exec_builds_command(runner, isolate_config):
    """Verify jp fix constructs the correct codex CLI call."""
    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"):

        result = runner.invoke(codex_exec, ["Fix the bug", "--full-auto"])

        assert result.exit_code == 0

        args, _ = mock_run.call_args
        cmd = args[0]

        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        assert "Fix the bug" in cmd
        assert "--full-auto" in cmd

def test_codex_exec_attaches_recent_files(runner, isolate_config, monkeypatch):
    """Verify --recent flag scans and attaches files."""
    # Mock scan_recent to return a fake file
    mock_entry = MagicMock()
    mock_entry.path = Path("fake_recent.py")

    with patch("jpscripts.commands.agent.subprocess.run") as mock_run, \
         patch("jpscripts.commands.agent.shutil.which", return_value="/usr/bin/codex"), \
         patch("jpscripts.commands.agent.scan_recent", return_value=[mock_entry]):

        result = runner.invoke(codex_exec, ["Refactor", "--recent"])

        assert result.exit_code == 0
        cmd = mock_run.call_args[0][0]

        # Ensure the file flag was added
        assert "--file" in cmd
        assert "fake_recent.py" in cmd
