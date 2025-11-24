from __future__ import annotations

from pathlib import Path
import pytest
from typer.testing import CliRunner
from typer.main import get_command

import jpscripts.main as jp_main
from jpscripts.main import app

runner = CliRunner()

def test_app_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout

def test_doctor_mocked(monkeypatch):
    """Ensure doctor runs without crashing even if tools are missing."""
    tool = jp_main.ExternalTool(name="mock", binary="mock-bin", required=False)
    fake = jp_main.ToolCheck(tool=tool, status="ok", version="1.0.0")

    async def fake_run(_tools):
        return [fake]

    monkeypatch.setattr(jp_main, "_run_doctor", fake_run)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "mock" in result.stdout

def test_all_commands_have_help():
    """
    Critical Smoke Test: Iterate over EVERY registered command and ensure
    it accepts --help. This catches import errors, syntax errors in decorators,
    and missing dependencies in the command modules.
    """
    click_app = get_command(app)
    for name in click_app.commands:
        # We skip 'com' because it's a simple catalog, but we test the rest
        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, f"Command 'jp {name} --help' failed!"
        assert "Usage:" in result.stdout

def test_init_command(isolate_config: Path):
    """Ensure init generates a config file."""
    # We pipe 'input' to simulate pressing Enter for all defaults
    inputs = "\n" * 10
    result = runner.invoke(app, ["init"], input=inputs)
    assert result.exit_code == 0
    assert isolate_config.exists()
    content = isolate_config.read_text()
    assert 'editor = "code -w"' in content
