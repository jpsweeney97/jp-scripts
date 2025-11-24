from __future__ import annotations

import asyncio
from pathlib import Path

from typer.testing import CliRunner

import jpscripts.main as jp_main


def test_help(runner: CliRunner):
    result = runner.invoke(jp_main.app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_doctor_mocked(runner: CliRunner, monkeypatch):
    tool = jp_main.ExternalTool(name="mock", binary="mock-bin", required=False)
    fake = jp_main.ToolCheck(tool=tool, status="ok", version="1.0.0")

    async def fake_run(_tools):
        return [fake]

    monkeypatch.setattr(jp_main, "_run_doctor", fake_run)

    result = runner.invoke(jp_main.app, ["doctor"])
    assert result.exit_code == 0
    assert "mock" in result.stdout


def test_config_defaults(runner: CliRunner, isolate_config: Path):
    # Ensure the config file exists but empty to trigger defaults
    isolate_config.touch()
    result = runner.invoke(jp_main.app, ["config"])
    assert result.exit_code == 0
    assert "notes_dir" in result.stdout
