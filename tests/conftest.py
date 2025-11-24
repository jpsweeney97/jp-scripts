from __future__ import annotations


import pytest
from rich.console import Console
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Point config to a temp path so tests don't touch user state."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setenv("JPSCRIPTS_CONFIG", str(cfg_path))
    return cfg_path


@pytest.fixture(autouse=True)
def capture_console(monkeypatch):
    """Use an in-memory Rich console during tests."""
    test_console = Console(record=True)
    import jpscripts.core.console as core_console
    import jpscripts.main as jp_main

    monkeypatch.setattr(core_console, "console", test_console)
    monkeypatch.setattr(jp_main, "console", test_console)
    return test_console
