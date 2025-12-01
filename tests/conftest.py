from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

# Detect CI environment (GitHub Actions sets CI=true)
IS_CI = os.environ.get("CI", "").lower() == "true"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "local_only: marks tests that require local environment (skip in CI)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip local_only tests when running in CI."""
    if not IS_CI:
        return
    skip_ci = pytest.mark.skip(reason="Skipped in CI (requires local environment)")
    for item in items:
        if "local_only" in item.keywords:
            item.add_marker(skip_ci)


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def ensure_commands_registered() -> None:
    """Ensure CLI commands are registered before tests run.

    Commands are lazily registered in production to improve CLI startup time.
    Tests using CliRunner need commands pre-registered since they bypass cli().
    """
    from jpscripts.main import _register_commands

    _register_commands()


@pytest.fixture(autouse=True)
def isolate_config(tmp_path: Path, monkeypatch: Any) -> Path:
    """Point config to a temp path so tests don't touch user state."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setenv("JPSCRIPTS_CONFIG", str(cfg_path))
    return cfg_path


@pytest.fixture(autouse=True)
def capture_console(monkeypatch: Any) -> Console:
    """Use an in-memory Rich console during tests."""
    test_console = Console(record=True)
    import jpscripts.core.console as core_console
    import jpscripts.main as jp_main

    monkeypatch.setattr(core_console, "console", test_console)
    monkeypatch.setattr(jp_main, "console", test_console)
    return test_console
