from __future__ import annotations

import shutil
from pathlib import Path
from typer.testing import CliRunner
from typer.main import get_command

from jpscripts import __version__
import jpscripts.commands.handbook as handbook_cmd
import jpscripts.core.diagnostics as diagnostics
from jpscripts.main import app

runner = CliRunner()

def test_app_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout

def test_doctor_mocked(monkeypatch):
    """Ensure doctor runs without crashing even if tools are missing."""
    tool = diagnostics.ExternalTool(name="mock", binary="mock-bin", required=False)
    fake = diagnostics.ToolCheck(tool=tool, status="ok", version="1.0.0")

    async def fake_run(_tools):
        return [fake]

    monkeypatch.setattr(diagnostics, "_run_doctor", fake_run)
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

def test_handbook_semantic_query(monkeypatch):
    """Ensure handbook semantic search runs without crashing and caches the index."""
    cache_root = Path.cwd() / ".tmp_handbook_cache"
    meta_path = cache_root / "meta.json"
    entries_path = cache_root / "entries.jsonl"
    store_path = cache_root / "lance"

    def _fake_cache_paths():
        base_root = Path.cwd()
        cache_dir = handbook_cmd.validate_path(cache_root, base_root)
        return (
            cache_dir,
            handbook_cmd.validate_path(meta_path, base_root),
            handbook_cmd.validate_path(entries_path, base_root),
            handbook_cmd.validate_path(store_path, base_root),
        )

    monkeypatch.setattr(handbook_cmd, "_cache_paths", _fake_cache_paths)
    try:
        result = runner.invoke(app, ["handbook", "mission"])
        assert result.exit_code == 0
        assert entries_path.exists()
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)


def test_serialize_snapshot_smoke(tmp_path: Path):
    """Ensure serialize command can emit a manifest in an empty workspace."""
    env = {"JP_WORKSPACE_ROOT": str(tmp_path)}
    output_path = tmp_path / "manifest.yaml"
    result = runner.invoke(app, ["serialize", "snapshot", "--output", str(output_path)], env=env)
    assert result.exit_code == 0
    assert output_path.exists()


def test_watch_help():
    """Smoke test for watch command help."""
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "God-Mode" in result.stdout or "watch" in result.stdout.lower()


def test_mcp_server_imports():
    """
    Critical Test: Ensure MCP server module can be imported without errors.
    This catches missing imports (like the Iterable bug) that would crash the server.
    """
    from jpscripts.mcp.server import create_server
    server = create_server()
    assert server is not None
