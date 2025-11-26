from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jpscripts.main import app
from jpscripts.core import nav as nav_core


@pytest.fixture
def workspace_env(tmp_path: Path, isolate_config: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=workspace, check=True)

    notes_dir = workspace / "notes"
    snapshots_dir = workspace / "snapshots"
    notes_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    config_body = "\n".join(
        [
            f'editor = "code -w"',
            f'notes_dir = "{notes_dir}"',
            f'workspace_root = "{workspace}"',
            'ignore_dirs = [".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"]',
            f'snapshots_dir = "{snapshots_dir}"',
            'log_level = "INFO"',
            f'worktree_root = "{workspace}"',
            'focus_audio_device = ""',
        ]
    )
    isolate_config.write_text(config_body + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "JP_WORKSPACE_ROOT": str(workspace),
        "JP_NOTES_DIR": str(notes_dir),
        "JP_SNAPSHOTS_DIR": str(snapshots_dir),
        "JP_MEMORY_STORE": str(workspace / "memory.sqlite"),
        "JP_WORKTREE_ROOT": str(workspace),
        "JPSCRIPTS_CONFIG": str(isolate_config),
    }

    return workspace, env


def test_end_to_end_fix_and_nav(
    workspace_env: tuple[Path, dict[str, str]],
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    capture_console,
) -> None:
    workspace, env = workspace_env

    from jpscripts.core.config import load_config

    cfg, _ = load_config(env=env)
    assert cfg.workspace_root == workspace

    buggy_file = workspace / "buggy.py"
    buggy_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
    print("workspace ready", buggy_file, flush=True)

    monkeypatch.setattr("jpscripts.commands.agent._ensure_codex", lambda: "/usr/bin/codex")
    async def fake_execute(cmd, *, status_label):
        return ["fixed bug"], None

    monkeypatch.setattr("jpscripts.commands.agent._execute_codex_prompt", fake_execute)
    from jpscripts.core.agent import PreparedPrompt

    async def fake_prepare_agent_prompt(**_kwargs) -> PreparedPrompt:
        return PreparedPrompt(prompt="fix the bug", attached_files=[])

    monkeypatch.setattr("jpscripts.commands.agent.prepare_agent_prompt", fake_prepare_agent_prompt)

    result_fix = runner.invoke(app, ["fix", "buggy.py"], env=env)
    assert result_fix.exit_code == 0
    print("fix command complete", flush=True)

    result_recent = runner.invoke(
        app,
        ["recent", "--no-fzf", "--files-only", "--limit", "1"],
        env=env,
    )
    assert result_recent.exit_code == 0
    entries = asyncio.run(
        nav_core.scan_recent(
            workspace,
            max_depth=4,
            include_dirs=False,
            ignore_dirs={".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"},
        )
    )
    assert entries and entries[0].path == buggy_file
    print("recent command complete", flush=True)

    result_sync = runner.invoke(app, ["sync"], env=env)
    assert result_sync.exit_code == 0
    print("sync command complete", flush=True)

    log_output = capture_console.export_text() or result_sync.stdout
    assert "fetched" in log_output
