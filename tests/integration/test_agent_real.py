from __future__ import annotations

import asyncio
import io
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from jpscripts.commands import agent as agent_cmd
from jpscripts.core.config import AppConfig
import typer
from typing import cast
from jpscripts.core import agent as agent_core


@pytest.mark.slow
def test_agent_xml_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_git_context(_root: Path) -> tuple[str, str, bool]:
        return "main", "abcdef0", False

    async def fake_git_diff(_root: Path, _max_chars: int) -> str:
        return "diff chunk with ]]> marker"

    monkeypatch.setattr(agent_core, "_collect_git_context", fake_git_context)
    monkeypatch.setattr(agent_core, "_collect_git_diff", fake_git_diff)

    captured_cmd: list[str] | None = None

    async def fake_execute(cmd, *, status_label):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        return ["done"], None

    monkeypatch.setattr(agent_cmd, "_ensure_codex", lambda: "codex")
    monkeypatch.setattr(agent_cmd, "_execute_codex_prompt", fake_execute)

    config = AppConfig(
        workspace_root=tmp_path,
        notes_dir=tmp_path,
        default_model="gpt-test",
        ignore_dirs=[],
        max_file_context_chars=5000,
        max_command_output_chars=5000,
    )
    state = SimpleNamespace(config=config)
    ctx = cast(typer.Context, SimpleNamespace(obj=state))

    agent_cmd.codex_exec(
        ctx,
        prompt="Fix the bug",
        attach_recent=False,
        diff=True,
        run_command=None,
        full_auto=True,
        model=None,
        loop=False,
        max_retries=3,
        keep_failed=False,
    )

    assert captured_cmd is not None
    prompt = captured_cmd[-1]
    assert "<system_context>" in prompt
    assert "<git_diff>" in prompt
    assert "<instruction>" in prompt
    assert "<git_context>" in prompt
    assert "<![CDATA[" in prompt
    assert prompt.rfind("<![CDATA[") < prompt.rfind("]]>")
    cdata_sections = re.findall(r"<!\[CDATA\[.*?\]\]>", prompt, flags=re.S)
    assert cdata_sections


def test_repair_loop_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    script = tmp_path / "script.py"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path)

    async def fake_prepare_agent_prompt(base_prompt: str, **_kwargs) -> agent_core.PreparedPrompt:
        return agent_core.PreparedPrompt(prompt=base_prompt, attached_files=[])

    monkeypatch.setattr(agent_core, "prepare_agent_prompt", fake_prepare_agent_prompt)

    patch_text = textwrap.dedent(
        """\
        diff --git a/script.py b/script.py
        index 1111111..2222222 100644
        --- a/script.py
        +++ b/script.py
        @@ -1,2 +1,1 @@
        -import sys
        -sys.exit(1)
        +print("ok")
        """
    )

    async def fake_fetch(_prepared: agent_core.PreparedPrompt) -> str:
        return patch_text

    success = asyncio.run(
        agent_core.run_repair_loop(
            base_prompt="fix loop",
            command=f"{sys.executable} {script}",
            config=config,
            attach_recent=False,
            include_diff=False,
            fetch_patch=fake_fetch,
            max_retries=2,
            keep_failed=False,
        )
    )

    assert success
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0
    assert "ok" in result.stdout
