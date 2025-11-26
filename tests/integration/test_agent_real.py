from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from jpscripts.commands import agent as agent_cmd
from jpscripts.core import agent as agent_core


@pytest.mark.slow
def test_agent_xml_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_git_context(_root: Path) -> tuple[str, str, bool]:
        return "main", "abcdef0", False

    async def fake_git_diff(_root: Path, _max_chars: int) -> str:
        return "diff chunk with ]]> marker"

    monkeypatch.setattr(agent_core, "_collect_git_context", fake_git_context)
    monkeypatch.setattr(agent_core, "_collect_git_diff", fake_git_diff)

    class FakePopen:
        last_cmd: list[str] | None = None

        def __init__(self, *args, **_kwargs) -> None:
            FakePopen.last_cmd = list(args[0])
            self.stdout = io.StringIO('{"event":"turn.completed","data":{"assistant_message":"done"}}\n')
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(agent_cmd, "_ensure_codex", lambda: "codex")
    monkeypatch.setattr(agent_cmd.subprocess, "Popen", FakePopen)

    state = SimpleNamespace(
        config=SimpleNamespace(
            workspace_root=tmp_path,
            notes_dir=tmp_path,
            default_model="gpt-test",
            ignore_dirs=[],
            max_file_context_chars=5000,
            max_command_output_chars=5000,
        )
    )
    ctx = SimpleNamespace(obj=state)

    agent_cmd.codex_exec(
        ctx,
        prompt="Fix the bug",
        attach_recent=False,
        diff=True,
        run_command=None,
        full_auto=True,
        model=None,
    )

    assert FakePopen.last_cmd is not None
    prompt = FakePopen.last_cmd[-1]
    assert "<system_context>" in prompt
    assert "<git_diff>" in prompt
    assert "<instruction>" in prompt
    assert "<git_context>" in prompt
    assert "<![CDATA[" in prompt
    assert prompt.rfind("<![CDATA[") < prompt.rfind("]]>")
    cdata_sections = re.findall(r"<!\[CDATA\[.*?\]\]>", prompt, flags=re.S)
    assert cdata_sections
