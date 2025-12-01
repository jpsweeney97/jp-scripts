from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import typer

from jpscripts.commands import agent as agent_cmd
from jpscripts.core import agent as agent_core
from jpscripts.core.agent import prompting as agent_prompting
from jpscripts.core.config import AppConfig
from jpscripts.core.runtime import runtime_context
from tests.mocks.mock_provider import MockProvider


@pytest.mark.slow
def test_agent_prompt_includes_json_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Mock git context (external dependency)
    async def fake_git_context(_root: Path) -> tuple[str, str, bool]:
        return "main", "abcdef0", False

    async def fake_git_diff(_root: Path, _max_chars: int) -> str:
        return "diff chunk with ]]> marker"

    monkeypatch.setattr(agent_prompting, "collect_git_context", fake_git_context)
    monkeypatch.setattr(agent_prompting, "collect_git_diff", fake_git_diff)

    # Create MockProvider - it already returns valid agent JSON responses
    mock_provider = MockProvider()

    # Patch get_provider in agent_cmd module (tests the full provider stack)
    def fake_get_provider(*_args: Any, **_kwargs: Any) -> MockProvider:
        return mock_provider

    monkeypatch.setattr(agent_cmd, "get_provider", fake_get_provider)

    config = AppConfig(
        workspace_root=tmp_path,
        notes_dir=tmp_path,
        default_model="gpt-test",
        ignore_dirs=[],
        max_file_context_chars=5000,
        max_command_output_chars=5000,
        use_semantic_search=False,
    )
    state = SimpleNamespace(config=config)
    ctx = cast(typer.Context, SimpleNamespace(obj=state))

    with runtime_context(config, workspace=tmp_path):
        agent_cmd.codex_exec(
            ctx,
            prompt="Fix the bug",
            attach_recent=False,
            diff=True,
            run_command=None,
            full_auto=True,
            model=None,
            provider=None,
            loop=False,
            max_retries=3,
            keep_failed=False,
            archive=True,
            web=False,
        )

    # Verify prompt was sent through the provider stack
    call_log = mock_provider.call_log
    assert len(call_log) > 0, "MockProvider should have received at least one message"

    # Get the prompt content from the captured message
    captured_message = call_log[0]
    assert captured_message is not None
    captured_prompt = captured_message.content

    prompt = json.loads(captured_prompt)
    assert "system_context" in prompt
    assert prompt["system_context"]["git_context"]["head"] == "abcdef0"
    assert prompt["git_diff"] == "diff chunk with ]]> marker"
    assert "instruction" in prompt
    assert "response_contract" in prompt


@pytest.mark.local_only
def test_repair_loop_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from jpscripts.core.agent import execution as agent_execution

    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    script = tmp_path / "script.py"
    script.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, use_semantic_search=False)

    async def fake_prepare_agent_prompt(
        base_prompt: str, **_kwargs: Any
    ) -> agent_core.PreparedPrompt:
        return agent_core.PreparedPrompt(prompt=base_prompt, attached_files=[])

    monkeypatch.setattr(agent_core, "prepare_agent_prompt", fake_prepare_agent_prompt)

    # Mock _run_command to bypass security policy in tests
    async def fake_run_command(command: str, root: Path) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    monkeypatch.setattr(agent_execution, "_run_command", fake_run_command)

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
        return json.dumps(
            {
                "thought_process": "apply patch",
                "criticism": "Self-correction applied.",
                "tool_call": None,
                "file_patch": patch_text,
                "final_message": None,
            }
        )

    with runtime_context(config, workspace=tmp_path):
        success = asyncio.run(
            agent_core.run_repair_loop(
                base_prompt="fix loop",
                command=f"{sys.executable} {script}",
                model=config.default_model,
                attach_recent=False,
                include_diff=False,
                fetch_response=fake_fetch,
                max_retries=2,
                keep_failed=False,
            )
        )

    assert success
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "ok" in result.stdout
