from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from typer.testing import CliRunner

import typer

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

if TYPE_CHECKING:
    T = TypeVar("T")

    class PreparedPrompt:
        prompt: str
        attached_files: list[Path]

    class Ok(Generic[T]):
        def __init__(self, value: T) -> None: ...

    class AgentResponseProto(Protocol):
        final_message: str | None

    def parse_agent_response(payload: str) -> AgentResponseProto: ...

    def codex_exec(*args: Any, **kwargs: Any) -> None: ...
else:  # pragma: no cover - runtime imports
    from jpscripts.commands.agent import codex_exec
    from jpscripts.core.agent import parse_agent_response
    from jpscripts.core.result import Ok

# Setup a test harness that mimics the main app's context injection
agent_app = typer.Typer()

@agent_app.callback()
def main_callback(ctx: typer.Context) -> None:
    # Inject a mock state object so ctx.obj.config works
    mock_state = MagicMock()
    mock_state.config.workspace_root = Path("/mock/workspace")
    mock_state.config.notes_dir = Path("/mock/notes")
    mock_state.config.ignore_dirs = [".git", "node_modules"]
    mock_state.config.max_file_context_chars = 50_000
    mock_state.config.max_command_output_chars = 20_000
    # Use a recognized OpenAI model so auto-detection triggers legacy Codex path
    mock_state.config.default_model = "gpt-4o"
    mock_state.config.model_context_limits = {"gpt-4o": 128_000, "default": 50_000}
    ctx.obj = mock_state

agent_app.command(name="fix")(codex_exec)

def test_codex_exec_invokes_provider(runner: CliRunner) -> None:
    """Verify jp fix invokes the provider correctly."""
    captured: list[str] = []

    async def fake_fetch_response(
        prepared: "PreparedPrompt",
        config: Any,
        model: str,
        provider_type: Any,
        full_auto: bool = False,
        web: bool = False,
    ) -> str:
        captured.append(prepared.prompt)
        return json.dumps({
            "thought_process": "done",
            "criticism": None,
            "tool_call": None,
            "file_patch": None,
            "final_message": "Completed",
        })

    with patch("jpscripts.commands.agent._fetch_agent_response", side_effect=fake_fetch_response), \
         patch("jpscripts.commands.agent.is_codex_available", return_value=False):

        result = runner.invoke(agent_app, ["fix", "Fix the bug", "--full-auto"])

        assert result.exit_code == 0
        assert captured

        # Verify prompt was captured
        prompt = captured[0]
        assert "Fix the bug" in prompt

def test_codex_exec_attaches_recent_files(runner: CliRunner) -> None:
    """Verify --recent flag scans and attaches files."""
    captured: list[str] = []

    with runner.isolated_filesystem():
        recent_path = Path("fake_recent.py")
        recent_path.write_text("hello world", encoding="utf-8")

        mock_entry = MagicMock()
        mock_entry.path = recent_path

        async def fake_scan_recent(*_args: Any, **_kwargs: Any) -> Any:
            return Ok([mock_entry])

        async def fake_fetch_response(
            prepared: "PreparedPrompt",
            config: Any,
            model: str,
            provider_type: Any,
            full_auto: bool = False,
            web: bool = False,
        ) -> str:
            captured.append(prepared.prompt)
            return json.dumps({
                "thought_process": "done",
                "criticism": None,
                "tool_call": None,
                "file_patch": None,
                "final_message": "Completed",
            })

        with patch("jpscripts.commands.agent._fetch_agent_response", side_effect=fake_fetch_response), \
             patch("jpscripts.commands.agent.is_codex_available", return_value=False), \
             patch("jpscripts.core.agent.scan_recent", side_effect=fake_scan_recent):

            result = runner.invoke(agent_app, ["fix", "Refactor", "--recent"])

            assert result.exit_code == 0
            assert captured

            # Prompt should include the recent file snippet/path
            prompt = captured[0]
            assert "fake_recent.py" in prompt


def test_run_repair_loop_auto_archives(monkeypatch: Any, tmp_path: Path) -> None:
    from importlib import import_module

    agent_core = import_module("jpscripts.core.agent")
    config_mod = import_module("jpscripts.core.config")
    AppConfig = cast(Any, config_mod).AppConfig

    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path)

    async def fake_run_shell_command(command: str, cwd: Path) -> tuple[int, str, str]:
        return 0, "ok", ""

    calls: list[str] = []

    async def fake_fetch(prepared: Any) -> str:
        calls.append(prepared.prompt)
        return "Fixed summary."

    saved: list[tuple[str, list[str] | None]] = []

    def fake_save_memory(
        content: str,
        tags: list[str] | None = None,
        *,
        config: Any = None,
        store_path: Any = None,
    ) -> MagicMock:
        saved.append((content, tags))
        return MagicMock()

    monkeypatch.setattr(agent_core, "_run_shell_command", fake_run_shell_command)
    monkeypatch.setattr(agent_core, "save_memory", fake_save_memory)

    success = asyncio.run(
        agent_core.run_repair_loop(
            base_prompt="Fix the thing",
            command="echo ok",
            config=config,
            model=config.default_model,
            attach_recent=False,
            include_diff=False,
            fetch_response=fake_fetch,
            auto_archive=True,
            max_retries=1,
            keep_failed=False,
        )
    )

    assert success
    assert calls  # Summary fetch invoked
    assert saved
    assert "auto-fix" in (saved[0][1] or [])


def test_parse_agent_response_handles_json_variants() -> None:
    base = {
        "thought_process": "Reasoned",
        "criticism": "No issues found",
        "tool_call": None,
        "file_patch": None,
        "final_message": "All good",
    }

    raw_json = json.dumps(base)
    fenced_json = f"```json\n{raw_json}\n```"
    prose_json = f"Here you go:\n{raw_json}\nThanks!"

    for payload in (raw_json, fenced_json, prose_json):
        parsed = parse_agent_response(payload)
        assert parsed.final_message == "All good"
