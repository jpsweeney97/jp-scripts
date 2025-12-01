from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

if TYPE_CHECKING:

    class AgentResponseProto(Protocol):
        final_message: str | None


from jpscripts.commands.agent import codex_exec
from jpscripts.core.agent import PreparedPrompt, parse_agent_response
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Ok
from jpscripts.core.runtime import RuntimeContext, runtime_context, set_runtime_context

# Setup a test harness that mimics the main app's context injection
agent_app = typer.Typer()

# Module-level temp dir for test harness (created once per test session)
_test_temp_dir: Path | None = None


def _get_test_temp_dir() -> Path:
    """Get or create a temp directory for the test harness."""
    global _test_temp_dir
    if _test_temp_dir is None or not _test_temp_dir.exists():
        _test_temp_dir = Path(tempfile.mkdtemp(prefix="jpscripts_test_"))
    return _test_temp_dir


@agent_app.callback()
def main_callback(ctx: typer.Context) -> None:
    # Inject a real config and runtime context so get_runtime() succeeds
    temp_dir = _get_test_temp_dir()
    config = AppConfig(
        workspace_root=temp_dir,
        notes_dir=temp_dir / "notes",
        ignore_dirs=[".git", "node_modules"],
        max_file_context_chars=50_000,
        max_command_output_chars=20_000,
        default_model="gpt-4o",
        model_context_limits={"gpt-4o": 128_000, "default": 50_000},
        use_semantic_search=False,
    )
    runtime = RuntimeContext(
        config=config,
        workspace_root=config.workspace_root,
        dry_run=False,
    )
    set_runtime_context(runtime)

    mock_state = MagicMock()
    mock_state.config = config
    ctx.obj = mock_state


agent_app.command(name="fix")(codex_exec)


def test_codex_exec_invokes_provider(runner: CliRunner) -> None:
    """Verify jp fix invokes the provider correctly."""
    captured: list[str] = []

    async def fake_fetch_response(
        prepared: PreparedPrompt,
        config: Any,
        model: str,
        provider_type: Any,
        full_auto: bool = False,
        web: bool = False,
    ) -> str:
        captured.append(prepared.prompt)
        return json.dumps(
            {
                "thought_process": "done",
                "criticism": None,
                "tool_call": None,
                "file_patch": None,
                "final_message": "Completed",
            }
        )

    with patch("jpscripts.commands.agent._fetch_agent_response", side_effect=fake_fetch_response):
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
            prepared: PreparedPrompt,
            config: Any,
            model: str,
            provider_type: Any,
            full_auto: bool = False,
            web: bool = False,
        ) -> str:
            captured.append(prepared.prompt)
            return json.dumps(
                {
                    "thought_process": "done",
                    "criticism": None,
                    "tool_call": None,
                    "file_patch": None,
                    "final_message": "Completed",
                }
            )

        with (
            patch(
                "jpscripts.commands.agent._fetch_agent_response", side_effect=fake_fetch_response
            ),
            patch("jpscripts.core.agent.prompting.scan_recent", side_effect=fake_scan_recent),
        ):
            result = runner.invoke(agent_app, ["fix", "Refactor", "--recent"])

            assert result.exit_code == 0
            assert captured

            # Prompt should include the recent file snippet/path
            prompt = captured[0]
            assert "fake_recent.py" in prompt


def test_run_repair_loop_auto_archives(monkeypatch: Any, tmp_path: Path) -> None:
    from importlib import import_module

    agent_core = import_module("jpscripts.core.agent")
    agent_execution = import_module("jpscripts.core.agent.execution")
    config_mod = import_module("jpscripts.core.config")
    AppConfig = cast(Any, config_mod).AppConfig

    config = AppConfig(workspace_root=tmp_path, notes_dir=tmp_path, use_semantic_search=False)

    async def fake_run_command(command: str, root: Path) -> tuple[int, str, str]:
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

    monkeypatch.setattr(agent_execution, "_run_command", fake_run_command)
    monkeypatch.setattr(agent_execution, "save_memory", fake_save_memory)

    with runtime_context(config, workspace=tmp_path):
        success = asyncio.run(
            agent_core.run_repair_loop(
                base_prompt="Fix the thing",
                command="echo ok",
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
    base: dict[str, object] = {
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
        parsed: AgentResponseProto = parse_agent_response(payload)
        assert parsed.final_message == "All good"
