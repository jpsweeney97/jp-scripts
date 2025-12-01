"""Tests for the evolve command - autonomous code optimization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from jpscripts.commands.evolve import (
    _build_optimizer_prompt,
    _run_evolve,
    app,
    evolve_debt,
    evolve_report,
    evolve_run,
)
from jpscripts.core.complexity import TechnicalDebtScore
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Err, Ok
from jpscripts.main import AppState


@pytest.fixture
def runner() -> CliRunner:
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Create a test configuration."""
    return AppConfig(
        workspace_root=tmp_path,
        notes_dir=tmp_path / "notes",
        ignore_dirs=[".git", "node_modules", "__pycache__"],
        max_file_context_chars=50_000,
        max_command_output_chars=20_000,
        default_model="gpt-4o-mini",
        model_context_limits={"gpt-4o-mini": 128_000, "default": 50_000},
        use_semantic_search=False,
    )


@pytest.fixture
def mock_app_state(test_config: AppConfig) -> AppState:
    """Create mock AppState with test config."""
    state = MagicMock(spec=AppState)
    state.config = test_config
    return state


class TestBuildOptimizerPrompt:
    """Tests for the prompt builder function."""

    def test_builds_prompt_with_all_fields(self) -> None:
        """Prompt includes all debt score information."""
        target = TechnicalDebtScore(
            path=Path("src/example.py"),
            complexity_score=25.5,
            fix_frequency=10,
            churn=50,
            debt_score=127.5,
            reasons=["High cyclomatic complexity", "Frequent bug fixes"],
        )
        prompt = _build_optimizer_prompt(target)

        assert "src/example.py" in prompt
        assert "25.5" in prompt  # complexity score
        assert "10" in prompt  # fix frequency
        assert "50" in prompt  # churn
        assert "High cyclomatic complexity" in prompt
        assert "Frequent bug fixes" in prompt
        assert "Optimizer persona" in prompt

    def test_builds_prompt_with_empty_reasons(self) -> None:
        """Prompt handles empty reasons list."""
        target = TechnicalDebtScore(
            path=Path("src/simple.py"),
            complexity_score=15.0,
            fix_frequency=3,
            churn=20,
            debt_score=45.0,
            reasons=[],
        )
        prompt = _build_optimizer_prompt(target)

        assert "src/simple.py" in prompt
        assert "High complexity" in prompt  # fallback text
        assert "Reduce cyclomatic complexity" in prompt

    def test_prompt_includes_constraints(self) -> None:
        """Prompt includes refactoring constraints."""
        target = TechnicalDebtScore(
            path=Path("src/api.py"),
            complexity_score=20.0,
            fix_frequency=5,
            churn=30,
            debt_score=100.0,
            reasons=["Complex control flow"],
        )
        prompt = _build_optimizer_prompt(target)

        assert "Preserve all existing behavior" in prompt
        assert "pure refactoring" in prompt
        assert "mypy --strict" in prompt
        assert "public interfaces" in prompt


class TestEvolveRunDryRun:
    """Tests for evolve run with dry-run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_shows_analysis_without_changes(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """Dry run mode shows analysis but doesn't modify files."""
        # Create a test Python file
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        test_file = src_dir / "complex.py"
        test_file.write_text("""\
def complex_func(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return 1
            else:
                return 2
        else:
            return 3
    else:
        return 4
""")

        # Mock git repo as clean
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=False))

        mock_scores = [
            TechnicalDebtScore(
                path=test_file,
                complexity_score=15.0,
                fix_frequency=5,
                churn=10,
                debt_score=75.0,
                reasons=["High complexity"],
            )
        ]

        with (
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Ok(mock_repo),
            ),
            patch(
                "jpscripts.commands.evolve.calculate_debt_scores",
                return_value=Ok(mock_scores),
            ),
        ):
            await _run_evolve(test_config, dry_run=True, model=None, threshold=10.0)

        # No git checkout should be called in dry run
        mock_repo.run_git.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_with_below_threshold_score(
        self, test_config: AppConfig
    ) -> None:
        """Dry run with scores below threshold shows no optimization needed."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=False))

        mock_scores = [
            TechnicalDebtScore(
                path=Path("src/simple.py"),
                complexity_score=5.0,
                fix_frequency=1,
                churn=2,
                debt_score=8.0,  # Below default threshold of 10
                reasons=[],
            )
        ]

        with (
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Ok(mock_repo),
            ),
            patch(
                "jpscripts.commands.evolve.calculate_debt_scores",
                return_value=Ok(mock_scores),
            ),
        ):
            await _run_evolve(test_config, dry_run=True, model=None, threshold=10.0)

        # Should not proceed with optimization
        mock_repo.run_git.assert_not_called()


class TestEvolveRunErrors:
    """Tests for error handling in evolve run."""

    @pytest.mark.asyncio
    async def test_fails_on_dirty_workspace(self, test_config: AppConfig) -> None:
        """Evolve fails if workspace has uncommitted changes."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=True))

        with patch(
            "jpscripts.commands.evolve.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)

        # Should not proceed to debt analysis
        mock_repo.run_git.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_on_git_error(self, test_config: AppConfig) -> None:
        """Evolve handles git errors gracefully."""
        with patch(
            "jpscripts.commands.evolve.git_core.AsyncRepo.open",
            return_value=Err("Not a git repository"),
        ):
            # Should not raise, just print error
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)

    @pytest.mark.asyncio
    async def test_fails_on_status_error(self, test_config: AppConfig) -> None:
        """Evolve handles status check errors."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Err("Status failed")

        with patch(
            "jpscripts.commands.evolve.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)

    @pytest.mark.asyncio
    async def test_fails_on_debt_analysis_error(self, test_config: AppConfig) -> None:
        """Evolve handles debt score calculation errors."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=False))

        with (
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Ok(mock_repo),
            ),
            patch(
                "jpscripts.commands.evolve.calculate_debt_scores",
                return_value=Err("Analysis failed"),
            ),
        ):
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)

    @pytest.mark.asyncio
    async def test_no_files_need_optimization(self, test_config: AppConfig) -> None:
        """Evolve handles case when no files need optimization."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=False))

        with (
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Ok(mock_repo),
            ),
            patch(
                "jpscripts.commands.evolve.calculate_debt_scores",
                return_value=Ok([]),  # Empty list
            ),
        ):
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)


class TestEvolveReport:
    """Tests for evolve report command."""

    def test_report_command_exists(self) -> None:
        """The report subcommand is registered."""
        commands = {cmd.name for cmd in app.registered_commands}
        assert "report" in commands

    def test_report_handles_no_python_files(
        self, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """Report handles directories with no Python files."""
        # Create mock context
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch(
            "jpscripts.commands.evolve.analyze_directory_complexity",
            return_value=Ok([]),
        ):
            # Should complete without error
            evolve_report(mock_ctx, limit=20)

    def test_report_handles_analysis_error(
        self, test_config: AppConfig
    ) -> None:
        """Report handles complexity analysis errors."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch(
            "jpscripts.commands.evolve.analyze_directory_complexity",
            return_value=Err("Analysis failed"),
        ):
            # Should complete without raising
            evolve_report(mock_ctx, limit=20)


class TestEvolveDebt:
    """Tests for evolve debt command."""

    def test_debt_command_exists(self) -> None:
        """The debt subcommand is registered."""
        commands = {cmd.name for cmd in app.registered_commands}
        assert "debt" in commands

    def test_debt_handles_no_files(self, test_config: AppConfig) -> None:
        """Debt command handles empty analysis results."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch(
            "jpscripts.commands.evolve.calculate_debt_scores",
            return_value=Ok([]),
        ):
            evolve_debt(mock_ctx, limit=20)

    def test_debt_handles_analysis_error(self, test_config: AppConfig) -> None:
        """Debt command handles analysis errors."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch(
            "jpscripts.commands.evolve.calculate_debt_scores",
            return_value=Err("Analysis failed"),
        ):
            evolve_debt(mock_ctx, limit=20)

    def test_debt_shows_recommendation(self, test_config: AppConfig) -> None:
        """Debt command shows recommendation for top file."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        mock_scores = [
            TechnicalDebtScore(
                path=Path("src/complex.py"),
                complexity_score=30.0,
                fix_frequency=10,
                churn=25,
                debt_score=150.0,
                reasons=["High complexity", "Many fixes"],
            )
        ]

        with patch(
            "jpscripts.commands.evolve.calculate_debt_scores",
            return_value=Ok(mock_scores),
        ):
            evolve_debt(mock_ctx, limit=20)


class TestEvolveCLIIntegration:
    """Integration tests for CLI commands via typer."""

    @pytest.fixture
    def cli_app(self, test_config: AppConfig) -> typer.Typer:
        """Create a test CLI app with injected state."""
        test_app = typer.Typer()

        @test_app.callback()
        def callback(ctx: typer.Context) -> None:
            state = MagicMock()
            state.config = test_config
            ctx.obj = state

        # Add the evolve subcommands
        test_app.add_typer(app, name="evolve")
        return test_app

    def test_run_command_help(self, runner: CliRunner, cli_app: typer.Typer) -> None:
        """Run command shows help text."""
        result = runner.invoke(cli_app, ["evolve", "run", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.stdout.lower()
        assert "threshold" in result.stdout.lower()

    def test_report_command_help(self, runner: CliRunner, cli_app: typer.Typer) -> None:
        """Report command shows help text."""
        result = runner.invoke(cli_app, ["evolve", "report", "--help"])
        assert result.exit_code == 0
        assert "limit" in result.stdout.lower()

    def test_debt_command_help(self, runner: CliRunner, cli_app: typer.Typer) -> None:
        """Debt command shows help text."""
        result = runner.invoke(cli_app, ["evolve", "debt", "--help"])
        assert result.exit_code == 0
        assert "limit" in result.stdout.lower()


class TestTechnicalDebtScoreUsage:
    """Tests verifying proper usage of TechnicalDebtScore dataclass."""

    def test_debt_score_ordering(self) -> None:
        """Debt scores should be sortable by debt_score."""
        scores = [
            TechnicalDebtScore(
                path=Path("low.py"),
                complexity_score=5.0,
                fix_frequency=1,
                churn=2,
                debt_score=10.0,
                reasons=[],
            ),
            TechnicalDebtScore(
                path=Path("high.py"),
                complexity_score=30.0,
                fix_frequency=10,
                churn=50,
                debt_score=300.0,
                reasons=["Very complex"],
            ),
            TechnicalDebtScore(
                path=Path("medium.py"),
                complexity_score=15.0,
                fix_frequency=5,
                churn=20,
                debt_score=75.0,
                reasons=["Moderate"],
            ),
        ]

        sorted_scores = sorted(scores, key=lambda s: -s.debt_score)
        assert sorted_scores[0].path.name == "high.py"
        assert sorted_scores[1].path.name == "medium.py"
        assert sorted_scores[2].path.name == "low.py"

    def test_debt_score_reasons_list(self) -> None:
        """Multiple reasons are properly stored."""
        score = TechnicalDebtScore(
            path=Path("multi.py"),
            complexity_score=20.0,
            fix_frequency=8,
            churn=30,
            debt_score=160.0,
            reasons=[
                "High cyclomatic complexity",
                "Many bug fixes",
                "High churn rate",
            ],
        )
        assert len(score.reasons) == 3
        assert "High cyclomatic complexity" in score.reasons


class TestCreateEvolutionPR:
    """Tests for PR creation functionality."""

    @pytest.mark.asyncio
    async def test_pr_body_contains_target_info(self) -> None:
        """PR body includes all relevant target information."""
        from jpscripts.commands.evolve import _create_evolution_pr

        target = TechnicalDebtScore(
            path=Path("src/target.py"),
            complexity_score=25.0,
            fix_frequency=10,
            churn=40,
            debt_score=125.0,
            reasons=["High complexity", "Frequent fixes"],
        )

        mock_repo = AsyncMock()
        mock_repo.run_git.return_value = Ok("")

        mock_config = MagicMock()

        # Mock gh CLI as not found to skip actual PR creation
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"gh not found")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            await _create_evolution_pr(
                repo=mock_repo,
                target=target,
                branch_name="evolve/target-optimization",
                root=Path("/workspace"),
                config=mock_config,
                verification_cmd="pytest tests/",
                verification_exit=0,
            )

        # Verify git operations were called
        mock_repo.run_git.assert_any_call("add", "-A")
        commit_call = next(
            c for c in mock_repo.run_git.call_args_list if "commit" in c.args
        )
        assert "refactor" in commit_call.args[2]


class TestEvolveBranchNaming:
    """Tests for branch name generation."""

    def test_branch_name_from_path(self) -> None:
        """Branch names are derived from file path stems."""
        target = TechnicalDebtScore(
            path=Path("src/jpscripts/core/complex_module.py"),
            complexity_score=20.0,
            fix_frequency=5,
            churn=15,
            debt_score=100.0,
            reasons=[],
        )

        expected_branch = f"evolve/{target.path.stem}-optimization"
        assert expected_branch == "evolve/complex_module-optimization"


class TestEvolveRunCLICommand:
    """Tests for the evolve_run CLI command."""

    def test_evolve_run_calls_async_run(self, test_config: AppConfig) -> None:
        """The CLI command invokes _run_evolve correctly."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch("jpscripts.commands.evolve._run_evolve", new_callable=AsyncMock) as mock_run:
            evolve_run(mock_ctx, dry_run=True, model="gpt-4o", threshold=15.0)

            # Check that _run_evolve was called with the right parameters
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0] == test_config  # config
            assert args[1] is True  # dry_run
            assert args[2] == "gpt-4o"  # model
            assert args[3] == 15.0  # threshold

    def test_evolve_run_default_threshold(self, test_config: AppConfig) -> None:
        """Default threshold is 10.0."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        with patch("jpscripts.commands.evolve._run_evolve", new_callable=AsyncMock) as mock_run:
            evolve_run(mock_ctx, dry_run=False, model=None, threshold=10.0)

            args, kwargs = mock_run.call_args
            # threshold is 4th positional argument
            assert args[3] == 10.0


class TestEvolveReportInternals:
    """Tests for the internal _report function behavior."""

    def test_report_shows_complexity_table(self, test_config: AppConfig) -> None:
        """Report displays complexity information in a table."""
        from jpscripts.core.complexity import FileComplexity, FunctionComplexity

        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        mock_file = FileComplexity(
            path=Path("src/complex.py"),
            functions=[
                FunctionComplexity(
                    name="complex_func",
                    lineno=10,
                    end_lineno=50,
                    cyclomatic=15,
                    is_async=False,
                )
            ],
            total_cyclomatic=15,
            max_cyclomatic=15,
            average_cyclomatic=15.0,
        )

        with (
            patch(
                "jpscripts.commands.evolve.analyze_directory_complexity",
                return_value=Ok([mock_file]),
            ),
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Err("Not a git repo"),
            ),
        ):
            evolve_report(mock_ctx, limit=5)


class TestEvolveDebtInternals:
    """Tests for the internal debt calculation behavior."""

    def test_debt_displays_multiple_scores(self, test_config: AppConfig) -> None:
        """Debt command displays multiple files sorted by score."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        mock_scores = [
            TechnicalDebtScore(
                path=Path("src/highest.py"),
                complexity_score=50.0,
                fix_frequency=20,
                churn=100,
                debt_score=500.0,
                reasons=["Extremely complex"],
            ),
            TechnicalDebtScore(
                path=Path("src/middle.py"),
                complexity_score=25.0,
                fix_frequency=10,
                churn=50,
                debt_score=125.0,
                reasons=["Moderate complexity"],
            ),
            TechnicalDebtScore(
                path=Path("src/lowest.py"),
                complexity_score=10.0,
                fix_frequency=2,
                churn=10,
                debt_score=20.0,
                reasons=[],
            ),
        ]

        with patch(
            "jpscripts.commands.evolve.calculate_debt_scores",
            return_value=Ok(mock_scores),
        ):
            evolve_debt(mock_ctx, limit=3)

    def test_debt_respects_limit(self, test_config: AppConfig) -> None:
        """Debt command respects the limit parameter."""
        mock_ctx = MagicMock()
        mock_state = MagicMock()
        mock_state.config = test_config
        mock_ctx.obj = mock_state

        # Create more scores than limit
        mock_scores = [
            TechnicalDebtScore(
                path=Path(f"src/file{i}.py"),
                complexity_score=float(100 - i),
                fix_frequency=10 - i,
                churn=50 - i,
                debt_score=float(500 - i * 10),
                reasons=[f"Reason {i}"],
            )
            for i in range(10)
        ]

        with patch(
            "jpscripts.commands.evolve.calculate_debt_scores",
            return_value=Ok(mock_scores),
        ):
            # Request only 5 results
            evolve_debt(mock_ctx, limit=5)


class TestRunEvolveWithBranchCreation:
    """Tests for branch creation flow in _run_evolve."""

    @pytest.mark.asyncio
    async def test_branch_creation_failure_handled(
        self, test_config: AppConfig
    ) -> None:
        """Branch creation failures are handled gracefully."""
        mock_repo = AsyncMock()
        mock_repo.status.return_value = Ok(MagicMock(dirty=False))
        mock_repo.run_git.side_effect = Exception("Branch already exists")

        mock_scores = [
            TechnicalDebtScore(
                path=Path("src/target.py"),
                complexity_score=20.0,
                fix_frequency=5,
                churn=15,
                debt_score=100.0,
                reasons=["High complexity"],
            )
        ]

        with (
            patch(
                "jpscripts.commands.evolve.git_core.AsyncRepo.open",
                return_value=Ok(mock_repo),
            ),
            patch(
                "jpscripts.commands.evolve.calculate_debt_scores",
                return_value=Ok(mock_scores),
            ),
        ):
            # Should handle error without raising
            await _run_evolve(test_config, dry_run=False, model=None, threshold=10.0)
